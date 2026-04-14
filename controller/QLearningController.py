from controller.RouteController import RouteController
from core.Util import ConnectionInfo, Vehicle
from keras.models import load_model
import numpy as np
import traci
import sumolib
import math
from collections import deque

from xml.dom.minidom import parse
import os
from core.junction_decision_engine import JunctionDecisionEngine, PendingDecision
from core.route_loop_safety import score_transition_risk

def parse_sumocfg(sumocfg_path):
    dom = parse(sumocfg_path)
    net_file = dom.getElementsByTagName('net-file')[0].attributes['value'].nodeValue
    return os.path.join(os.path.dirname(sumocfg_path), net_file)
net_path = parse_sumocfg("./configurations/myconfig.sumocfg")


class QLearningPolicy(RouteController):
    def __init__(
        self,
        vehicles,
        connection_info,
        model_file,
        net_xml_file=net_path,
        enable_safety_action_filter=True,
        enable_safety_q_penalty=True,
        enable_lane_change_recheck_before_fallback=True,
    ):
        super().__init__(connection_info)
        self.model = load_model(model_file)
        self.model_state_size = int(self.model.input_shape[-1])
        self.vehicles = vehicles
        self.net = sumolib.net.readNet(net_xml_file)
        self.decision_engine = JunctionDecisionEngine(connection_info, self.net, self.direction_choices)
        self._visit_count = {}
        self._best_dist = {}
        self._recent_edges = {}
        self._pending_decisions = {}
        self._lane_change_deferrals = {}
        self._metrics = {
            "decisions": 0,
            "overrides": 0,
            "loop_overrides": 0,
            "distance_overrides": 0,
            "impossible_action_overrides": 0,
            "deadend_overrides": 0,
            "decision_committed_skips": 0,
            "pending_decision_timeouts": 0,
            "fallback_to_lane_feasible_now": 0,
            "deferred_lane_change_actions": 0,
        }
        self._last_metrics_snapshot = None
        # Cache for shortest-path distances (edge_id, dest_id) -> cost
        # How many actions to plan ahead each time
        self.decision_horizon = 1
        self.loop_window = 10
        self.loop_repeat_threshold = 2
        self.score_slack = 30.0
        self.enable_safety_action_filter = bool(enable_safety_action_filter)
        self.enable_safety_q_penalty = bool(enable_safety_q_penalty)
        self.safety_q_penalty_scale = 1.5
        self.enable_lane_change_recheck_before_fallback = bool(enable_lane_change_recheck_before_fallback)
        self.deadline_deficit_override_slack = 2.0
        self.distance_tiebreak_scale = 0.05
        self.edge_embedding_dim = 8
        self.local_congestion_k = 6
        self.compact_state_size = (2 * self.edge_embedding_dim) + 24 + 1 + 3 + 3 + self.local_congestion_k
        self.legacy_state_size = 2 + 6 + 3 + 3 + len(self.connection_info.edge_list)
        self.use_compact_state = (self.model_state_size == self.compact_state_size)
        self.direction_mask_start = (2 * self.edge_embedding_dim) if self.use_compact_state else 2
        self._init_edge_embeddings(seed=1337)

    def _init_edge_embeddings(self, seed=1337):
        rng = np.random.default_rng(seed)
        self._edge_embeddings = {}
        for edge_id in self.connection_info.edge_list:
            emb = rng.normal(loc=0.0, scale=0.1, size=self.edge_embedding_dim).astype(np.float32)
            self._edge_embeddings[edge_id] = emb

    def _get_edge_embedding(self, edge_id):
        return self._edge_embeddings.get(
            edge_id,
            np.zeros(self.edge_embedding_dim, dtype=np.float32),
        )

    def _local_congestion_features(self, edge_id):
        lengths = self.connection_info.edge_length_dict
        current_density = traci.edge.getLastStepVehicleNumber(edge_id) / max(lengths.get(edge_id, 5.0), 5.0)
        outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {})
        outgoing_densities = [
            traci.edge.getLastStepVehicleNumber(next_edge) / max(lengths.get(next_edge, 5.0), 5.0)
            for next_edge in outgoing.values()
        ]

        all_densities = [
            traci.edge.getLastStepVehicleNumber(edge) / max(lengths.get(edge, 5.0), 5.0)
            for edge in self.connection_info.edge_list
        ]
        mean_global = float(np.mean(all_densities)) if len(all_densities) > 0 else 0.0
        std_global = float(np.std(all_densities)) if len(all_densities) > 0 else 0.0
        mean_out = float(np.mean(outgoing_densities)) if outgoing_densities else current_density
        max_out = float(np.max(outgoing_densities)) if outgoing_densities else current_density
        min_out = float(np.min(outgoing_densities)) if outgoing_densities else current_density

        return [
            current_density,
            mean_out,
            max_out,
            min_out,
            current_density - mean_global,
            std_global,
        ]


    


    def _compute_deadline_features(self, vehicle_id):
        vehicle_obj = self.vehicles.get(str(vehicle_id))
        if vehicle_obj is None:
            return [0.0, 0.0, 0.0]

        now = traci.simulation.getTime()
        deadline_window = max(float(vehicle_obj.deadline) - float(vehicle_obj.start_time), 1.0)
        time_left = max(float(vehicle_obj.deadline) - float(now), 0.0)
        elapsed = max(float(now) - float(vehicle_obj.start_time), 0.0)
        urgency = 1.0 - min(time_left / deadline_window, 1.0)

        return [
            min(time_left / deadline_window, 1.0),
            min(elapsed / deadline_window, 1.0),
            urgency,
        ]

    #-----------------------DEBUGGING-------------------------------------
    def _dist_to_dest(self, edge_id, dest_id):
        try:
            from_edge = self.net.getEdge(edge_id)
            to_edge = self.net.getEdge(dest_id)
            path_edges, path_cost = self.net.getShortestPath(from_edge, to_edge, vClass="passenger")
            if path_edges is None:
                return float("inf")
            return path_cost
        except Exception:
            return float("inf")

    def _estimate_eta(self, edge_id, dest_id):
        dist = self._dist_to_dest(edge_id, dest_id)
        if not np.isfinite(dist):
            return float("inf")
        return float(dist) / 8.0

    def _edge_out_degree(self, edge_id):
        outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {})
        return len(outgoing)

    def _action_safety_score(self, current_edge, next_edge, destination, recent_history):
        edge_out_degree = {edge: self._edge_out_degree(edge) for edge in set(recent_history) | {next_edge, current_edge}}
        current_dist = self._dist_to_dest(current_edge, destination)
        next_dist = self._dist_to_dest(next_edge, destination)
        score_card = score_transition_risk(
            history=recent_history,
            current_edge=current_edge,
            next_edge=next_edge,
            destination=destination,
            current_distance=current_dist,
            next_distance=next_dist,
            edge_out_degree=edge_out_degree,
            distance_slack=self.score_slack,
        )
        return (
            score_card["score"],
            {
                "short_cycle": score_card["short_cycle"],
                "aba_bounce": score_card["aba_bounce"],
                "dead_end_reentry": score_card["dead_end_reentry"],
            },
            score_card["distance_worsen"],
            score_card["trap_like"],
        )

    def _safety_rank_actions(self, current_edge, destination, available_actions, recent_history):
        scored = []
        for action in available_actions:
            next_edge = self.decision_engine.get_next_edge(current_edge, action)
            if next_edge is None:
                continue
            score, signals, dist_worsen, trap_like = self._action_safety_score(
                current_edge, next_edge, destination, recent_history
            )
            scored.append((action, float(score), signals, dist_worsen, trap_like))
        if not scored:
            return list(available_actions), {}, False, False, False

        scores = {action: score for action, score, *_ in scored}
        min_score = min(scores.values())
        safest = [action for action, score in scores.items() if score <= min_score + 1e-6]
        risky_only = len(safest) == len(scores) and min_score > 0.0
        filtered = False
        downranked = False
        if self.enable_safety_action_filter and len(safest) > 0 and len(safest) < len(scores):
            filtered = True
            return safest, scores, filtered, downranked, risky_only
        if self.enable_safety_q_penalty and any(score > min_score for score in scores.values()):
            downranked = True
        return list(available_actions), scores, filtered, downranked, risky_only

    def _finalize_commitment(self, vehicle):
        vid = vehicle.vehicle_id
        pending = self._pending_decisions.get(vid)
        if not pending:
            return
        step = int(traci.simulation.getTime())
        if vehicle.current_edge == pending.decision_edge:
            if self.decision_engine.should_timeout_pending(pending, step):
                self._pending_decisions.pop(vid, None)
                self._metrics["pending_decision_timeouts"] += 1
                return
            self._metrics["decision_committed_skips"] += 1
            return
        self._pending_decisions.pop(vid, None)
        self._lane_change_deferrals[vid] = 0
    #----------------------------------------------------------------------


    def make_decisions(self, vehicles, connection_info: ConnectionInfo):
        local_targets = {}

        if not hasattr(self, "_debug_net_checked"):
            self._debug_net_checked = True
            # print("[DEBUG] has self.net:", hasattr(self, "net"))
            # print("[DEBUG] self.net type:", type(self.net))

        for vehicle in vehicles:
            start_edge = vehicle.current_edge
            if vehicle.destination == start_edge:
                continue

            vid = vehicle.vehicle_id
            if vid not in self._recent_edges:
                self._recent_edges[vid] = deque(maxlen=self.loop_window)
            self._visit_count.setdefault(vid, {})
            self._best_dist.setdefault(vid, float("inf"))
            self._finalize_commitment(vehicle)

            if vid in self._pending_decisions:
                # Keep commitment semantics aligned with training:
                # one decision is open until the vehicle exits the decision edge.
                continue

            step = int(traci.simulation.getTime())
            context = self.decision_engine.build_context(str(vid), start_edge, vehicle.destination, step)

            # Skip non-meaningful junction points; apply forced action directly.
            if context.forced_action is not None:
                action_idx = context.forced_action
            elif not self.decision_engine.is_decision_open(context):
                continue
            else:
                state = self.getState(vid, start_edge, vehicle.destination, context=context)
                candidate_actions, safety_scores, _, apply_q_penalty, _ = self._safety_rank_actions(
                    start_edge,
                    vehicle.destination,
                    context.available_actions,
                    self._recent_edges[vid],
                )
                action_idx = self.act(state, available_actions=candidate_actions, safety_scores=safety_scores if apply_q_penalty else None)
                self._metrics["decisions"] += 1

            if action_idx not in context.available_actions:
                self._metrics["impossible_action_overrides"] += 1
                continue

            selected_next_edge = self.decision_engine.get_next_edge(start_edge, action_idx)
            if selected_next_edge is None:
                continue

            lane_change_requested, lane_change_ok = self.decision_engine.try_request_lane_change(context, action_idx)
            if lane_change_requested:
                if lane_change_ok:
                    self._metrics["deferred_lane_change_actions"] += 1
                    self._lane_change_deferrals[vid] = self._lane_change_deferrals.get(vid, 0) + 1
                    if self._lane_change_deferrals[vid] < self.decision_engine.lane_change_defer_limit:
                        continue
                    fallback_actions = None
                    if self.enable_lane_change_recheck_before_fallback:
                        retry_context = self.decision_engine.build_context(str(vid), start_edge, vehicle.destination, step)
                        if action_idx in retry_context.available_actions:
                            context = retry_context
                        else:
                            fallback_actions = self.decision_engine.lane_feasible_fallback_actions(context, blocked_action=action_idx)
                    else:
                        fallback_actions = self.decision_engine.lane_feasible_fallback_actions(context, blocked_action=action_idx)
                    if fallback_actions is None:
                        pass
                    elif not fallback_actions:
                        continue
                    else:
                        state = self.getState(vid, start_edge, vehicle.destination, context=context)
                        action_idx = self.act(state, available_actions=fallback_actions)
                        selected_next_edge = self.decision_engine.get_next_edge(start_edge, action_idx)
                        if selected_next_edge is None:
                            continue
                        self._metrics["fallback_to_lane_feasible_now"] += 1
                else:
                    fallback_actions = self.decision_engine.lane_feasible_fallback_actions(context, blocked_action=action_idx)
                    if not fallback_actions:
                        continue
                    state = self.getState(vid, start_edge, vehicle.destination, context=context)
                    action_idx = self.act(state, available_actions=fallback_actions)
                    selected_next_edge = self.decision_engine.get_next_edge(start_edge, action_idx)
                    if selected_next_edge is None:
                        continue
                    self._metrics["fallback_to_lane_feasible_now"] += 1
            else:
                self._lane_change_deferrals[vid] = 0

            full_route, committed_next_edge, apply_error = self.decision_engine.apply_route_decision(
                str(vid),
                start_edge,
                action_idx,
                vehicle.destination,
            )
            if apply_error:
                self._metrics["overrides"] += 1
                continue

            next_edge = committed_next_edge
            if next_edge:
                self._recent_edges[vid].append(next_edge)
                self._visit_count[vid][next_edge] = self._visit_count[vid].get(next_edge, 0) + 1
                self._best_dist[vid] = min(self._best_dist[vid], self._dist_to_dest(next_edge, vehicle.destination))
                self._pending_decisions[vid] = PendingDecision(
                    state=None,
                    intended_action=action_idx,
                    intended_next_edge=next_edge,
                    decision_edge=start_edge,
                    decision_step=step,
                    last_credit_edge=start_edge,
                    last_credit_step=step,
                    destination=vehicle.destination,
                    context=context,
                    lane_change_requested=lane_change_requested,
                    route_fragment=list(full_route[1:]) if full_route else [],
                )
                self._lane_change_deferrals[vid] = 0
            # Route already committed directly via shared apply_route_decision.

        if self._metrics["decisions"] > 0:
            snapshot = (
                self._metrics["decisions"],
                self._metrics["overrides"],
                self._metrics["loop_overrides"],
                self._metrics["distance_overrides"],
                self._metrics["impossible_action_overrides"],
                self._metrics["deadend_overrides"],
            )
            if snapshot == self._last_metrics_snapshot:
                return local_targets

            self._last_metrics_snapshot = snapshot
            ratio = self._metrics["overrides"] / float(self._metrics["decisions"])
            # print(
            #     "[Q-METRICS] decisions={} overrides={} override_ratio={:.2%} "
            #     "loop_overrides={} distance_overrides={} impossible_action_overrides={}".format(
            #         self._metrics["decisions"],
            #         self._metrics["overrides"],
            #         ratio,
            #         self._metrics["loop_overrides"],
            #         self._metrics["distance_overrides"],
            #         self._metrics["impossible_action_overrides"],
            #     )
            # )

        return local_targets




    # this function reacheds the Neural Network trained before and let it make a decision for the situation now
    def act(self, state, available_actions=None, safety_scores=None):
        act_values = self.model.predict(state, verbose=0)[0]
        if available_actions is None:
            mask_start = self.direction_mask_start + 18 if self.use_compact_state else self.direction_mask_start
            available = [i for i, v in enumerate(state[0][mask_start:mask_start + 6]) if v > 0.5]
        else:
            available = list(available_actions)
        if not available:
            return int(np.argmax(act_values))
        masked = np.full_like(act_values, -1e9)
        masked[available] = act_values[available]
        if safety_scores:
            for action in available:
                masked[action] -= self.safety_q_penalty_scale * float(safety_scores.get(action, 0.0))
        return int(np.argmax(masked))

    # this function gives the current state of the vehicle based on the state size
    def getState(self, vehicle_id, edge_now, destination_edge, context=None):
        en = edge_now
        state = []
        if self.use_compact_state:
            state.extend(self._get_edge_embedding(en).tolist())
            state.extend(self._get_edge_embedding(destination_edge).tolist())
        else:
            state.append(self.connection_info.edge_index_dict[en])
            state.append(self.connection_info.edge_index_dict[destination_edge])
        if context is None:
            context = self.decision_engine.build_context(str(vehicle_id), en, destination_edge, int(traci.simulation.getTime()))
        edge_mask, lane_mask, reach_mask, avail_mask = self.decision_engine.direction_masks(context)
        if self.use_compact_state:
            state.extend(edge_mask)
            state.extend(lane_mask)
            state.extend(reach_mask)
            state.extend(avail_mask)
            state.append(1.0 if context.commit_window else 0.0)
        else:
            for c in self.direction_choices:
                state.append(1 if c in self.connection_info.outgoing_edges_dict[en].keys() else 0)
        # put the congestion ratio of all edges into the state.

        lane_idx_norm = 0.0
        lane_count_norm = 0.0
        dist_to_end_norm = 0.0
        try:
            lane_idx = context.lane_index
            lane_count = max(context.lane_count, 1)
            dist_to_end = max(context.dist_to_end, 0.0)

            lane_idx_norm = lane_idx / max(lane_count - 1, 1)
            lane_count_norm = min(lane_count, 6) / 6.0
            dist_to_end_norm = min(dist_to_end, 200.0) / 200.0
        except traci.TraCIException:
            # Vehicle may have arrived/teleported between steps. Keep neutral defaults.
            pass

        state.extend([lane_idx_norm, lane_count_norm, dist_to_end_norm])
        state.extend(self._compute_deadline_features(vehicle_id))

        if self.use_compact_state:
            state.extend(self._local_congestion_features(en))
        else:
            for edge_now in self.connection_info.edge_list:
                car_num = traci.edge.getLastStepVehicleNumber(edge_now)
                density = car_num / self.connection_info.edge_length_dict[edge_now]
                state.append(density)

        state = np.reshape(state, [1, len(state)])
        return state
