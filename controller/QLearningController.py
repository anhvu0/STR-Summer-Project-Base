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
from core.route_loop_safety import transition_signal, would_worsen_distance

def parse_sumocfg(sumocfg_path):
    dom = parse(sumocfg_path)
    net_file = dom.getElementsByTagName('net-file')[0].attributes['value'].nodeValue
    return os.path.join(os.path.dirname(sumocfg_path), net_file)
net_path = parse_sumocfg("./configurations/myconfig.sumocfg")


class QLearningPolicy(RouteController):
    def __init__(self, vehicles, connection_info, model_file, net_xml_file = net_path):
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
        self._metrics = {
            "decisions": 0,
            "overrides": 0,
            "loop_overrides": 0,
            "distance_overrides": 0,
            "impossible_action_overrides": 0,
            "deadend_overrides": 0,
            "decision_committed_skips": 0,
        }
        self._last_metrics_snapshot = None
        # Cache for shortest-path distances (edge_id, dest_id) -> cost
        # How many actions to plan ahead each time
        self.decision_horizon = 1
        self.loop_window = 10
        self.loop_repeat_threshold = 2
        self.score_slack = 30.0
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
        edge_out_degree = {edge: self._edge_out_degree(edge) for edge in set(recent_history) | {next_edge}}
        signals = transition_signal(recent_history, next_edge, edge_out_degree=edge_out_degree)
        current_dist = self._dist_to_dest(current_edge, destination)
        next_dist = self._dist_to_dest(next_edge, destination)
        dist_worsen = would_worsen_distance(current_dist, next_dist, slack=self.score_slack)
        trap_like = (
            next_edge != destination
            and self._edge_out_degree(next_edge) <= 1
            and len(recent_history) > 0
            and recent_history[-1] == current_edge
        )
        score = 0
        if signals["short_cycle"]:
            score += 5
        if signals["aba_bounce"]:
            score += 5
        if signals["dead_end_reentry"]:
            score += 3
        if dist_worsen:
            score += 2
        if trap_like:
            score += 4
        return score, signals, dist_worsen, trap_like

    def _finalize_commitment(self, vehicle):
        vid = vehicle.vehicle_id
        pending = self._pending_decisions.get(vid)
        if not pending:
            return
        if vehicle.current_edge == pending.decision_edge:
            self._metrics["decision_committed_skips"] += 1
            return
        self._pending_decisions.pop(vid, None)
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
                action_idx = self.act(state, available_actions=context.available_actions)
                self._metrics["decisions"] += 1

            if action_idx not in context.available_actions:
                self._metrics["impossible_action_overrides"] += 1
                if not context.available_actions:
                    continue
                action_idx = context.available_actions[0]
                self._metrics["overrides"] += 1

            selected_next_edge = self.decision_engine.get_next_edge(start_edge, action_idx)
            if selected_next_edge is None:
                continue

            if context.available_actions:
                best_action = action_idx
                best_score = None
                best_reason = None
                for candidate in context.available_actions:
                    candidate_next = self.decision_engine.get_next_edge(start_edge, candidate)
                    if candidate_next is None:
                        continue
                    score, signals, dist_worsen, trap_like = self._action_safety_score(
                        start_edge, candidate_next, vehicle.destination, self._recent_edges[vid]
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_action = candidate
                        best_reason = (signals, dist_worsen, trap_like)
                if best_action != action_idx:
                    self._metrics["overrides"] += 1
                    signals, dist_worsen, trap_like = best_reason
                    if signals["short_cycle"] or signals["aba_bounce"] or signals["dead_end_reentry"]:
                        self._metrics["loop_overrides"] += 1
                    if dist_worsen:
                        self._metrics["distance_overrides"] += 1
                    if trap_like:
                        self._metrics["deadend_overrides"] += 1
                    action_idx = best_action
                    selected_next_edge = self.decision_engine.get_next_edge(start_edge, action_idx)

            lane_change_requested, lane_change_ok = self.decision_engine.try_request_lane_change(context, action_idx)
            if lane_change_requested and not lane_change_ok:
                self._metrics["overrides"] += 1

            fragment, local_target, frag_error = self.decision_engine.build_route_fragment(
                start_edge, action_idx, vehicle.destination
            )
            if frag_error or local_target is None:
                self._metrics["overrides"] += 1
                continue

            next_edge = self.decision_engine.get_next_edge(start_edge, action_idx)
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
                    destination=vehicle.destination,
                    context=context,
                    lane_change_requested=lane_change_requested,
                )
            local_targets[vehicle.vehicle_id] = local_target

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
    def act(self, state, available_actions=None):
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
