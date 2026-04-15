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
from core.junction_decision_engine import JunctionDecisionEngine, PendingDecision, VehicleSnapshot
from core.route_loop_safety import transition_signal, would_worsen_distance

def parse_sumocfg(sumocfg_path):
    dom = parse(sumocfg_path)
    net_file = dom.getElementsByTagName('net-file')[0].attributes['value'].nodeValue
    return os.path.join(os.path.dirname(sumocfg_path), net_file)
net_path = parse_sumocfg("./configurations/myconfig.sumocfg")
MAX_SIMULATION_STEPS = 2000


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
        self._lane_change_deferrals = {}
        self._lane_change_cooldown = {}
        self._edge_commitments = deque(maxlen=200)
        self._pending_progress = {}
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
            "lane_change_observe_started": 0,
            "lane_change_observe_success": 0,
            "lane_change_observe_abort_no_progress": 0,
            "lane_change_observe_abort_commit_window": 0,
            "same_edge_pending_released_no_progress": 0,
            "cooldown_replans_blocked": 0,
            "loop_override_count": 0,
            "dead_end_reentry_override_count": 0,
            "pending_preemptions": 0,
            "trapped_replan_skips": 0,
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

    def _edge_density(self, edge_id):
        return traci.edge.getLastStepVehicleNumber(edge_id) / max(
            self.connection_info.edge_length_dict.get(edge_id, 5.0),
            5.0,
        )

    def _congestion_score(self, current_edge, next_edge):
        current_density = self._edge_density(current_edge)
        next_density = self._edge_density(next_edge) if next_edge else current_density
        return (2.0 * current_density) + (2.5 * next_density)

    def _branch_pressure_penalty(self, next_edge):
        if not next_edge:
            return 0.0
        count = sum(1 for edge in self._edge_commitments if edge == next_edge)
        return min(float(count) * 0.45, 6.0)

    def _record_commitment(self, next_edge):
        if next_edge:
            self._edge_commitments.append(next_edge)

    def _physically_recoverable_actions(self, context):
        lane_now = set(context.lane_feasible_now_actions)
        reachable = set(context.reachable_with_lane_change_actions)
        recoverable = []
        for action in context.available_actions:
            if action in lane_now:
                recoverable.append(action)
                continue
            if context.commit_window:
                continue
            if action not in reachable:
                continue
            required_shift = int(context.required_lane_shift.get(action, 99))
            if required_shift >= 99:
                continue
            lane_change_budget = max(float(context.dist_to_end) - float(self.decision_engine.commit_min_distance), 0.0)
            required_budget = required_shift * float(self.decision_engine.lane_change_margin_m)
            if lane_change_budget >= required_budget and float(context.speed) >= 1.0:
                recoverable.append(action)
        return sorted(set(recoverable))

    def _should_preempt_pending_for_congestion(self, pending, context, step):
        pending_age = self.decision_engine.pending_age_steps(pending, step)
        progress = self._pending_progress.setdefault(
            pending.context.vehicle_id,
            {"edge": context.edge_id, "edge_steps": 0, "best_dist_to_end": float(context.dist_to_end)},
        )
        if progress["edge"] != context.edge_id:
            progress["edge"] = context.edge_id
            progress["edge_steps"] = 0
            progress["best_dist_to_end"] = float(context.dist_to_end)
        progress["edge_steps"] += 1
        progress["best_dist_to_end"] = min(progress["best_dist_to_end"], float(context.dist_to_end))
        dist_gain = max(progress["best_dist_to_end"] - float(context.dist_to_end), 0.0)

        recoverable = self._physically_recoverable_actions(context)
        feasible_escape = [a for a in recoverable if a != pending.intended_action]
        if not feasible_escape:
            return False, True
        if context.commit_window and pending.intended_action not in context.lane_feasible_now_actions:
            return True, False

        blocked = (
            progress["edge_steps"] >= 4
            and float(context.speed) <= 0.6
            and self._edge_density(context.edge_id) >= 0.18
            and dist_gain <= 2.0
        )
        stale = pending_age >= max(4, self.decision_engine.pending_progress_timeout_steps // 2) and dist_gain <= 1.0
        return (blocked or stale), False

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


    


    def _compute_objective_features(self, vehicle_id, edge_id, destination_edge, step=None):
        vehicle_obj = self.vehicles.get(str(vehicle_id))
        if vehicle_obj is None or edge_id is None:
            return [0.0, 0.0, 0.0]

        now = float(traci.simulation.getTime()) if step is None else float(step)
        elapsed = max(now - float(vehicle_obj.start_time), 0.0)
        remaining_eta = self._estimate_eta(edge_id, destination_edge)
        density = traci.edge.getLastStepVehicleNumber(edge_id) / max(
            self.connection_info.edge_length_dict.get(edge_id, 5.0),
            5.0,
        )

        return [
            min(elapsed / float(MAX_SIMULATION_STEPS), 1.0),
            min(float(remaining_eta) / float(MAX_SIMULATION_STEPS), 1.0) if math.isfinite(remaining_eta) else 1.0,
            min(float(density), 1.0),
        ]

    def _policy_action_candidates(self, context, recent_history, cooldown_active, destination):
        available_actions = list(context.available_actions)
        if not available_actions:
            return []

        lane_now = set(context.lane_feasible_now_actions)
        recent_history = list(recent_history or [])

        commit_distance = max(
            float(self.decision_engine.commit_min_distance),
            float(context.speed) * float(self.decision_engine.commit_time_s),
        )
        extra_buffer = max(10.0, 0.5 * float(self.decision_engine.lane_change_margin_m))
        comfortable_dist_threshold = commit_distance + extra_buffer

        safe_lane_now_actions = []
        strict_non_lane_actions = []
        filtered_available_actions = []

        for action in available_actions:
            safe_ok, _ = self.decision_engine.prefilter_action_for_loops(
                context=context,
                action_idx=action,
                destination=destination,
                recent_history=recent_history,
                distance_fn=self._dist_to_dest,
            )
            if not safe_ok:
                continue
            filtered_available_actions.append(action)
            if action in lane_now:
                safe_lane_now_actions.append(action)
                continue

            if cooldown_active:
                continue
            if context.commit_window:
                continue
            if float(context.speed) < 1.2:
                continue
            if int(context.required_lane_shift.get(action, 99)) != 1:
                continue
            if float(context.dist_to_end) <= comfortable_dist_threshold:
                continue
            strict_non_lane_actions.append(action)

        if safe_lane_now_actions:
            return sorted(set(safe_lane_now_actions))
        if strict_non_lane_actions:
            return sorted(set(strict_non_lane_actions))
        if filtered_available_actions:
            return sorted(set(filtered_available_actions))
        return available_actions

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
        relevant_edges = set(recent_history) | {current_edge, next_edge}
        edge_out_degree = {edge: self._edge_out_degree(edge) for edge in relevant_edges}
        edge_distance_lookup = {edge: self._dist_to_dest(edge, destination) for edge in relevant_edges}
        signals = transition_signal(
            recent_history,
            next_edge,
            edge_out_degree=edge_out_degree,
            edge_distance_lookup=edge_distance_lookup,
            progress_slack=self.score_slack,
        )
        current_dist = self._dist_to_dest(current_edge, destination)
        next_dist = self._dist_to_dest(next_edge, destination)
        dist_worsen = would_worsen_distance(current_dist, next_dist, slack=self.score_slack)
        trap_like = (
            next_edge != destination
            and self._edge_out_degree(next_edge) == 0
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
        if signals.get("long_horizon_loop"):
            score += 6
        if signals.get("revisit_without_progress"):
            score += 6
        if dist_worsen:
            score += 2
        if trap_like:
            score += 4
        return score, signals, dist_worsen, trap_like

    def _finalize_commitment(self, vehicle):
        vid = vehicle.vehicle_id
        pending = self._pending_decisions.get(vid)
        if not pending:
            self._pending_progress.pop(vid, None)
            return
        step = int(traci.simulation.getTime())
        snapshot = self._snapshot_vehicle(vid, vehicle.current_edge, step)
        if vehicle.current_edge == pending.decision_edge:
            if snapshot is None:
                return
            context = self.decision_engine.build_context(str(vid), vehicle.current_edge, vehicle.destination, step, snapshot=snapshot)
            if self.decision_engine.should_timeout_pending(pending, step, max_age_steps=self.decision_engine.pending_progress_timeout_steps):
                self._pending_decisions.pop(vid, None)
                self._metrics["pending_decision_timeouts"] += 1
                self._metrics["same_edge_pending_released_no_progress"] += 1
                self._lane_change_cooldown[(vid, vehicle.current_edge)] = step + self.decision_engine.cooldown_steps
                self._pending_progress.pop(vid, None)
                return
            if context.commit_window and pending.intended_action not in context.lane_feasible_now_actions:
                self._pending_decisions.pop(vid, None)
                self._metrics["same_edge_pending_released_no_progress"] += 1
                self._lane_change_cooldown[(vid, vehicle.current_edge)] = step + self.decision_engine.cooldown_steps
                self._pending_progress.pop(vid, None)
                return
            self._metrics["decision_committed_skips"] += 1
            return
        self._pending_decisions.pop(vid, None)
        self._lane_change_deferrals[vid] = 0
        self._pending_progress.pop(vid, None)

    def _snapshot_vehicle(self, vid, edge_id, step):
        try:
            lane_id = traci.vehicle.getLaneID(vid)
            lane_index = int(traci.vehicle.getLaneIndex(vid))
            lane_position = float(traci.vehicle.getLanePosition(vid))
            lane_length = float(traci.lane.getLength(lane_id))
            lane_count = max(len(self.connection_info.edge_lane_ids.get(edge_id, [])), 1)
            speed = max(float(traci.vehicle.getSpeed(vid)), 0.0)
            dist_to_end = max(lane_length - lane_position, 0.0)
            return VehicleSnapshot(
                vehicle_id=str(vid), step=int(step), edge_id=edge_id, lane_id=lane_id,
                lane_index=lane_index, lane_count=lane_count, lane_position=lane_position,
                lane_length=lane_length, dist_to_end=dist_to_end, speed=speed,
            )
        except traci.TraCIException:
            return None
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
            step = int(traci.simulation.getTime())
            if vid not in self._recent_edges:
                self._recent_edges[vid] = deque(maxlen=self.loop_window)
            self._recent_edges[vid].append(start_edge)
            self._visit_count.setdefault(vid, {})
            self._best_dist.setdefault(vid, float("inf"))
            self._visit_count[vid][start_edge] = self._visit_count[vid].get(start_edge, 0) + 1
            self._best_dist[vid] = min(self._best_dist[vid], self._dist_to_dest(start_edge, vehicle.destination))
            self._finalize_commitment(vehicle)

            if vid in self._pending_decisions:
                pending = self._pending_decisions[vid]
                phase = pending.metadata.get("phase", "route_pending")
                if phase == "observe_lane_change":
                    obs_snapshot = self._snapshot_vehicle(vid, start_edge, step)
                    if obs_snapshot is None:
                        continue
                    obs_context = self.decision_engine.build_context(str(vid), start_edge, vehicle.destination, step, snapshot=obs_snapshot)
                    status, reason = self.decision_engine.evaluate_lane_change_observation(
                        pending.metadata, obs_context, pending.intended_action
                    )
                    if status == "continue":
                        continue
                    self._pending_decisions.pop(vid, None)
                    if status == "success":
                        self._metrics["lane_change_observe_success"] += 1
                        context = obs_context
                        action_idx = pending.intended_action
                        selected_next_edge = self.decision_engine.get_next_edge(start_edge, action_idx)
                        if selected_next_edge is None:
                            continue
                        full_route, committed_next_edge, apply_error = self.decision_engine.apply_route_decision(
                            str(vid), start_edge, action_idx, vehicle.destination
                        )
                        if apply_error:
                            self._metrics["overrides"] += 1
                            continue
                        self._pending_decisions[vid] = PendingDecision(
                            state=None,
                            intended_action=action_idx,
                            intended_next_edge=committed_next_edge,
                            decision_edge=start_edge,
                            decision_step=step,
                            last_credit_edge=start_edge,
                            last_credit_step=step,
                            destination=vehicle.destination,
                            context=context,
                            lane_change_requested=True,
                            route_fragment=list(full_route[1:]) if full_route else [],
                            metadata={"phase": "route_pending"},
                        )
                        continue
                    if reason == "commit_window":
                        self._metrics["lane_change_observe_abort_commit_window"] += 1
                    else:
                        self._metrics["lane_change_observe_abort_no_progress"] += 1
                    self._lane_change_cooldown[(vid, start_edge)] = step + self.decision_engine.cooldown_steps
                    fallback_actions = self.decision_engine.ranked_fallback_actions(
                        context=obs_context,
                        destination=vehicle.destination,
                        recent_history=list(self._recent_edges.get(vid, deque(maxlen=self.loop_window))),
                        blocked_action=pending.intended_action,
                        distance_fn=self._dist_to_dest,
                        congestion_score_fn=self._congestion_score,
                        branch_pressure_fn=self._branch_pressure_penalty,
                    )
                    recoverable = set(self._physically_recoverable_actions(obs_context))
                    fallback_actions = [a for a in fallback_actions if a in recoverable]
                    if not fallback_actions:
                        continue
                    action_idx = fallback_actions[0]
                    selected_next_edge = self.decision_engine.get_next_edge(start_edge, action_idx)
                    if selected_next_edge is None:
                        continue
                    self._metrics["fallback_to_lane_feasible_now"] += 1
                    full_route, committed_next_edge, apply_error = self.decision_engine.apply_route_decision(
                        str(vid), start_edge, action_idx, vehicle.destination
                    )
                    if apply_error:
                        continue
                    self._pending_decisions[vid] = PendingDecision(
                        state=None,
                        intended_action=action_idx,
                        intended_next_edge=committed_next_edge,
                        decision_edge=start_edge,
                        decision_step=step,
                        last_credit_edge=start_edge,
                        last_credit_step=step,
                        destination=vehicle.destination,
                        context=obs_context,
                        lane_change_requested=False,
                        route_fragment=list(full_route[1:]) if full_route else [],
                        metadata={"phase": "route_pending"},
                    )
                    continue
                if phase == "route_pending":
                    obs_snapshot = self._snapshot_vehicle(vid, start_edge, step)
                    if obs_snapshot is None:
                        continue
                    obs_context = self.decision_engine.build_context(str(vid), start_edge, vehicle.destination, step, snapshot=obs_snapshot)
                    should_preempt, trapped = self._should_preempt_pending_for_congestion(pending, obs_context, step)
                    if trapped:
                        self._metrics["trapped_replan_skips"] += 1
                        continue
                    if should_preempt:
                        fallback_actions = self.decision_engine.ranked_fallback_actions(
                            context=obs_context,
                            destination=vehicle.destination,
                            recent_history=list(self._recent_edges.get(vid, deque(maxlen=self.loop_window))),
                            blocked_action=pending.intended_action,
                            distance_fn=self._dist_to_dest,
                            congestion_score_fn=self._congestion_score,
                            branch_pressure_fn=self._branch_pressure_penalty,
                        )
                        recoverable = set(self._physically_recoverable_actions(obs_context))
                        fallback_actions = [a for a in fallback_actions if a in recoverable]
                        if fallback_actions:
                            action_idx = fallback_actions[0]
                            full_route, committed_next_edge, apply_error = self.decision_engine.apply_route_decision(
                                str(vid), start_edge, action_idx, vehicle.destination
                            )
                            if not apply_error:
                                self._pending_decisions[vid] = PendingDecision(
                                    state=None,
                                    intended_action=action_idx,
                                    intended_next_edge=committed_next_edge,
                                    decision_edge=start_edge,
                                    decision_step=step,
                                    last_credit_edge=start_edge,
                                    last_credit_step=step,
                                    destination=vehicle.destination,
                                    context=obs_context,
                                    lane_change_requested=False,
                                    route_fragment=list(full_route[1:]) if full_route else [],
                                    metadata={"phase": "route_pending", "action_source": "congestion_preempt"},
                                )
                                self._metrics["pending_preemptions"] += 1
                                self._metrics["fallback_to_lane_feasible_now"] += 1
                                self._record_commitment(committed_next_edge)
                        continue
                    continue

            snapshot = self._snapshot_vehicle(vid, start_edge, step)
            if snapshot is None:
                continue
            context = self.decision_engine.build_context(str(vid), start_edge, vehicle.destination, step, snapshot=snapshot)

            # Skip non-meaningful junction points; apply forced action directly.
            if context.forced_action is not None:
                action_idx = context.forced_action
            elif not self.decision_engine.is_decision_open(context):
                continue
            else:
                state = self.getState(vid, start_edge, vehicle.destination, context=context)
                cooldown_until = self._lane_change_cooldown.get((vid, start_edge), -1)
                cooldown_active = step < cooldown_until
                recent = list(self._recent_edges.get(vid, deque(maxlen=self.loop_window)))
                policy_actions = self._policy_action_candidates(
                    context=context,
                    recent_history=recent,
                    cooldown_active=cooldown_active,
                    destination=vehicle.destination,
                )
                action_idx = self.act(state, available_actions=policy_actions)
                self._metrics["decisions"] += 1

            if action_idx not in context.available_actions:
                self._metrics["impossible_action_overrides"] += 1
                continue
            recent = list(self._recent_edges.get(vid, deque(maxlen=self.loop_window)))
            safe_ok, signal = self.decision_engine.prefilter_action_for_loops(
                context=context,
                action_idx=action_idx,
                destination=vehicle.destination,
                recent_history=recent,
                distance_fn=self._dist_to_dest,
                distance_slack=self.score_slack,
            )
            if not safe_ok:
                self._metrics["loop_override_count"] += 1
                if signal.get("dead_end_reentry"):
                    self._metrics["dead_end_reentry_override_count"] += 1
                fallback_actions = self.decision_engine.ranked_fallback_actions(
                    context=context,
                    destination=vehicle.destination,
                    recent_history=recent,
                    blocked_action=action_idx,
                    distance_fn=self._dist_to_dest,
                    congestion_score_fn=self._congestion_score,
                    branch_pressure_fn=self._branch_pressure_penalty,
                )
                recoverable = set(self._physically_recoverable_actions(context))
                fallback_actions = [a for a in fallback_actions if a in recoverable]
                if not fallback_actions:
                    continue
                action_idx = self.act(state, available_actions=fallback_actions) if context.forced_action is None else fallback_actions[0]

            selected_next_edge = self.decision_engine.get_next_edge(start_edge, action_idx)
            if selected_next_edge is None:
                continue

            lane_change_requested = False
            if action_idx not in context.lane_feasible_now_actions:
                cooldown_until = self._lane_change_cooldown.get((vid, start_edge), -1)
                if step < cooldown_until:
                    self._metrics["cooldown_replans_blocked"] += 1
                    fallback_actions = self.decision_engine.ranked_fallback_actions(
                        context=context,
                        destination=vehicle.destination,
                        recent_history=recent,
                        blocked_action=action_idx,
                        distance_fn=self._dist_to_dest,
                        congestion_score_fn=self._congestion_score,
                        branch_pressure_fn=self._branch_pressure_penalty,
                    )
                    recoverable = set(self._physically_recoverable_actions(context))
                    fallback_actions = [a for a in fallback_actions if a in recoverable]
                    if not fallback_actions:
                        continue
                    action_idx = fallback_actions[0]
                    selected_next_edge = self.decision_engine.get_next_edge(start_edge, action_idx)
                    if selected_next_edge is None:
                        continue
                else:
                    lane_change_requested, lane_change_ok = self.decision_engine.try_request_lane_change(context, action_idx)
                    observe_meta = self.decision_engine.start_lane_change_observe(context, action_idx, step, lane_change_requested, lane_change_ok)
                    self._pending_decisions[vid] = PendingDecision(
                        state=None,
                        intended_action=action_idx,
                        intended_next_edge=selected_next_edge,
                        decision_edge=start_edge,
                        decision_step=step,
                        last_credit_edge=start_edge,
                        last_credit_step=step,
                        destination=vehicle.destination,
                        context=context,
                        lane_change_requested=lane_change_requested,
                        route_fragment=[],
                        metadata=observe_meta,
                    )
                    self._metrics["lane_change_observe_started"] += 1
                    self._metrics["deferred_lane_change_actions"] += 1
                    continue

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
                    metadata={"phase": "route_pending"},
                )
                self._lane_change_deferrals[vid] = 0
                self._record_commitment(next_edge)
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
        state.extend(self._compute_objective_features(vehicle_id, en, destination_edge))

        if self.use_compact_state:
            state.extend(self._local_congestion_features(en))
        else:
            for edge_now in self.connection_info.edge_list:
                car_num = traci.edge.getLastStepVehicleNumber(edge_now)
                density = car_num / self.connection_info.edge_length_dict[edge_now]
                state.append(density)

        state = np.reshape(state, [1, len(state)])
        return state
