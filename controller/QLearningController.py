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
        self.step_control_extra_buffer_m = 35.0
        self._last_observed_edge = {}
        self._last_control_step = {}
        self._decision_seq = 0
        # Inference telemetry glossary:
        # - event counts: increments once per discrete occurrence in this process lifetime
        # - gauges: instantaneous snapshots (none currently stored in _metrics)
        # - ratios: derived at print-time only (not stored as counters)
        self._metrics = {
            "decisions": 0,
            "overrides": 0,
            "loop_overrides": 0,
            "distance_overrides": 0,
            "impossible_action_overrides": 0,
            "deadend_overrides": 0,
            "decision_committed_skips": 0,
            "pending_decision_timeouts": 0,
            "fallback_selected_total": 0,
            "fallback_selected_lane_now": 0,
            "deferred_lane_change_actions": 0,
            "lane_change_observe_started": 0,
            "lane_change_observe_success": 0,
            "lane_change_observe_abort_no_progress": 0,
            "lane_change_observe_abort_low_speed": 0,
            "lane_change_observe_abort_commit_window": 0,
            "pending_release_observe_abort_no_progress": 0,
            "pending_release_observe_abort_commit_window": 0,
            "pending_release_observe_abort_low_speed": 0,
            "pending_release_wrong_lane_commit": 0,
            "pending_release_route_no_progress_abort": 0,
            "pending_release_route_stall_timeout": 0,
            "pending_release_route_hard_timeout": 0,
            "pending_release_events_total": 0,
            "pending_release_abort_events_total": 0,
            "pending_release_timeout_events_total": 0,
            "cooldown_replans_blocked": 0,
            "loop_override_count": 0,
            "dead_end_reentry_override_count": 0,
            "policy_candidates_with_broader_available": 0,
            "policy_candidates_collapsed_to_lane_now_only": 0,
            "pending_commit_window_grace_kept": 0,
            "proactive_shift2_candidates_seen": 0,
            "proactive_shift2_candidates_rejected": 0,
            "commit_window_candidates_rejected": 0,
            "commit_window_non_lane_candidates_seen": 0,
            "step_control_edge_change": 0,
            "step_control_pending": 0,
            "step_control_near_junction": 0,
            "step_control_lane_change_candidate": 0,
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
        # Must match RLTrainingPipeline compact state size; retrained models are required when this changes.
        self.compact_state_size = (2 * self.edge_embedding_dim) + 24 + 1 + 3 + 3 + self.local_congestion_k + 30
        self.legacy_state_size = 2 + 6 + 3 + 3 + len(self.connection_info.edge_list)
        self.use_compact_state = (self.model_state_size == self.compact_state_size)
        self.density_scale_m = 100.0
        self.direction_mask_start = (2 * self.edge_embedding_dim) if self.use_compact_state else 2
        self._init_edge_embeddings(seed=1337)

    def _next_decision_id(self):
        self._decision_seq += 1
        return f"inf_dec_{self._decision_seq}"

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

    def _edge_lane_count(self, edge_id):
        return max(len(self.connection_info.edge_lane_ids.get(edge_id, [])), 1)

    def _edge_lane_meters(self, edge_id):
        edge_len = max(float(self.connection_info.edge_length_dict.get(edge_id, 5.0)), 5.0)
        return edge_len * float(self._edge_lane_count(edge_id))

    def _edge_density(self, edge_id):
        count = traci.edge.getLastStepVehicleNumber(edge_id)
        return (float(count) * float(self.density_scale_m)) / max(self._edge_lane_meters(edge_id), 5.0)

    def _global_density_stats(self):
        edge_list = self.connection_info.edge_list
        if not edge_list:
            return 0.0, 0.0

        lane_meters = np.array([self._edge_lane_meters(edge) for edge in edge_list], dtype=np.float32)
        densities = np.array([self._edge_density(edge) for edge in edge_list], dtype=np.float32)
        total_vehicles = float(sum(traci.edge.getLastStepVehicleNumber(edge) for edge in edge_list))
        total_lane_meters = float(np.sum(lane_meters))
        mean_global = (total_vehicles * float(self.density_scale_m)) / max(total_lane_meters, 1.0)
        std_global = float(np.sqrt(np.average((densities - mean_global) ** 2, weights=lane_meters)))
        return mean_global, std_global

    def _local_congestion_features(self, edge_id):
        current_density = self._edge_density(edge_id)
        outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {})
        outgoing_densities = [
            self._edge_density(next_edge)
            for next_edge in outgoing.values()
        ]

        mean_global, std_global = self._global_density_stats()
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

    def _per_action_branch_features(self, context, destination):
        features = np.zeros(30, dtype=np.float32)
        lane_now = set(context.lane_feasible_now_actions)
        for action_idx in range(6):
            base = action_idx * 5
            if action_idx not in context.edge_valid_actions:
                continue
            next_edge = self.decision_engine.get_next_edge(context.edge_id, action_idx)
            if next_edge is None:
                continue
            features[base + 0] = float(context.required_lane_shift.get(action_idx, 0)) / 3.0
            features[base + 1] = 1.0 if action_idx in lane_now else 0.0
            features[base + 2] = float(self._edge_density(next_edge))
            eta = self._estimate_eta(next_edge, destination)
            features[base + 3] = min(float(eta) / 2000.0, 1.0) if np.isfinite(eta) else 1.0
            social = (1.25 * float(self._edge_density(next_edge))) + (0.01 * float(eta) if np.isfinite(eta) else 20.0)
            features[base + 4] = min(float(social), 10.0)
        return features

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
        extra_buffer = max(
            float(self.decision_engine.proactive_extra_buffer_m),
            0.35 * float(self.decision_engine.lane_change_margin_m),
        )
        comfortable_dist_threshold = (
            commit_distance
            + extra_buffer
            + float(self.decision_engine.proactive_safety_margin_m)
        )

        safe_lane_now_actions = []
        proactive_actions = []
        filtered_available_actions = []

        for action in available_actions:
            safe_ok, _ = self.decision_engine.prefilter_action_for_loops(
                context=context,
                action_idx=action,
                destination=destination,
                recent_history=recent_history,
                distance_fn=self._dist_to_dest,
                distance_slack=self.score_slack,
            )
            if not safe_ok:
                continue
            filtered_available_actions.append(action)
            if action in lane_now:
                safe_lane_now_actions.append(action)
                continue
            if cooldown_active:
                continue
            if float(context.speed) < 0.5:
                continue

            required_shift = int(context.required_lane_shift.get(action, 99))

            if context.commit_window:
                self._metrics["commit_window_non_lane_candidates_seen"] += 1
                self._metrics["commit_window_candidates_rejected"] += 1
                continue
            if required_shift == 2:
                self._metrics["proactive_shift2_candidates_seen"] += 1
            if required_shift not in (1, 2):
                continue
            dist_threshold = comfortable_dist_threshold
            if required_shift == 2:
                dist_threshold = comfortable_dist_threshold + (0.9 * float(self.decision_engine.lane_change_margin_m))
            if float(context.dist_to_end) <= dist_threshold:
                if required_shift == 2:
                    self._metrics["proactive_shift2_candidates_rejected"] += 1
                continue

            proactive_actions.append(action)

        policy_actions = sorted(set(safe_lane_now_actions) | set(proactive_actions))
        if not policy_actions:
            policy_actions = sorted(set(filtered_available_actions))
        if not policy_actions:
            return available_actions

        broader_available_set = set(filtered_available_actions)
        lane_now_set = set(safe_lane_now_actions)
        policy_set = set(policy_actions)
        if len(broader_available_set) > len(lane_now_set):
            self._metrics["policy_candidates_with_broader_available"] += 1
            if policy_set == lane_now_set and len(policy_set) < len(broader_available_set):
                self._metrics["policy_candidates_collapsed_to_lane_now_only"] += 1
        return policy_actions


    


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
            return
        step = int(traci.simulation.getTime())
        snapshot = self._snapshot_vehicle(vid, vehicle.current_edge, step)
        if vehicle.current_edge == pending.decision_edge:
            if snapshot is None:
                return
            context = self.decision_engine.build_context(str(vid), vehicle.current_edge, vehicle.destination, step, snapshot=snapshot)
            metadata = pending.metadata if isinstance(pending.metadata, dict) else {}
            metadata["lane_position_now"] = float(snapshot.lane_position)
            pending.metadata = metadata
            metadata, progress_view = self.decision_engine.pending_progress_update(
                pending=pending,
                context=context,
                step=step,
            )
            last_progress_step = int(progress_view["last_progress_step"])

            current_shift = int(context.required_lane_shift.get(pending.intended_action, 99))
            grace_keep = (
                current_shift <= 1
                and context.dist_to_end >= max(self.decision_engine.commit_min_distance + 2.0, 6.0)
            )
            wrong_lane_commit = (
                context.commit_window
                and pending.intended_action not in context.lane_feasible_now_actions
                and not grace_keep
            )
            stall_age = max(int(step) - int(last_progress_step), 0)
            total_age = max(int(step) - int(pending.decision_step), 0)
            no_progress_window = int(self.decision_engine.route_pending_no_progress_window_steps)
            no_progress_stall = (
                stall_age >= max(no_progress_window, 1)
                and not bool(progress_view["made_progress"])
            )
            stalled_timeout = stall_age >= int(self.decision_engine.route_pending_stall_steps)
            hard_timeout = total_age >= int(self.decision_engine.route_pending_hard_timeout_steps)

            if wrong_lane_commit or no_progress_stall or stalled_timeout or hard_timeout:
                self._pending_decisions.pop(vid, None)
                release_reason = None
                release_as_timeout = False
                if hard_timeout:
                    release_reason = "route_hard_timeout"
                    release_as_timeout = True
                elif stalled_timeout:
                    release_reason = "route_stall_timeout"
                    release_as_timeout = True
                elif no_progress_stall:
                    release_reason = "route_no_progress_abort"
                elif wrong_lane_commit:
                    release_reason = "wrong_lane_commit"
                if release_reason is not None:
                    self._record_pending_release(release_reason)
                if release_as_timeout:
                    self._metrics["pending_decision_timeouts"] += 1
                self._lane_change_cooldown[(vid, vehicle.current_edge)] = (
                    step + self.decision_engine.cooldown_after_pending_release(timeout=release_as_timeout)
                )
                return
            if context.commit_window and pending.intended_action not in context.lane_feasible_now_actions and grace_keep:
                self._metrics["pending_commit_window_grace_kept"] += 1
            self._metrics["decision_committed_skips"] += 1
            return
        self._pending_decisions.pop(vid, None)
        self._lane_change_deferrals[vid] = 0

    def _record_pending_release(self, reason):
        key_map = {
            "observe_abort_no_progress": "pending_release_observe_abort_no_progress",
            "observe_abort_commit_window": "pending_release_observe_abort_commit_window",
            "observe_abort_low_speed": "pending_release_observe_abort_low_speed",
            "wrong_lane_commit": "pending_release_wrong_lane_commit",
            "route_no_progress_abort": "pending_release_route_no_progress_abort",
            "route_stall_timeout": "pending_release_route_stall_timeout",
            "route_hard_timeout": "pending_release_route_hard_timeout",
        }
        key = key_map.get(reason)
        if key:
            self._metrics[key] += 1
        self._metrics["pending_release_events_total"] += 1
        if reason in {"route_stall_timeout", "route_hard_timeout"}:
            self._metrics["pending_release_timeout_events_total"] += 1
        else:
            self._metrics["pending_release_abort_events_total"] += 1

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

    def should_control_vehicle(self, vehicle_id, vehicle, step):
        vid = str(vehicle_id)

        # Avoid duplicate same-step work
        if self._last_control_step.get(vid) == int(step):
            return False

        try:
            current_edge = traci.vehicle.getRoadID(vid)
        except traci.TraCIException:
            return False

        if current_edge not in self.connection_info.edge_index_dict:
            return False
        if current_edge == vehicle.destination:
            return False

        # Always revisit while a pending decision exists.
        if vid in self._pending_decisions:
            self._metrics["step_control_pending"] += 1
            return True

        # Preserve normal edge-change-driven control.
        if current_edge != vehicle.current_edge:
            self._metrics["step_control_edge_change"] += 1
            return True

        snapshot = self._snapshot_vehicle(vid, current_edge, int(step))
        if snapshot is None:
            return False

        context = self.decision_engine.build_context(
            vid,
            current_edge,
            vehicle.destination,
            int(step),
            snapshot=snapshot,
        )

        # Only ask for step-wise control near meaningful junction decision zones.
        if len(context.edge_valid_actions) <= 1:
            return False

        reaction_distance = max(
            float(self.decision_engine.base_reaction_distance),
            float(snapshot.speed) * float(self.decision_engine.reaction_time_s),
        )
        near_threshold = reaction_distance + float(self.step_control_extra_buffer_m)
        near_junction = float(context.dist_to_end) <= float(near_threshold)

        commit_distance = max(
            float(self.decision_engine.commit_min_distance),
            float(snapshot.speed) * float(self.decision_engine.commit_time_s),
        )
        extra_buffer = max(6.0, 0.35 * float(self.decision_engine.lane_change_margin_m))
        comfortable_dist_threshold = commit_distance + extra_buffer
        proactive_control_threshold = max(
            float(near_threshold),
            float(comfortable_dist_threshold + 1.5 * float(self.decision_engine.lane_change_margin_m)),
        )

        lane_now = set(context.lane_feasible_now_actions)
        proactive_candidates = [
            a for a in context.available_actions
            if (a not in lane_now and int(context.required_lane_shift.get(a, 99)) <= 1)
        ]

        cooldown_until = self._lane_change_cooldown.get((vid, current_edge), -1)
        cooldown_active = int(step) < int(cooldown_until)

        if proactive_candidates and not cooldown_active and float(context.dist_to_end) <= proactive_control_threshold:
            self._metrics["step_control_lane_change_candidate"] += 1
            return True

        if not near_junction:
            return False

        if self.decision_engine.is_decision_open(context):
            self._metrics["step_control_near_junction"] += 1
            return True

        return False
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
            self._last_control_step[vid] = step
            if vid not in self._recent_edges:
                self._recent_edges[vid] = deque(maxlen=self.loop_window)
            prev_seen_edge = self._last_observed_edge.get(vid)
            edge_changed_runtime = (prev_seen_edge != start_edge)

            self._visit_count.setdefault(vid, {})
            self._best_dist.setdefault(vid, float("inf"))

            if prev_seen_edge is None or edge_changed_runtime:
                self._recent_edges[vid].append(start_edge)
                self._visit_count[vid][start_edge] = self._visit_count[vid].get(start_edge, 0) + 1
                self._best_dist[vid] = min(self._best_dist[vid], self._dist_to_dest(start_edge, vehicle.destination))

            self._last_observed_edge[vid] = start_edge
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
                            self._metrics["distance_overrides"] += 1
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
                            decision_id=str((pending.metadata or {}).get("decision_id", pending.decision_id or self._next_decision_id())),
                            decision_origin_mode=str((pending.metadata or {}).get("decision_origin_mode", pending.decision_origin_mode or "proactive")),
                            decision_current_phase="route_pending",
                            decision_open_recorded=True,
                            route_fragment=list(full_route[1:]) if full_route else [],
                            metadata={
                                **(pending.metadata if isinstance(pending.metadata, dict) else {}),
                                "phase": "route_pending",
                                "decision_current_phase": "route_pending",
                                "best_dist_to_end": float(obs_context.dist_to_end),
                                "last_progress_step": int(step),
                            },
                        )
                        continue
                    if reason == "commit_window":
                        self._metrics["lane_change_observe_abort_commit_window"] += 1
                        self._record_pending_release("observe_abort_commit_window")
                    elif reason == "low_speed":
                        self._metrics["lane_change_observe_abort_low_speed"] += 1
                        self._record_pending_release("observe_abort_low_speed")
                    else:
                        self._metrics["lane_change_observe_abort_no_progress"] += 1
                        self._record_pending_release("observe_abort_no_progress")
                    self._metrics["overrides"] += 1
                    self._lane_change_cooldown[(vid, start_edge)] = (
                        step + self.decision_engine.cooldown_after_pending_release(timeout=False)
                    )
                    fallback_actions = self.decision_engine.ranked_fallback_actions(
                        context=obs_context,
                        destination=vehicle.destination,
                        recent_history=list(self._recent_edges.get(vid, deque(maxlen=self.loop_window))),
                        blocked_action=pending.intended_action,
                        distance_fn=self._dist_to_dest,
                    )
                    if not fallback_actions:
                        continue
                    action_idx = fallback_actions[0]
                    selected_next_edge = self.decision_engine.get_next_edge(start_edge, action_idx)
                    if selected_next_edge is None:
                        continue
                    self._metrics["fallback_selected_total"] += 1
                    if action_idx in obs_context.lane_feasible_now_actions:
                        self._metrics["fallback_selected_lane_now"] += 1
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
                        decision_id=str((pending.metadata or {}).get("decision_id", pending.decision_id or self._next_decision_id())),
                        decision_origin_mode=str((pending.metadata or {}).get("decision_origin_mode", pending.decision_origin_mode or "proactive")),
                        decision_current_phase="route_pending",
                        decision_open_recorded=True,
                        route_fragment=list(full_route[1:]) if full_route else [],
                        metadata={
                            "phase": "route_pending",
                            "decision_current_phase": "route_pending",
                            "best_dist_to_end": float(obs_context.dist_to_end),
                            "last_progress_step": int(step),
                            "decision_id": str((pending.metadata or {}).get("decision_id", pending.decision_id or self._next_decision_id())),
                            "decision_origin_mode": str((pending.metadata or {}).get("decision_origin_mode", pending.decision_origin_mode or "proactive")),
                            "decision_open_recorded": True,
                        },
                    )
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
                self._metrics["overrides"] += 1
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
                self._metrics["overrides"] += 1
                self._metrics["loop_overrides"] += 1
                self._metrics["loop_override_count"] += 1
                if signal.get("dead_end_reentry"):
                    self._metrics["deadend_overrides"] += 1
                    self._metrics["dead_end_reentry_override_count"] += 1
                if signal.get("distance_worsen"):
                    self._metrics["distance_overrides"] += 1
                fallback_actions = self.decision_engine.ranked_fallback_actions(
                    context=context,
                    destination=vehicle.destination,
                    recent_history=recent,
                    blocked_action=action_idx,
                    distance_fn=self._dist_to_dest,
                )
                if not fallback_actions:
                    continue
                action_idx = self.act(state, available_actions=fallback_actions) if context.forced_action is None else fallback_actions[0]
                self._metrics["fallback_selected_total"] += 1
                if action_idx in context.lane_feasible_now_actions:
                    self._metrics["fallback_selected_lane_now"] += 1

            selected_next_edge = self.decision_engine.get_next_edge(start_edge, action_idx)
            if selected_next_edge is None:
                continue

            lane_change_requested = False
            if action_idx not in context.lane_feasible_now_actions:
                cooldown_until = self._lane_change_cooldown.get((vid, start_edge), -1)
                if step < cooldown_until:
                    self._metrics["overrides"] += 1
                    self._metrics["cooldown_replans_blocked"] += 1
                    fallback_actions = self.decision_engine.ranked_fallback_actions(
                        context=context,
                        destination=vehicle.destination,
                        recent_history=recent,
                        blocked_action=action_idx,
                        distance_fn=self._dist_to_dest,
                    )
                    if not fallback_actions:
                        continue
                    action_idx = fallback_actions[0]
                    self._metrics["fallback_selected_total"] += 1
                    if action_idx in context.lane_feasible_now_actions:
                        self._metrics["fallback_selected_lane_now"] += 1
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
                        decision_id=self._next_decision_id(),
                        decision_origin_mode=("lane_now" if action_idx in context.lane_feasible_now_actions else "proactive"),
                        decision_current_phase="observe_lane_change",
                        decision_open_recorded=True,
                        route_fragment=[],
                        metadata={
                            **observe_meta,
                            "decision_id": f"inf_dec_{self._decision_seq}",
                            "decision_origin_mode": ("lane_now" if action_idx in context.lane_feasible_now_actions else "proactive"),
                            "decision_current_phase": "observe_lane_change",
                            "decision_open_recorded": True,
                        },
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
                self._metrics["distance_overrides"] += 1
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
                    decision_id=self._next_decision_id(),
                    decision_origin_mode=("lane_now" if action_idx in context.lane_feasible_now_actions else "proactive"),
                    decision_current_phase="route_pending",
                    decision_open_recorded=True,
                    route_fragment=list(full_route[1:]) if full_route else [],
                    metadata={
                        "phase": "route_pending",
                        "decision_current_phase": "route_pending",
                        "best_dist_to_end": float(context.dist_to_end),
                        "last_progress_step": int(step),
                        "decision_id": f"inf_dec_{self._decision_seq}",
                        "decision_origin_mode": ("lane_now" if action_idx in context.lane_feasible_now_actions else "proactive"),
                        "decision_open_recorded": True,
                    },
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
                self._metrics["step_control_edge_change"],
                self._metrics["step_control_pending"],
                self._metrics["step_control_near_junction"],
                self._metrics["step_control_lane_change_candidate"],
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
            state.extend(self._per_action_branch_features(context, destination_edge).tolist())
        else:
            for edge_now in self.connection_info.edge_list:
                state.append(self._edge_density(edge_now))

        state = np.reshape(state, [1, len(state)])
        return state
