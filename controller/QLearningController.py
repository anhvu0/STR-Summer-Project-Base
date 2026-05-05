from controller.RouteController import RouteController
from core.Util import ConnectionInfo, Vehicle
from keras.models import load_model
import numpy as np
import traci
from traci import constants as tc
import sumolib
import math
from collections import deque

from xml.dom.minidom import parse
import os
from core.dueling_q_layers import DuelingQCombine  # noqa: F401
from core.junction_decision_engine import JunctionDecisionEngine, PendingDecision, VehicleSnapshot
from core.shared_decision_policy import SharedDecisionPolicy
from core.route_loop_safety import transition_signal

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
        self._distance_cache = {}
        self._lane_change_cooldown = {}
        self._stale_lane_now_replan_targets = {}
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
            "loop_events": 0,
            "short_cycle_events": 0,
            "aba_bounce_events": 0,
            "dead_end_reentry_events": 0,
            "long_horizon_loop_events": 0,
            "revisit_without_progress_events": 0,
            "small_set_loop_events": 0,
            "small_set_loop_unique3_or_less_events": 0,
            "small_set_loop_unique4_events": 0,
            "committed_cyclic_revisit_events": 0,
            "committed_cyclic_revisit_after_fallback_events": 0,
            "policy_candidates_with_broader_available": 0,
            "policy_candidates_collapsed_to_lane_now_only": 0,
            "pending_commit_window_grace_kept": 0,
            "proactive_shift2_candidates_seen": 0,
            "proactive_shift2_candidates_rejected": 0,
            "proactive_brake_risk_candidates_seen": 0,
            "proactive_brake_risk_candidates_rejected": 0,
            "proactive_brake_risk_fallback_kept": 0,
            "lane_now_replan_releases": 0,
            "lane_now_replan_forced_alternative": 0,
            "lane_now_replan_blocked_reopen_actions": 0,
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
        self.edge_embedding_dim = 8
        self.shared_policy = SharedDecisionPolicy(
            self.connection_info,
            self.decision_engine,
            self.direction_choices,
            edge_embedding_dim=self.edge_embedding_dim,
            density_scale_m=100.0,
        )
        # Must match RLTrainingPipeline compact state spec; retrained models are required when this changes.
        self.compact_state_size = self.shared_policy.compact_state_size
        self.legacy_state_size = self.shared_policy.legacy_state_size(len(self.connection_info.edge_list))
        self.use_compact_state = (self.model_state_size == self.compact_state_size)
        self.density_scale_m = 100.0
        self._edge_list = tuple(self.connection_info.edge_list)
        self._lane_length_cache = {}
        self._edge_lane_meters_cache = {
            edge_id: max(float(self.connection_info.edge_length_dict.get(edge_id, 5.0)), 5.0)
            * float(max(len(self.connection_info.edge_lane_ids.get(edge_id, [])), 1))
            for edge_id in self._edge_list
        }
        self._edge_lane_meters_vec = np.array(
            [self._edge_lane_meters_cache[edge_id] for edge_id in self._edge_list],
            dtype=np.float32,
        )
        self._edge_vehicle_count_cache = {}
        self._density_vec = np.zeros(len(self._edge_list), dtype=np.float32)
        self._density_mean = 0.0
        self._density_std = 0.0
        self._density_p95 = 0.0
        self._last_density_step = -10**9
        self._vehicle_subscription_vars = (
            tc.VAR_ROAD_ID,
            tc.VAR_LANE_ID,
            tc.VAR_LANE_INDEX,
            tc.VAR_LANEPOSITION,
            tc.VAR_SPEED,
        )
        self._active_vehicle_subscriptions = set()
        self._step_cache_step = None
        self._step_vehicle_results = {}
        self._step_snapshot_cache = {}
        self._step_context_cache = {}
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
        cached = self._edge_lane_meters_cache.get(edge_id)
        if cached is not None:
            return float(cached)
        edge_len = max(float(self.connection_info.edge_length_dict.get(edge_id, 5.0)), 5.0)
        return edge_len * float(self._edge_lane_count(edge_id))

    def _lane_length(self, lane_id):
        cached = self._lane_length_cache.get(lane_id)
        if cached is not None:
            return float(cached)
        lane_len = float(traci.lane.getLength(lane_id))
        self._lane_length_cache[lane_id] = lane_len
        return lane_len

    def _edge_density(self, edge_id):
        cached_count = self._edge_vehicle_count_cache.get(edge_id)
        if cached_count is None:
            cached_count = traci.edge.getLastStepVehicleNumber(edge_id)
        return (float(cached_count) * float(self.density_scale_m)) / max(self._edge_lane_meters(edge_id), 5.0)

    def _occupied_density_p95(self, density_vec):
        occupied = density_vec[density_vec > 0.0]
        if occupied.size == 0:
            return 0.0
        return float(np.percentile(occupied, 95))

    def _refresh_density_stats(self, step, every=1):
        if hasattr(self, "_last_density_step") and (int(step) - int(self._last_density_step)) < int(every):
            return

        edge_list = self._edge_list
        lane_meters_vec = self._edge_lane_meters_vec
        counts = np.array(
            [float(self.connection_info.edge_vehicle_count.get(edge_id, 0)) for edge_id in edge_list],
            dtype=np.float32,
        )
        self._edge_vehicle_count_cache = {edge_id: int(count) for edge_id, count in zip(edge_list, counts)}
        self._density_vec = (counts * float(self.density_scale_m)) / np.maximum(lane_meters_vec, 5.0)

        if len(self._density_vec) > 0 and len(lane_meters_vec) > 0:
            total_vehicles = float(np.sum(counts))
            total_lane_meters = float(np.sum(lane_meters_vec))
            self._density_mean = (total_vehicles * float(self.density_scale_m)) / max(total_lane_meters, 1.0)
            density_diff_sq = (self._density_vec - self._density_mean) ** 2
            self._density_std = float(np.sqrt(np.average(density_diff_sq, weights=lane_meters_vec)))
            self._density_p95 = self._occupied_density_p95(self._density_vec)
        else:
            self._density_mean = 0.0
            self._density_std = 0.0
            self._density_p95 = 0.0
        self._last_density_step = int(step)

    def _prepare_step_cache(self, step):
        step = int(step)
        if self._step_cache_step == step:
            return
        self._step_cache_step = step
        self._step_snapshot_cache = {}
        self._step_context_cache = {}
        try:
            self._step_vehicle_results = traci.vehicle.getAllSubscriptionResults() or {}
        except Exception:
            self._step_vehicle_results = {}

    def _ensure_vehicle_subscription(self, vehicle_id):
        vehicle_id = str(vehicle_id)
        if vehicle_id in self._active_vehicle_subscriptions:
            return
        try:
            traci.vehicle.subscribe(vehicle_id, self._vehicle_subscription_vars)
            self._active_vehicle_subscriptions.add(vehicle_id)
        except traci.TraCIException:
            return

    def cleanup_vehicle_state(self, vehicle_id):
        """
        Clear per-vehicle inference state once SUMO reports a terminal outcome.
        """
        vid = str(vehicle_id)
        self._pending_decisions.pop(vid, None)
        self._visit_count.pop(vid, None)
        self._best_dist.pop(vid, None)
        self._recent_edges.pop(vid, None)
        self._last_observed_edge.pop(vid, None)
        self._last_control_step.pop(vid, None)
        self._step_snapshot_cache.pop(vid, None)
        self._active_vehicle_subscriptions.discard(vid)
        stale_cooldowns = [key for key in self._lane_change_cooldown if key and key[0] == vid]
        for key in stale_cooldowns:
            self._lane_change_cooldown.pop(key, None)
        stale_replans = [key for key in self._stale_lane_now_replan_targets if key and key[0] == vid]
        for key in stale_replans:
            self._stale_lane_now_replan_targets.pop(key, None)

    #-----------------------DEBUGGING-------------------------------------
    def _dist_to_dest(self, edge_id, dest_id):
        key = (edge_id, dest_id)
        if key in self._distance_cache:
            return self._distance_cache[key]
        try:
            from_edge = self.net.getEdge(edge_id)
            to_edge = self.net.getEdge(dest_id)
            path_edges, path_cost = self.net.getShortestPath(from_edge, to_edge, vClass="passenger")
            if path_edges is None:
                self._distance_cache[key] = float("inf")
            else:
                self._distance_cache[key] = path_cost
            return self._distance_cache[key]
        except Exception:
            self._distance_cache[key] = float("inf")
            return float("inf")

    def _estimate_eta(self, edge_id, dest_id):
        dist = self._dist_to_dest(edge_id, dest_id)
        if not np.isfinite(dist):
            return float("inf")
        return float(dist) / 8.0

    def _edge_out_degree_map(self, edges):
        return {
            edge: len(self.connection_info.outgoing_edges_dict.get(edge, {}))
            for edge in set(edges)
        }

    def _record_runtime_loop_signals(self, current_edge, destination, recent_history):
        signal_edges = set(recent_history) | {current_edge}
        edge_distance_lookup = {
            edge: self._dist_to_dest(edge, destination)
            for edge in signal_edges
        }
        loop_signals = transition_signal(
            deque(recent_history, maxlen=self.loop_window),
            current_edge,
            edge_out_degree=self._edge_out_degree_map(list(recent_history) + [current_edge]),
            edge_distance_lookup=edge_distance_lookup,
            progress_slack=self.decision_engine.loop_distance_slack,
        )
        if loop_signals.get("aba_bounce"):
            self._metrics["aba_bounce_events"] += 1
        if loop_signals.get("short_cycle"):
            self._metrics["short_cycle_events"] += 1
        if loop_signals.get("dead_end_reentry"):
            self._metrics["dead_end_reentry_events"] += 1
        if loop_signals.get("long_horizon_loop"):
            self._metrics["long_horizon_loop_events"] += 1
        if loop_signals.get("revisit_without_progress"):
            self._metrics["revisit_without_progress_events"] += 1

        if (
            loop_signals.get("aba_bounce")
            or loop_signals.get("short_cycle")
            or loop_signals.get("dead_end_reentry")
            or loop_signals.get("long_horizon_loop")
            or loop_signals.get("revisit_without_progress")
        ):
            self._metrics["loop_events"] += 1

        probe = list(recent_history) + [current_edge]
        recent_probe = probe[-8:]
        repeated_current = current_edge in recent_history
        if repeated_current and len(recent_probe) >= 4:
            unique_edges = len(set(recent_probe))
            if unique_edges <= 4:
                self._metrics["small_set_loop_events"] += 1
                if unique_edges <= 3:
                    self._metrics["small_set_loop_unique3_or_less_events"] += 1
                else:
                    self._metrics["small_set_loop_unique4_events"] += 1

    def _record_committed_cyclic_revisit(self, pending, actual_edge, destination, recent_history):
        if pending is None or actual_edge == pending.decision_edge:
            return
        if not self.decision_engine.route_matches_expected(pending, actual_edge):
            return

        repeated_recent_edges = sum(1 for edge in recent_history if edge == actual_edge)
        signal_edges = set(recent_history) | {actual_edge}
        edge_distance_lookup = {
            edge: self._dist_to_dest(edge, destination)
            for edge in signal_edges
        }
        loop_signals = transition_signal(
            deque(recent_history, maxlen=self.loop_window),
            actual_edge,
            edge_out_degree=self._edge_out_degree_map(list(recent_history) + [actual_edge]),
            edge_distance_lookup=edge_distance_lookup,
            progress_slack=self.decision_engine.loop_distance_slack,
        )
        committed_cyclic = bool(
            repeated_recent_edges > 1
            or loop_signals.get("short_cycle")
            or loop_signals.get("aba_bounce")
            or loop_signals.get("dead_end_reentry")
            or loop_signals.get("long_horizon_loop")
            or loop_signals.get("revisit_without_progress")
        )
        if not committed_cyclic:
            return

        self._metrics["committed_cyclic_revisit_events"] += 1
        action_source = str((pending.metadata or {}).get("action_source", ""))
        if "fallback" in action_source:
            self._metrics["committed_cyclic_revisit_after_fallback_events"] += 1

    def _format_pending_descriptor(self, descriptor):
        if not descriptor:
            return "none"
        return (
            f"{descriptor['vehicle_id']}@{descriptor['edge']}:"
            f"age={descriptor['age']},stall={descriptor['stall_age']},"
            f"phase={descriptor['phase']},mode={descriptor['resolution_mode']},"
            f"shift={descriptor['current_shift']}"
        )

    def _runtime_pending_snapshot(self):
        summary = {
            "total_open": 0,
            "observe_open": 0,
            "route_open": 0,
            "lane_now_open": 0,
            "proactive_open": 0,
            "active_monitoring_open": 0,
            "oldest_descriptor": None,
        }
        if not self._pending_decisions:
            return summary

        try:
            step = int(traci.simulation.getTime())
        except Exception:
            step = 0

        oldest_descriptor = None
        for vehicle_id, pending in self._pending_decisions.items():
            metadata = pending.metadata if isinstance(pending.metadata, dict) else {}
            phase = self.shared_policy.pending_phase(pending)
            resolution_mode = self.shared_policy.pending_resolution_mode(pending)
            active_monitoring = self.shared_policy.pending_requires_active_same_edge_monitoring(pending)
            age = max(step - int(pending.decision_step), 0)
            last_progress_step = int(metadata.get("last_progress_step", pending.decision_step))
            stall_age = max(step - last_progress_step, 0)
            current_shift = int(metadata.get("last_required_shift", metadata.get("observe_last_required_shift", 99)))
            descriptor = {
                "vehicle_id": str(vehicle_id),
                "edge": str(pending.decision_edge),
                "age": int(age),
                "stall_age": int(stall_age),
                "phase": str(phase),
                "resolution_mode": str(resolution_mode),
                "current_shift": int(current_shift),
            }
            if oldest_descriptor is None or (descriptor["age"], descriptor["stall_age"], descriptor["vehicle_id"]) > (
                oldest_descriptor["age"],
                oldest_descriptor["stall_age"],
                oldest_descriptor["vehicle_id"],
            ):
                oldest_descriptor = descriptor

            summary["total_open"] += 1
            if phase == "observe_lane_change":
                summary["observe_open"] += 1
            else:
                summary["route_open"] += 1
            if resolution_mode == "lane_now":
                summary["lane_now_open"] += 1
            else:
                summary["proactive_open"] += 1
            if active_monitoring:
                summary["active_monitoring_open"] += 1

        summary["oldest_descriptor"] = oldest_descriptor
        return summary

    def get_runtime_metrics(self):
        metrics = dict(self._metrics)
        decisions = float(max(metrics.get("decisions", 0), 1))
        fallback_total = float(max(metrics.get("fallback_selected_total", 0), 1))
        committed_total = float(max(metrics.get("committed_cyclic_revisit_events", 0), 1))
        metrics["override_ratio"] = float(metrics.get("overrides", 0)) / decisions
        metrics["fallback_lane_now_ratio"] = (
            float(metrics.get("fallback_selected_lane_now", 0)) / fallback_total
        )
        metrics["committed_cyclic_revisit_after_fallback_ratio"] = (
            float(metrics.get("committed_cyclic_revisit_after_fallback_events", 0)) / committed_total
        )
        return metrics

    def format_runtime_metrics_summary(self):
        metrics = self.get_runtime_metrics()
        pending_snapshot = self._runtime_pending_snapshot()
        return [
            (
                "[RL-INFER] decisions={} overrides={} override_ratio={:.1%} "
                "fallbacks(total/lane_now)={}/{}"
            ).format(
                int(metrics["decisions"]),
                int(metrics["overrides"]),
                float(metrics["override_ratio"]),
                int(metrics["fallback_selected_total"]),
                int(metrics["fallback_selected_lane_now"]),
            ),
            (
                "[RL-INFER] loops total={} short={} aba={} dead_end={} long={} revisit_no_progress={}"
            ).format(
                int(metrics["loop_events"]),
                int(metrics["short_cycle_events"]),
                int(metrics["aba_bounce_events"]),
                int(metrics["dead_end_reentry_events"]),
                int(metrics["long_horizon_loop_events"]),
                int(metrics["revisit_without_progress_events"]),
            ),
            (
                "[RL-INFER] small_set_loops total={} unique<=3={} unique=4={} "
                "committed_cyclic={} after_fallback={}"
            ).format(
                int(metrics["small_set_loop_events"]),
                int(metrics["small_set_loop_unique3_or_less_events"]),
                int(metrics["small_set_loop_unique4_events"]),
                int(metrics["committed_cyclic_revisit_events"]),
                int(metrics["committed_cyclic_revisit_after_fallback_events"]),
            ),
            (
                "[RL-INFER] pending_timeouts={} release(no_prog/stall/hard)={}/{}/{} "
                "lane_now_rescue(release/forced/blocked)={}/{}/{}"
            ).format(
                int(metrics["pending_decision_timeouts"]),
                int(metrics["pending_release_route_no_progress_abort"]),
                int(metrics["pending_release_route_stall_timeout"]),
                int(metrics["pending_release_route_hard_timeout"]),
                int(metrics["lane_now_replan_releases"]),
                int(metrics["lane_now_replan_forced_alternative"]),
                int(metrics["lane_now_replan_blocked_reopen_actions"]),
            ),
            (
                "[RL-INFER] observe(start/success/abort_np/abort_ls/abort_cw)={}/{}/{}/{}/{} "
                "cooldown_blocked={}"
            ).format(
                int(metrics["lane_change_observe_started"]),
                int(metrics["lane_change_observe_success"]),
                int(metrics["lane_change_observe_abort_no_progress"]),
                int(metrics["lane_change_observe_abort_low_speed"]),
                int(metrics["lane_change_observe_abort_commit_window"]),
                int(metrics["cooldown_replans_blocked"]),
            ),
            (
                "[RL-INFER] pending_end total={} observe/route={}/{} "
                "lane_now/proactive={}/{} active_monitor={} oldest={}"
            ).format(
                int(pending_snapshot["total_open"]),
                int(pending_snapshot["observe_open"]),
                int(pending_snapshot["route_open"]),
                int(pending_snapshot["lane_now_open"]),
                int(pending_snapshot["proactive_open"]),
                int(pending_snapshot["active_monitoring_open"]),
                self._format_pending_descriptor(pending_snapshot["oldest_descriptor"]),
            ),
        ]

    def _finalize_commitment(self, vehicle):
        vid = vehicle.vehicle_id
        pending = self._pending_decisions.get(vid)
        if not pending:
            return
        phase = str((pending.metadata or {}).get("phase", "route_pending"))
        if phase != "route_pending":
            return
        step = int(traci.simulation.getTime())
        self._prepare_step_cache(step)
        snapshot = self._snapshot_vehicle(vid, vehicle.current_edge, step)
        if vehicle.current_edge == pending.decision_edge:
            if snapshot is None:
                return
            active_pending = self.shared_policy.pending_requires_active_same_edge_monitoring(pending)
            context = self._get_step_context(str(vid), vehicle.current_edge, vehicle.destination, step, snapshot=snapshot)
            if active_pending and self.decision_engine.should_timeout_pending(pending, step):
                self._pending_decisions.pop(vid, None)
                self._record_pending_release("route_stall_timeout")
                self._metrics["pending_decision_timeouts"] += 1
                self._lane_change_cooldown[(vid, vehicle.current_edge)] = (
                    step + self.decision_engine.cooldown_after_pending_release(timeout=True)
                )
                return
            release_eval = self.shared_policy.evaluate_route_pending_release(
                pending,
                context=context,
                step=step,
                lane_position_now=float(snapshot.lane_position),
                edge_density_fn=self._edge_density,
                distance_fn=self._dist_to_dest,
                recent_history=list(self._recent_edges.get(vid, deque(maxlen=self.loop_window))),
            )
            if release_eval.should_release:
                self._pending_decisions.pop(vid, None)
                self._record_pending_release(release_eval.release_reason)
                if release_eval.release_as_timeout:
                    self._metrics["pending_decision_timeouts"] += 1
                elif (
                    release_eval.avoid_reopen_action is not None
                    and release_eval.preferred_replan_action is not None
                ):
                    self._stale_lane_now_replan_targets[(vid, vehicle.current_edge)] = {
                        "blocked_action": int(release_eval.avoid_reopen_action),
                        "preferred_action": int(release_eval.preferred_replan_action),
                        "until": int(step + self.decision_engine.cooldown_after_pending_release(timeout=False)),
                    }
                    self._metrics["lane_now_replan_releases"] += 1
                self._lane_change_cooldown[(vid, vehicle.current_edge)] = (
                    step + self.decision_engine.cooldown_after_pending_release(timeout=release_eval.release_as_timeout)
                )
                return
            if (
                context.commit_window
                and pending.intended_action not in context.lane_feasible_now_actions
                and release_eval.grace_keep
            ):
                self._metrics["pending_commit_window_grace_kept"] += 1
            if active_pending:
                self._metrics["decision_committed_skips"] += 1
            return
        self._pending_decisions.pop(vid, None)

    def _force_stale_lane_now_replan_if_available(self, vid, current_edge, step, context, policy_actions):
        key = (vid, current_edge)
        target_info = self._stale_lane_now_replan_targets.get(key)
        if not target_info:
            return policy_actions
        if int(step) > int(target_info.get("until", step)):
            self._stale_lane_now_replan_targets.pop(key, None)
            return policy_actions

        blocked_action = int(target_info.get("blocked_action", -1))
        preferred_action = int(target_info.get("preferred_action", -1))
        available = set(int(action) for action in context.available_actions)
        if preferred_action in available:
            self._metrics["lane_now_replan_forced_alternative"] += 1
            return [preferred_action]
        if blocked_action in policy_actions and len(policy_actions) > 1:
            filtered_actions = [action for action in policy_actions if int(action) != blocked_action]
            if filtered_actions:
                self._metrics["lane_now_replan_blocked_reopen_actions"] += 1
                return filtered_actions
        return policy_actions

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
        step = int(step)
        vid = str(vid)
        self._prepare_step_cache(step)
        cached = self._step_snapshot_cache.get(vid)
        if cached is not None and cached.edge_id == edge_id:
            return cached

        result = self._step_vehicle_results.get(vid) or {}
        try:
            lane_id = result.get(tc.VAR_LANE_ID)
            if lane_id is None:
                lane_id = traci.vehicle.getLaneID(vid)
            lane_index = result.get(tc.VAR_LANE_INDEX)
            if lane_index is None:
                lane_index = traci.vehicle.getLaneIndex(vid)
            lane_position = result.get(tc.VAR_LANEPOSITION)
            if lane_position is None:
                lane_position = traci.vehicle.getLanePosition(vid)
            speed = result.get(tc.VAR_SPEED)
            if speed is None:
                speed = traci.vehicle.getSpeed(vid)
            lane_index = int(lane_index)
            lane_position = float(lane_position)
            speed = max(float(speed), 0.0)
            lane_length = self._lane_length(lane_id)
            lane_count = max(len(self.connection_info.edge_lane_ids.get(edge_id, [])), 1)
            dist_to_end = max(lane_length - lane_position, 0.0)
            snapshot = VehicleSnapshot(
                vehicle_id=str(vid), step=int(step), edge_id=edge_id, lane_id=lane_id,
                lane_index=lane_index, lane_count=lane_count, lane_position=lane_position,
                lane_length=lane_length, dist_to_end=dist_to_end, speed=speed,
            )
            self._step_snapshot_cache[vid] = snapshot
            return snapshot
        except traci.TraCIException:
            return None

    def _get_step_context(self, vid, edge_id, destination, step, snapshot=None):
        step = int(step)
        key = (str(vid), edge_id, destination, step)
        cached = self._step_context_cache.get(key)
        if cached is not None:
            return cached
        context = self.decision_engine.build_context(
            str(vid),
            edge_id,
            destination,
            step,
            snapshot=snapshot,
        )
        self._step_context_cache[key] = context
        return context

    def should_control_vehicle(self, vehicle_id, vehicle, step):
        vid = str(vehicle_id)
        step = int(step)
        self._prepare_step_cache(step)
        self._ensure_vehicle_subscription(vid)

        # Avoid duplicate same-step work
        if self._last_control_step.get(vid) == step:
            return False

        result = self._step_vehicle_results.get(vid) or {}
        try:
            current_edge = result.get(tc.VAR_ROAD_ID)
            if current_edge is None:
                current_edge = traci.vehicle.getRoadID(vid)
        except traci.TraCIException:
            return False

        if current_edge not in self.connection_info.edge_index_dict:
            return False
        if current_edge == vehicle.destination:
            return False

        # Observe and proactive pendings need active same-edge monitoring.
        if vid in self._pending_decisions:
            pending = self._pending_decisions[vid]
            if self.shared_policy.pending_requires_active_same_edge_monitoring(pending):
                self._metrics["step_control_pending"] += 1
                return True
            if current_edge != pending.decision_edge:
                self._metrics["step_control_pending"] += 1
                return True
            snapshot = self._snapshot_vehicle(vid, current_edge, step)
            if snapshot is not None:
                context = self._get_step_context(vid, current_edge, vehicle.destination, step, snapshot=snapshot)
                if context.commit_window and pending.intended_action not in context.lane_feasible_now_actions:
                    self._metrics["step_control_pending"] += 1
                    return True
                pending_age = self.decision_engine.pending_age_steps(pending, step)
                lane_now_pending = self.shared_policy.pending_resolution_mode(pending) == "lane_now"
                low_speed = float(snapshot.speed) <= float(self.shared_policy.lane_now_replan_low_speed_mps)
                stale_lane_now_age = (
                    pending_age >= int(self.decision_engine.route_pending_hard_timeout_steps)
                    and low_speed
                )
                very_old_lane_now_age = pending_age >= int(self.shared_policy.lane_now_stale_timeout_max_age_steps)
                low_speed_lane_now = (
                    pending_age >= int(self.shared_policy.lane_now_replan_min_age_steps)
                    and low_speed
                )
                if lane_now_pending and (stale_lane_now_age or very_old_lane_now_age or low_speed_lane_now):
                    self._metrics["step_control_pending"] += 1
                    return True

        # Preserve normal edge-change-driven control.
        if current_edge != vehicle.current_edge:
            self._metrics["step_control_edge_change"] += 1
            return True

        snapshot = self._snapshot_vehicle(vid, current_edge, step)
        if snapshot is None:
            return False

        context = self._get_step_context(vid, current_edge, vehicle.destination, step, snapshot=snapshot)

        # Match training more closely:
        # - forced decisions are always evaluated
        # - open decisions are always evaluated
        # - everything else is left to edge-change / pending monitoring
        decision_mode = self.shared_policy.classify_decision(context)
        if decision_mode.mode in {"forced", "open"}:
            return True
        return False
    #----------------------------------------------------------------------


    def make_decisions(self, vehicles, connection_info: ConnectionInfo):
        local_targets = {}
        step = int(traci.simulation.getTime())
        self._prepare_step_cache(step)
        self._refresh_density_stats(step)
        open_decision_batch = []

        def process_selected_action(vid, vehicle, start_edge, context, recent, action_idx):
            if action_idx not in context.available_actions:
                self._metrics["overrides"] += 1
                self._metrics["impossible_action_overrides"] += 1
                return
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
                fallback_action = self.shared_policy.select_fallback_action(
                    context,
                    blocked_action=action_idx,
                    destination=vehicle.destination,
                    recent_history=recent,
                    distance_fn=self._dist_to_dest,
                    edge_density_fn=self._edge_density,
                )
                if fallback_action is None:
                    return
                action_idx = fallback_action
                self._metrics["fallback_selected_total"] += 1
                if action_idx in context.lane_feasible_now_actions:
                    self._metrics["fallback_selected_lane_now"] += 1

            selected_next_edge = self.decision_engine.get_next_edge(start_edge, action_idx)
            if selected_next_edge is None:
                return

            lane_change_requested = False
            if action_idx not in context.lane_feasible_now_actions:
                cooldown_until = self._lane_change_cooldown.get((vid, start_edge), -1)
                if step < cooldown_until:
                    self._metrics["overrides"] += 1
                    self._metrics["cooldown_replans_blocked"] += 1
                    action_idx = self.shared_policy.select_fallback_action(
                        context,
                        blocked_action=action_idx,
                        destination=vehicle.destination,
                        recent_history=recent,
                        distance_fn=self._dist_to_dest,
                        edge_density_fn=self._edge_density,
                        lane_now_only=True,
                    )
                    if action_idx is None:
                        return
                    self._metrics["fallback_selected_total"] += 1
                    if action_idx in context.lane_feasible_now_actions:
                        self._metrics["fallback_selected_lane_now"] += 1
                else:
                    lane_change_requested, lane_change_ok = self.decision_engine.try_request_lane_change(context, action_idx)
                    observe_meta = self.decision_engine.start_lane_change_observe(context, action_idx, step, lane_change_requested, lane_change_ok)
                    decision_id = self._next_decision_id()
                    origin_mode = ("lane_now" if action_idx in context.lane_feasible_now_actions else "proactive")
                    self._pending_decisions[vid] = self.shared_policy.build_observe_pending(
                        state=None,
                        action_idx=action_idx,
                        intended_next_edge=selected_next_edge,
                        decision_edge=start_edge,
                        step=step,
                        destination=vehicle.destination,
                        context=context,
                        lane_change_requested=lane_change_requested,
                        decision_id=decision_id,
                        origin_mode=origin_mode,
                        action_source="policy",
                        observe_metadata=observe_meta,
                        decision_open_recorded=True,
                    )
                    self._metrics["lane_change_observe_started"] += 1
                    self._metrics["deferred_lane_change_actions"] += 1
                    return

            full_route, committed_next_edge, apply_error = self.decision_engine.apply_route_decision(
                str(vid),
                start_edge,
                action_idx,
                vehicle.destination,
            )
            if apply_error:
                self._metrics["overrides"] += 1
                self._metrics["distance_overrides"] += 1
                return

            next_edge = committed_next_edge
            if next_edge:
                decision_id = self._next_decision_id()
                origin_mode = ("lane_now" if action_idx in context.lane_feasible_now_actions else "proactive")
                self._pending_decisions[vid] = self.shared_policy.build_route_pending(
                    state=None,
                    action_idx=action_idx,
                    committed_next_edge=next_edge,
                    decision_edge=start_edge,
                    step=step,
                    destination=vehicle.destination,
                    context=context,
                    lane_change_requested=lane_change_requested,
                    decision_id=decision_id,
                    origin_mode=origin_mode,
                    action_source="policy",
                    full_route=full_route,
                    decision_open_recorded=True,
                )
            # Route already committed directly via shared apply_route_decision.

        if not hasattr(self, "_debug_net_checked"):
            self._debug_net_checked = True
            # print("[DEBUG] has self.net:", hasattr(self, "net"))
            # print("[DEBUG] self.net type:", type(self.net))

        for vehicle in vehicles:
            start_edge = vehicle.current_edge
            if vehicle.destination == start_edge:
                continue

            vid = vehicle.vehicle_id
            self._last_control_step[vid] = step
            if vid not in self._recent_edges:
                self._recent_edges[vid] = deque(maxlen=self.loop_window)
            recent = list(self._recent_edges.get(vid, deque(maxlen=self.loop_window)))
            prev_seen_edge = self._last_observed_edge.get(vid)
            edge_changed_runtime = (prev_seen_edge != start_edge)

            self._visit_count.setdefault(vid, {})
            self._best_dist.setdefault(vid, float("inf"))

            if prev_seen_edge is None or edge_changed_runtime:
                pending_before_transition = self._pending_decisions.get(vid)
                if pending_before_transition is not None:
                    self._record_committed_cyclic_revisit(
                        pending_before_transition,
                        start_edge,
                        vehicle.destination,
                        recent,
                    )
                self._record_runtime_loop_signals(start_edge, vehicle.destination, recent)
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
                    obs_context = self._get_step_context(str(vid), start_edge, vehicle.destination, step, snapshot=obs_snapshot)
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
                        self._pending_decisions[vid] = self.shared_policy.promote_observe_success(
                            pending,
                            context=context,
                            step=step,
                            committed_next_edge=committed_next_edge,
                            full_route=full_route,
                            state=None,
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
                    action_idx = self.shared_policy.select_fallback_action(
                        obs_context,
                        blocked_action=pending.intended_action,
                        destination=vehicle.destination,
                        recent_history=recent,
                        distance_fn=self._dist_to_dest,
                        edge_density_fn=self._edge_density,
                        lane_now_only=True,
                    )
                    if action_idx is None:
                        continue
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
                    self._pending_decisions[vid] = self.shared_policy.build_route_pending(
                        state=None,
                        action_idx=action_idx,
                        committed_next_edge=committed_next_edge,
                        decision_edge=start_edge,
                        step=step,
                        destination=vehicle.destination,
                        context=obs_context,
                        lane_change_requested=False,
                        decision_id=str((pending.metadata or {}).get("decision_id", pending.decision_id or self._next_decision_id())),
                        origin_mode=str((pending.metadata or {}).get("decision_origin_mode", pending.decision_origin_mode or "proactive")),
                        action_source="observe_fallback",
                        full_route=full_route,
                        decision_open_recorded=True,
                    )
                    continue
                continue

            snapshot = self._snapshot_vehicle(vid, start_edge, step)
            if snapshot is None:
                continue
            context = self._get_step_context(str(vid), start_edge, vehicle.destination, step, snapshot=snapshot)
            decision_mode = self.shared_policy.classify_decision(context)
            if decision_mode.mode == "forced":
                process_selected_action(vid, vehicle, start_edge, context, recent, int(decision_mode.action))
                continue
            elif decision_mode.mode != "open":
                continue
            else:
                state = self.getState(vid, start_edge, vehicle.destination, context=context)
                cooldown_until = self._lane_change_cooldown.get((vid, start_edge), -1)
                cooldown_active = step < cooldown_until
                policy_actions = self.shared_policy.policy_action_candidates(
                    context,
                    recent_history=recent,
                    cooldown_active=cooldown_active,
                    destination=vehicle.destination,
                    distance_fn=self._dist_to_dest,
                    edge_density_fn=self._edge_density,
                    metrics=self._metrics,
                    distance_slack=self.score_slack,
                )
                policy_actions = self.shared_policy.rank_policy_actions(
                    context=context,
                    actions=policy_actions,
                    destination=vehicle.destination,
                    distance_fn=self._dist_to_dest,
                    edge_density_fn=self._edge_density,
                    recent_history=recent,
                )
                policy_actions = self._force_stale_lane_now_replan_if_available(
                    vid,
                    start_edge,
                    step,
                    context,
                    policy_actions,
                )
                self._metrics["decisions"] += 1
                open_decision_batch.append(
                    {
                        "vid": vid,
                        "vehicle": vehicle,
                        "start_edge": start_edge,
                        "context": context,
                        "recent": recent,
                        "state": state,
                        "policy_actions": policy_actions,
                    }
                )

        if open_decision_batch:
            action_indices = self.act_batch(
                [entry["state"] for entry in open_decision_batch],
                [entry["policy_actions"] for entry in open_decision_batch],
            )
            for entry, action_idx in zip(open_decision_batch, action_indices):
                process_selected_action(
                    entry["vid"],
                    entry["vehicle"],
                    entry["start_edge"],
                    entry["context"],
                    entry["recent"],
                    action_idx,
                )

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
        act_values = self.model(state, training=False).numpy()[0]
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

    def act_batch(self, states, available_actions_batch):
        if not states:
            return []
        state_batch = np.array([state[0] for state in states], dtype=np.float32)
        q_batch = self.model(state_batch, training=False).numpy()
        results = []
        for q_values, available_actions in zip(q_batch, available_actions_batch):
            available = list(available_actions) if available_actions is not None else []
            if not available:
                results.append(int(np.argmax(q_values)))
                continue
            masked = np.full_like(q_values, -1e9)
            valid_idx = np.asarray(available, dtype=np.intp)
            masked[valid_idx] = q_values[valid_idx]
            results.append(int(np.argmax(masked)))
        return results
    # this function gives the current state of the vehicle based on the state size
    def getState(self, vehicle_id, edge_now, destination_edge, context=None):
        en = edge_now
        current_step = int(context.step) if context is not None else int(
            self._step_cache_step if self._step_cache_step is not None else traci.simulation.getTime()
        )
        if context is None:
            self._prepare_step_cache(current_step)
            context = self._get_step_context(str(vehicle_id), en, destination_edge, current_step)

        vehicle_obj = self.vehicles.get(str(vehicle_id))
        deadline_features = [0.0, 0.0, 0.0]
        if vehicle_obj is not None:
            now = float(current_step)
            deadline_window = max(float(vehicle_obj.deadline) - float(vehicle_obj.start_time), 1.0)
            time_left = max(float(vehicle_obj.deadline) - float(now), 0.0)
            elapsed = max(float(now) - float(vehicle_obj.start_time), 0.0)
            urgency = 1.0 - min(time_left / deadline_window, 1.0)
            deadline_features = [
                min(time_left / deadline_window, 1.0),
                min(elapsed / deadline_window, 1.0),
                urgency,
            ]

        density_cache = {}
        eta_cache = {}

        def cached_edge_density(edge_id):
            if edge_id is None:
                return float("inf")
            cached = density_cache.get(edge_id)
            if cached is not None:
                return cached
            cached = float(self._edge_density(edge_id))
            density_cache[edge_id] = cached
            return cached

        def cached_eta(edge_id, destination):
            if edge_id is None:
                return float("inf")
            key = (edge_id, destination)
            cached = eta_cache.get(key)
            if cached is not None:
                return cached
            cached = float(self._estimate_eta(edge_id, destination))
            eta_cache[key] = cached
            return cached

        social_cost_cache = {}
        for action_idx in context.edge_valid_actions:
            stats = self.shared_policy.action_corridor_stats(
                edge_id=en,
                action_idx=action_idx,
                destination=destination_edge,
                edge_density_fn=cached_edge_density,
                distance_fn=self._dist_to_dest,
                eta_fn=cached_eta,
            )
            if stats is None:
                social_cost_cache[action_idx] = float("inf")
                continue
            current_distance = self._dist_to_dest(en, destination_edge)
            social_cost = float(stats.score)
            if np.isfinite(current_distance) and np.isfinite(stats.next_distance):
                if stats.next_distance >= (current_distance - 1.0):
                    social_cost += 0.45
                loop_distance_slack = float(getattr(self.decision_engine, "loop_distance_slack", 30.0))
                if stats.best_distance <= (current_distance - max(8.0, 0.25 * loop_distance_slack)):
                    social_cost -= 0.10
            social_cost_cache[action_idx] = float(max(social_cost, 0.0))

        return self.shared_policy.encode_state(
            edge_id=en,
            destination_edge=destination_edge,
            context=context,
            use_compact_state=self.use_compact_state,
            edge_embedding_fn=self._get_edge_embedding,
            edge_density_fn=cached_edge_density,
            eta_fn=cached_eta,
            social_cost_fn=lambda current_edge, action_idx, destination: social_cost_cache.get(action_idx, float("inf")),
            global_density_stats=(float(self._density_mean), float(self._density_std)) if self.use_compact_state else None,
            edge_lane_meters_fn=self._edge_lane_meters,
            step=current_step,
            vehicle_start_time=(float(vehicle_obj.start_time) if vehicle_obj is not None else None),
            edge_index_lookup=self.connection_info.edge_index_dict,
            legacy_aux_features=deadline_features,
            legacy_density_values=[cached_edge_density(edge_id) for edge_id in self.connection_info.edge_list] if not self.use_compact_state else None,
        )
