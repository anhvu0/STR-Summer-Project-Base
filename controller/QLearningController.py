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
from core.shared_decision_policy import SharedDecisionPolicy

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
            active_pending = self.shared_policy.pending_requires_active_same_edge_monitoring(pending)
            context = self.decision_engine.build_context(str(vid), vehicle.current_edge, vehicle.destination, step, snapshot=snapshot)
            release_eval = self.shared_policy.evaluate_route_pending_release(
                pending,
                context=context,
                step=step,
                lane_position_now=float(snapshot.lane_position),
            )
            if release_eval.should_release:
                self._pending_decisions.pop(vid, None)
                self._record_pending_release(release_eval.release_reason)
                if release_eval.release_as_timeout:
                    self._metrics["pending_decision_timeouts"] += 1
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

        # Observe and proactive pendings need active same-edge monitoring.
        if vid in self._pending_decisions:
            pending = self._pending_decisions[vid]
            if self.shared_policy.pending_requires_active_same_edge_monitoring(pending):
                self._metrics["step_control_pending"] += 1
                return True
            if current_edge != pending.decision_edge:
                self._metrics["step_control_pending"] += 1
                return True
            snapshot = self._snapshot_vehicle(vid, current_edge, int(step))
            if snapshot is not None:
                context = self.decision_engine.build_context(
                    vid,
                    current_edge,
                    vehicle.destination,
                    int(step),
                    snapshot=snapshot,
                )
                if context.commit_window and pending.intended_action not in context.lane_feasible_now_actions:
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
            context = self.decision_engine.build_context(str(vid), start_edge, vehicle.destination, step, snapshot=snapshot)

            decision_mode = self.shared_policy.classify_decision(context)
            if decision_mode.mode == "forced":
                action_idx = int(decision_mode.action)
            elif decision_mode.mode != "open":
                continue
            else:
                state = self.getState(vid, start_edge, vehicle.destination, context=context)
                cooldown_until = self._lane_change_cooldown.get((vid, start_edge), -1)
                cooldown_active = step < cooldown_until
                recent = list(self._recent_edges.get(vid, deque(maxlen=self.loop_window)))
                policy_actions = self.shared_policy.policy_action_candidates(
                    context,
                    recent_history=recent,
                    cooldown_active=cooldown_active,
                    destination=vehicle.destination,
                    distance_fn=self._dist_to_dest,
                    metrics=self._metrics,
                    distance_slack=self.score_slack,
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
                fallback_action = self.shared_policy.select_fallback_action(
                    context,
                    blocked_action=action_idx,
                    destination=vehicle.destination,
                    recent_history=recent,
                    distance_fn=self._dist_to_dest,
                )
                if fallback_action is None:
                    continue
                action_idx = self.act(state, available_actions=[fallback_action]) if context.forced_action is None else fallback_action
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
                    action_idx = self.shared_policy.select_fallback_action(
                        context,
                        blocked_action=action_idx,
                        destination=vehicle.destination,
                        recent_history=recent,
                        distance_fn=self._dist_to_dest,
                        lane_now_only=True,
                    )
                    if action_idx is None:
                        continue
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
        if context is None:
            context = self.decision_engine.build_context(str(vehicle_id), en, destination_edge, int(traci.simulation.getTime()))

        vehicle_obj = self.vehicles.get(str(vehicle_id))
        deadline_features = [0.0, 0.0, 0.0]
        if vehicle_obj is not None:
            now = traci.simulation.getTime()
            deadline_window = max(float(vehicle_obj.deadline) - float(vehicle_obj.start_time), 1.0)
            time_left = max(float(vehicle_obj.deadline) - float(now), 0.0)
            elapsed = max(float(now) - float(vehicle_obj.start_time), 0.0)
            urgency = 1.0 - min(time_left / deadline_window, 1.0)
            deadline_features = [
                min(time_left / deadline_window, 1.0),
                min(elapsed / deadline_window, 1.0),
                urgency,
            ]

        return self.shared_policy.encode_state(
            edge_id=en,
            destination_edge=destination_edge,
            context=context,
            use_compact_state=self.use_compact_state,
            edge_embedding_fn=self._get_edge_embedding,
            edge_density_fn=self._edge_density,
            eta_fn=self._estimate_eta,
            social_cost_fn=lambda current_edge, action_idx, destination: (
                (1.25 * float(self._edge_density(self.decision_engine.get_next_edge(current_edge, action_idx))))
                + (0.01 * float(self._estimate_eta(self.decision_engine.get_next_edge(current_edge, action_idx), destination))
                   if self.decision_engine.get_next_edge(current_edge, action_idx) is not None and np.isfinite(self._estimate_eta(self.decision_engine.get_next_edge(current_edge, action_idx), destination))
                   else 20.0)
                if self.decision_engine.get_next_edge(current_edge, action_idx) is not None else float("inf")
            ),
            global_density_stats=self.shared_policy.global_density_stats(self._edge_density, self._edge_lane_meters) if self.use_compact_state else None,
            edge_lane_meters_fn=self._edge_lane_meters,
            step=int(traci.simulation.getTime()),
            vehicle_start_time=(float(vehicle_obj.start_time) if vehicle_obj is not None else None),
            edge_index_lookup=self.connection_info.edge_index_dict,
            legacy_aux_features=deadline_features,
            legacy_density_values=[self._edge_density(edge_id) for edge_id in self.connection_info.edge_list] if not self.use_compact_state else None,
        )


