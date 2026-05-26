from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import math
import traci
from core.route_loop_safety import transition_signal, would_worsen_distance, short_horizon_trap_score


@dataclass
class DecisionContext:
    """
    Snapshot of one strategic decision point at a specific step.

    Terminology:
    - commit_window=True means lane-feasible-now only; proactive lane-change actions are rejected.
    - available_actions is the final action set after feasibility filters.
    - This object is a gauge/state snapshot (not an event counter).
    """
    vehicle_id: str
    edge_id: str
    destination: str
    step: int
    speed: float
    lane_id: str
    lane_index: int
    lane_count: int
    dist_to_end: float
    edge_valid_actions: List[int]
    lane_feasible_now_actions: List[int]
    reachable_with_lane_change_actions: List[int]
    available_actions: List[int]
    required_lane_shift: Dict[int, int] = field(default_factory=dict)
    commit_window: bool = False
    forced_action: Optional[int] = None
    branch_with_choice: bool = False
    skip_reason: Optional[str] = None


@dataclass
class PendingDecision:
    """
    Lifecycle record for one strategic decision_id while it is unresolved.

    A strategic decision may move across phases (observe_lane_change -> route_pending),
    but it keeps the same decision_id and decision_origin_mode so telemetry counts it once
    for opened/finalized semantics.
    """
    state: object
    intended_action: int
    intended_next_edge: str
    decision_edge: str
    decision_step: int
    last_credit_edge: str
    last_credit_step: int
    destination: str
    context: DecisionContext
    lane_change_requested: bool
    decision_id: str = ""
    decision_origin_mode: str = "lane_now"
    decision_current_phase: str = "route_pending"
    decision_open_recorded: bool = False
    route_fragment: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class VehicleSnapshot:
    vehicle_id: str
    step: int
    edge_id: str
    lane_id: str
    lane_index: int
    lane_count: int
    lane_position: float
    lane_length: float
    dist_to_end: float
    speed: float


class JunctionDecisionEngine:
    """
    Shared decision feasibility and route-fragment planner for training + inference.
    """

    def __init__(self, connection_info, net, direction_choices):
        self.connection_info = connection_info
        self.net = net
        self.direction_choices = direction_choices
        self._direction_to_index = {
            direction: idx for idx, direction in enumerate(self.direction_choices)
        }

        self.base_reaction_distance = 25.0
        self.reaction_time_s = 1.3
        self.commit_time_s = 0.45
        self.lane_change_margin_m = 10.0
        self.commit_min_distance = 8.0
        self.default_fragment_horizon_m = 180.0
        self.pending_timeout_steps = 32
        self.lane_change_defer_limit = 6
        self.observe_steps_min = 2
        self.observe_steps_max = 8
        self.observe_low_speed_mps = 0.5
        self.observe_stall_steps = 5
        self.cooldown_steps = 3
        self.observe_timeout_steps = 16
        self.route_pending_stall_steps = 8
        self.route_pending_hard_timeout_steps = 60
        self.route_pending_progress_eps_m = 2.0
        self.route_pending_lane_progress_eps = 0.15
        self.route_pending_no_progress_window_steps = 5
        self.pending_progress_timeout_steps = 32
        self.loop_distance_slack = 30.0
        self.proactive_extra_buffer_m = 6.0
        self.proactive_safety_margin_m = 8.0
        self.cooldown_after_abort_extra_steps = 2
        self.cooldown_after_timeout_extra_steps = 4
        self._edge_allows_passenger_cache = {}
        self._shortest_path_suffix_cache = {}
        self._edge_valid_actions_by_edge = {}
        self._lane_now_actions_by_lane = {}
        self._required_lane_shift_by_lane = {}
        self._reachable_actions_by_lane = {}
        self._precompute_static_lane_action_metadata()

    def _precompute_static_lane_action_metadata(self):
        outgoing_lookup = self.connection_info.outgoing_edges_dict
        lane_outgoing_lookup = self.connection_info.lane_outgoing_edges_dict
        edge_lane_ids = self.connection_info.edge_lane_ids

        for edge_id, outgoing in outgoing_lookup.items():
            edge_valid = tuple(
                sorted(
                    self._direction_to_index[direction]
                    for direction in outgoing.keys()
                    if direction in self._direction_to_index
                )
            )
            self._edge_valid_actions_by_edge[edge_id] = edge_valid

            lane_ids = edge_lane_ids.get(edge_id, [])
            direction_candidate_lane_indices = {}
            for target_lane_idx, lane_candidate in enumerate(lane_ids):
                lane_candidate_map = lane_outgoing_lookup.get(lane_candidate, {})
                for direction in lane_candidate_map.keys():
                    action_idx = self._direction_to_index.get(direction)
                    if action_idx is None or direction not in outgoing:
                        continue
                    direction_candidate_lane_indices.setdefault(action_idx, []).append(target_lane_idx)

            for lane_idx, lane_id in enumerate(lane_ids):
                lane_map = lane_outgoing_lookup.get(lane_id, {})
                lane_now = tuple(
                    sorted(
                        self._direction_to_index[direction]
                        for direction in lane_map.keys()
                        if direction in outgoing and direction in self._direction_to_index
                    )
                )
                required_shift = {}
                for action_idx in edge_valid:
                    candidate_indices = direction_candidate_lane_indices.get(action_idx, [])
                    if not candidate_indices:
                        continue
                    required_shift[action_idx] = int(
                        min(abs(target_lane_idx - lane_idx) for target_lane_idx in candidate_indices)
                    )
                self._lane_now_actions_by_lane[lane_id] = lane_now
                self._required_lane_shift_by_lane[lane_id] = required_shift
                self._reachable_actions_by_lane[lane_id] = tuple(sorted(required_shift.keys()))

    def _lane_data(self, vehicle_id: str, edge_id: str, snapshot: Optional[VehicleSnapshot] = None):
        if snapshot is not None:
            return (
                snapshot.lane_id,
                int(snapshot.lane_index),
                max(int(snapshot.lane_count), 1),
                max(float(snapshot.dist_to_end), 0.0),
                max(float(snapshot.speed), 0.0),
            )
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        lane_idx = traci.vehicle.getLaneIndex(vehicle_id)
        lane_count = max(len(self.connection_info.edge_lane_ids.get(edge_id, [])), 1)
        lane_len = traci.lane.getLength(lane_id)
        lane_pos = traci.vehicle.getLanePosition(vehicle_id)
        dist_to_end = max(lane_len - lane_pos, 0.0)
        speed = max(traci.vehicle.getSpeed(vehicle_id), 0.0)
        return lane_id, lane_idx, lane_count, dist_to_end, speed

    def _edge_allows_passenger(self, edge_id: str) -> bool:
        cached = self._edge_allows_passenger_cache.get(edge_id)
        if cached is not None:
            return bool(cached)
        try:
            allows = bool(self.net.getEdge(edge_id).allows("passenger"))
        except Exception:
            allows = False
        self._edge_allows_passenger_cache[edge_id] = allows
        return allows

    def build_context(
        self,
        vehicle_id: str,
        edge_id: str,
        destination: str,
        step: int,
        snapshot: Optional[VehicleSnapshot] = None,
    ) -> DecisionContext:
        lane_id, lane_idx, lane_count, dist_to_end, speed = self._lane_data(vehicle_id, edge_id, snapshot=snapshot)

        outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {})
        edge_valid = list(self._edge_valid_actions_by_edge.get(edge_id, ()))
        lane_now = list(self._lane_now_actions_by_lane.get(lane_id, ()))
        required_shift = dict(self._required_lane_shift_by_lane.get(lane_id, {}))
        reachable = list(self._reachable_actions_by_lane.get(lane_id, ()))

        if not edge_valid:
            edge_valid = [i for i, d in enumerate(self.direction_choices) if d in outgoing]
        if not lane_now:
            lane_map = self.connection_info.lane_outgoing_edges_dict.get(lane_id, {})
            lane_now = [i for i, d in enumerate(self.direction_choices) if d in lane_map and d in outgoing]
        if not required_shift and edge_valid:
            lane_ids = self.connection_info.edge_lane_ids.get(edge_id, [])
            for idx in edge_valid:
                direction = self.direction_choices[idx]
                min_shift = None
                for target_lane_idx, lane_candidate in enumerate(lane_ids):
                    lane_candidate_map = self.connection_info.lane_outgoing_edges_dict.get(lane_candidate, {})
                    if direction in lane_candidate_map:
                        shift = abs(target_lane_idx - lane_idx)
                        min_shift = shift if min_shift is None else min(min_shift, shift)
                if min_shift is not None:
                    required_shift[idx] = int(min_shift)
            reachable = sorted(required_shift.keys())

        reaction_distance = max(self.base_reaction_distance, speed * self.reaction_time_s)
        commit_distance = max(self.commit_min_distance, speed * self.commit_time_s)
        commit_window = dist_to_end <= commit_distance

        available = []

        if commit_window:
            # Hard commit window: lane-feasible-now actions only.
            available = list(lane_now)
        else:
            lane_change_budget = max(dist_to_end - commit_distance, 0.0)
            for idx in edge_valid:
                if idx in lane_now:
                    available.append(idx)
                    continue
                shift = required_shift.get(idx, 999)
                if shift not in (1, 2):
                    continue
                if shift == 1:
                    required_budget = self.lane_change_margin_m
                    strong_threshold = (
                        commit_distance
                        + max(self.proactive_extra_buffer_m, 0.35 * self.lane_change_margin_m)
                        + self.proactive_safety_margin_m
                    )
                    if lane_change_budget < required_budget:
                        continue
                    if dist_to_end >= max(0.75 * reaction_distance, strong_threshold):
                        available.append(idx)
                    continue

                required_budget = 1.6 * self.lane_change_margin_m
                strong_threshold = (
                    commit_distance
                    + self.proactive_extra_buffer_m
                    + (1.10 * self.lane_change_margin_m)
                    + self.proactive_safety_margin_m
                )
                if lane_change_budget < required_budget:
                    continue
                if dist_to_end >= max(1.10 * reaction_distance, strong_threshold):
                    available.append(idx)

        available = sorted(set(available))

        skip_reason = None
        forced_action = None
        branch_with_choice = len(available) > 1
        if len(edge_valid) == 0:
            skip_reason = "no_branch"
        elif len(edge_valid) == 1:
            forced_action = edge_valid[0]
            skip_reason = "forced_single_path"
        elif len(available) == 0:
            skip_reason = "too_late_or_unreachable"
        elif len(available) == 1:
            forced_action = available[0]
            skip_reason = "forced_by_lane_commit"

        return DecisionContext(
            vehicle_id=vehicle_id,
            edge_id=edge_id,
            destination=destination,
            step=int(step),
            speed=float(speed),
            lane_id=lane_id,
            lane_index=int(lane_idx),
            lane_count=int(lane_count),
            dist_to_end=float(dist_to_end),
            edge_valid_actions=edge_valid,
            lane_feasible_now_actions=lane_now,
            reachable_with_lane_change_actions=sorted(set(reachable)),
            available_actions=available,
            required_lane_shift=required_shift,
            commit_window=commit_window,
            forced_action=forced_action,
            branch_with_choice=branch_with_choice,
            skip_reason=skip_reason,
        )

    def is_decision_open(self, context: DecisionContext) -> bool:
        return context.branch_with_choice and len(context.available_actions) > 1

    def pending_age_steps(self, pending: PendingDecision, step: int) -> int:
        return max(int(step) - int(pending.decision_step), 0)

    def should_timeout_pending(self, pending: PendingDecision, step: int, max_age_steps: Optional[int] = None) -> bool:
        threshold = self.pending_timeout_steps if max_age_steps is None else int(max_age_steps)
        return self.pending_age_steps(pending, step) >= max(threshold, 1)

    def cooldown_after_pending_release(self, timeout: bool = False) -> int:
        base = int(max(self.cooldown_steps, 1))
        bonus = self.cooldown_after_timeout_extra_steps if timeout else self.cooldown_after_abort_extra_steps
        return int(base + max(int(bonus), 0))

    def pending_progress_update(
        self,
        pending: PendingDecision,
        context: DecisionContext,
        step: int,
    ) -> Tuple[Dict[str, object], Dict[str, object]]:
        metadata = pending.metadata if isinstance(pending.metadata, dict) else {}
        pending.metadata = metadata
        best_dist = float(metadata.get("best_dist_to_end", context.dist_to_end))
        last_progress_step = int(metadata.get("last_progress_step", pending.decision_step))
        best_lane_pos = float(metadata.get("best_lane_position", 0.0))
        lane_position = max(float(metadata.get("lane_position_now", 0.0)), 0.0)

        progress_eps = float(self.route_pending_progress_eps_m)
        lane_progress_eps = float(self.route_pending_lane_progress_eps)
        current_shift = int(context.required_lane_shift.get(pending.intended_action, 99))
        prior_shift = int(metadata.get("last_required_shift", current_shift))
        resolution_mode = str(metadata.get("decision_resolution_mode", pending.decision_origin_mode or "proactive"))
        is_proactive_pending = resolution_mode != "lane_now"

        if is_proactive_pending:
            strong_dist_eps = max(float(progress_eps) * 4.0, float(self.lane_change_margin_m) * 0.8, 8.0)
            dist_progress = context.dist_to_end <= (best_dist - strong_dist_eps)
            lane_now_progress = pending.intended_action in context.lane_feasible_now_actions and prior_shift > 0
            made_progress = bool(dist_progress or lane_now_progress)
        else:
            shift_progress = current_shift < prior_shift
            lane_now_progress = pending.intended_action in context.lane_feasible_now_actions and prior_shift > 0
            dist_progress = context.dist_to_end <= (best_dist - progress_eps)
            lane_pos_progress = lane_position >= (best_lane_pos + lane_progress_eps)
            made_progress = bool(dist_progress or shift_progress or lane_now_progress or lane_pos_progress)

        if made_progress:
            last_progress_step = int(step)
            best_dist = min(best_dist, float(context.dist_to_end))
            best_lane_pos = max(best_lane_pos, lane_position)
        else:
            best_dist = min(best_dist, float(context.dist_to_end))
            best_lane_pos = max(best_lane_pos, lane_position)

        metadata["best_dist_to_end"] = float(best_dist)
        metadata["last_progress_step"] = int(last_progress_step)
        metadata["best_lane_position"] = float(best_lane_pos)
        metadata["last_required_shift"] = int(current_shift)
        metadata["last_seen_lane_index"] = int(context.lane_index)

        return metadata, {
            "made_progress": made_progress,
            "current_shift": int(current_shift),
            "last_progress_step": int(last_progress_step),
        }
    def lane_change_observe_limit(self, context: DecisionContext) -> int:
        limit = self.observe_steps_min
        if context.speed >= 8.0 and context.dist_to_end >= 55.0:
            limit += 1
        if context.speed >= 14.0 and context.dist_to_end >= 95.0:
            limit += 1
        return int(max(self.observe_steps_min, min(limit, self.observe_steps_max)))

    def start_lane_change_observe(
        self,
        context: DecisionContext,
        action_idx: int,
        step: int,
        lane_change_sent: bool,
        lane_change_ok: bool,
    ) -> Dict[str, object]:
        return {
            "phase": "observe_lane_change",
            "observe_action": int(action_idx),
            "observe_started_step": int(step),
            "observe_steps": 0,
            "observe_limit": self.lane_change_observe_limit(context),
            "observe_last_lane_index": int(context.lane_index),
            "observe_last_required_shift": int(context.required_lane_shift.get(action_idx, 99)),
            "observe_stall_steps": 0,
            "lane_change_requested_once": bool(lane_change_sent),
            "lane_change_request_ok": bool(lane_change_ok),
        }

    def evaluate_lane_change_observation(
        self,
        observe_meta: Dict[str, object],
        context: DecisionContext,
        action_idx: int,
    ) -> Tuple[str, Optional[str]]:
        current_shift = int(context.required_lane_shift.get(action_idx, 99))
        last_lane = int(observe_meta.get("observe_last_lane_index", context.lane_index))
        last_shift = int(observe_meta.get("observe_last_required_shift", current_shift))
        observe_steps = int(observe_meta.get("observe_steps", 0)) + 1
        observe_limit = int(observe_meta.get("observe_limit", self.observe_steps_min))
        stall_steps = int(observe_meta.get("observe_stall_steps", 0))

        toward_target = current_shift < last_shift
        lane_changed = context.lane_index != last_lane
        feasible_now = action_idx in context.lane_feasible_now_actions
        good_motion = context.speed >= self.observe_low_speed_mps
        progressing = toward_target or lane_changed or feasible_now or good_motion
        if progressing:
            stall_steps = 0
        else:
            stall_steps += 1

        observe_meta["observe_steps"] = observe_steps
        observe_meta["observe_last_lane_index"] = int(context.lane_index)
        observe_meta["observe_last_required_shift"] = int(current_shift)
        observe_meta["observe_stall_steps"] = int(stall_steps)

        if feasible_now:
            return "success", None

        commit_window_grace = (
            current_shift <= 1
            and context.dist_to_end >= max(self.commit_min_distance + 2.0, 6.0)
        )

        if context.commit_window and action_idx not in context.lane_feasible_now_actions and not commit_window_grace:
            return "abort", "commit_window"

        if context.speed < self.observe_low_speed_mps and observe_steps >= 2 and stall_steps >= 2:
            return "abort", "low_speed"

        if stall_steps >= self.observe_stall_steps:
            return "abort", "no_progress"

        if observe_steps >= observe_limit and current_shift >= last_shift and not lane_changed:
            return "abort", "no_progress"

        return "continue", None

    def lane_feasible_fallback_actions(self, context: DecisionContext, blocked_action: Optional[int] = None) -> List[int]:
        candidates = sorted(set(context.lane_feasible_now_actions))
        if blocked_action is None:
            return candidates
        return [a for a in candidates if a != blocked_action] or candidates

    def safe_connected_fallback_actions(self, context: DecisionContext, blocked_action: Optional[int] = None) -> List[int]:
        candidates = []
        for action in context.available_actions:
            if blocked_action is not None and action == blocked_action:
                continue
            next_edge = self.get_next_edge(context.edge_id, action)
            if next_edge is None:
                continue
            if self._edge_allows_passenger(next_edge):
                candidates.append(action)
        return sorted(set(candidates))

    def ranked_fallback_actions(
        self,
        context: DecisionContext,
        destination: str,
        recent_history: List[str],
        blocked_action: Optional[int] = None,
        distance_fn: Optional[Callable[[str, str], float]] = None,
        candidate_actions: Optional[List[int]] = None,
    ) -> List[int]:
        lane_now_candidates = self.lane_feasible_fallback_actions(context, blocked_action=blocked_action)
        safe_connected_candidates = self.safe_connected_fallback_actions(context, blocked_action=blocked_action)
        if candidate_actions is not None:
            allowed = set(candidate_actions)
            candidate_pool = sorted(
                set(action for action in lane_now_candidates if action in allowed)
                | set(action for action in safe_connected_candidates if action in allowed)
            )
        else:
            candidate_pool = sorted(set(lane_now_candidates) | set(safe_connected_candidates))
        if not candidate_pool:
            return []

        scored = []
        for action in candidate_pool:
            safe_ok, details = self.prefilter_action_for_loops(
                context=context,
                action_idx=action,
                destination=destination,
                recent_history=recent_history,
                distance_fn=distance_fn,
            )
            score = 0.0
            if not safe_ok:
                score += 50.0
            if details.get("dead_end_reentry"):
                score += 10.0
            if details.get("short_cycle") or details.get("aba_bounce"):
                score += 12.0
            if details.get("trap_like_reversal"):
                score += 8.0
            if details.get("distance_worsen"):
                score += 5.0
            if details.get("recent_revisit_low_progress"):
                score += 16.0
            next_edge = self.get_next_edge(context.edge_id, action)
            if distance_fn is not None and next_edge is not None:
                current_dist = distance_fn(context.edge_id, destination)
                next_dist = distance_fn(next_edge, destination)
                if math.isfinite(next_dist):
                    score += min(float(next_dist) / 250.0, 10.0)
                else:
                    score += 25.0
                if math.isfinite(current_dist) and math.isfinite(next_dist) and next_dist >= (current_dist - 1.0):
                    score += 9.0
                edge_distance_lookup = {
                    edge: distance_fn(edge, destination)
                    for edge in set(recent_history) | {context.edge_id, next_edge}
                }
                trap_score = short_horizon_trap_score(
                    start_edge=context.edge_id,
                    candidate_edge=next_edge,
                    destination=destination,
                    outgoing_lookup=self.connection_info.outgoing_edges_dict,
                    edge_distance_lookup=edge_distance_lookup,
                    recent_history=recent_history,
                    horizon_steps=3,
                    progress_slack=self.loop_distance_slack,
                )
                score += 1.5 * float(trap_score)
                if next_edge in set(recent_history[-6:]):
                    score += 12.0
                out_degree = len(self.connection_info.outgoing_edges_dict.get(next_edge, {}))
                if out_degree == 1 and math.isfinite(current_dist) and math.isfinite(next_dist):
                    if next_dist >= (current_dist - max(6.0, 0.25 * self.loop_distance_slack)):
                        score += 10.0
            score += 0.12 * float(context.required_lane_shift.get(action, 0))
            scored.append((score, action))
        scored.sort(key=lambda x: x[0])
        return [action for _, action in scored]

    def prefilter_action_for_loops(
        self,
        context: DecisionContext,
        action_idx: int,
        destination: str,
        recent_history: List[str],
        distance_fn: Optional[Callable[[str, str], float]] = None,
        distance_slack: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, bool]]:
        next_edge = self.get_next_edge(context.edge_id, action_idx)
        if next_edge is None:
            return False, {"invalid_action": True}
        history_deque = recent_history if isinstance(recent_history, deque) else deque(recent_history, maxlen=max(len(recent_history), 1))
        edge_out_degree = {edge: len(self.connection_info.outgoing_edges_dict.get(edge, {})) for edge in set(history_deque) | {next_edge}}
        edge_distance_lookup = None
        if distance_fn is not None:
            edge_distance_lookup = {}
            for edge in set(history_deque) | {next_edge, context.edge_id}:
                edge_distance_lookup[edge] = distance_fn(edge, destination)
        signals = transition_signal(
            history_deque,
            next_edge,
            edge_out_degree=edge_out_degree,
            edge_distance_lookup=edge_distance_lookup,
            progress_slack=self.loop_distance_slack,
        )
        trap_like = (
            next_edge != destination
            and edge_out_degree.get(next_edge, 0) == 0
            and len(history_deque) > 0
            and history_deque[-1] == context.edge_id
        )
        dist_worsen = False
        recent_revisit_low_progress = False
        if distance_fn is not None:
            current_distance = distance_fn(context.edge_id, destination)
            next_distance = distance_fn(next_edge, destination)
            revisit_slack = self.loop_distance_slack if distance_slack is None else float(distance_slack)
            dist_worsen = would_worsen_distance(
                current_distance,
                next_distance,
                slack=revisit_slack,
            )
            recent_window = list(history_deque)[-6:]
            if next_edge in recent_window:
                if not (math.isfinite(current_distance) and math.isfinite(next_distance)):
                    recent_revisit_low_progress = True
                else:
                    recent_revisit_low_progress = (
                        next_distance >= (current_distance - max(8.0, 0.30 * revisit_slack))
                    )
        blocked = bool(
            signals.get("short_cycle")
            or signals.get("aba_bounce")
            or signals.get("dead_end_reentry")
            or signals.get("long_horizon_loop")
            or signals.get("revisit_without_progress")
            or trap_like
            or dist_worsen
            or recent_revisit_low_progress
        )
        details = dict(signals)
        details["trap_like_reversal"] = trap_like
        details["distance_worsen"] = dist_worsen
        details["recent_revisit_low_progress"] = recent_revisit_low_progress
        return (not blocked), details

    def try_request_lane_change(self, context: DecisionContext, action_idx: int, duration: int = 70) -> Tuple[bool, bool]:
        direction = self.direction_choices[action_idx]
        lane_now_map = self.connection_info.lane_outgoing_edges_dict.get(context.lane_id, {})
        if direction in lane_now_map:
            return False, True

        lane_ids = self.connection_info.edge_lane_ids.get(context.edge_id, [])
        target_lane = None
        for target_lane_idx, lane_candidate in enumerate(lane_ids):
            candidate_map = self.connection_info.lane_outgoing_edges_dict.get(lane_candidate, {})
            if direction in candidate_map:
                target_lane = target_lane_idx
                break
        if target_lane is None:
            return False, False

        try:
            traci.vehicle.changeLane(context.vehicle_id, target_lane, duration)
            return True, True
        except traci.TraCIException:
            return True, False

    def get_next_edge(self, edge_id: str, action_idx: int) -> Optional[str]:
        outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {})
        direction = self.direction_choices[action_idx]
        return outgoing.get(direction)

    def _shortest_path_suffix(self, from_edge_id: str, destination: str) -> Optional[Tuple[str, ...]]:
        key = (from_edge_id, destination)
        if key in self._shortest_path_suffix_cache:
            return self._shortest_path_suffix_cache[key]

        try:
            from_edge = self.net.getEdge(from_edge_id)
            to_edge = self.net.getEdge(destination)
            path_edges, _ = self.net.getShortestPath(from_edge, to_edge, vClass="passenger")
        except Exception:
            path_edges = None

        if not path_edges:
            path_ids = (from_edge_id,) if from_edge_id == destination else None
        else:
            path_ids = tuple(edge.getID() for edge in path_edges)

        self._shortest_path_suffix_cache[key] = path_ids
        return path_ids

    def build_route_fragment(self, edge_id: str, action_idx: int, destination: str, horizon_m: Optional[float] = None):
        horizon = self.default_fragment_horizon_m if horizon_m is None else float(horizon_m)
        immediate = self.get_next_edge(edge_id, action_idx)
        if immediate is None:
            return [], None, "invalid_action"
        if not self._edge_allows_passenger(immediate):
            return [], None, "non_passenger_edge"
        immediate_outgoing = self.connection_info.outgoing_edges_dict.get(immediate, {})
        if immediate != destination and len(immediate_outgoing) == 1 and edge_id in immediate_outgoing.values():
            return [], None, "trap_like_reversal"

        path_ids = self._shortest_path_suffix(immediate, destination)
        if not path_ids:
            return [], None, "unreachable_destination"

        fragment = []
        cumulative = 0.0
        for edge in path_ids:
            if not self._edge_allows_passenger(edge):
                return [], None, "non_passenger_edge"
            fragment.append(edge)
            cumulative += float(self.connection_info.edge_length_dict.get(edge, 40.0))
            if cumulative >= horizon:
                break

        if fragment[-1] != destination and destination not in path_ids:
            return [], None, "fragment_disconnected"

        local_target = fragment[-1]
        return fragment, local_target, None

    def build_full_route(self, edge_id: str, action_idx: int, destination: str):
        """
        Build a contiguous SUMO route that starts at the current edge and follows
        the chosen immediate next edge all the way to the true destination.
        """
        immediate = self.get_next_edge(edge_id, action_idx)
        if immediate is None:
            return [], None, "invalid_action"
        if not self._edge_allows_passenger(immediate):
            return [], None, "non_passenger_edge"

        suffix = self._shortest_path_suffix(immediate, destination)
        if not suffix:
            return [], None, "unreachable_destination"
        for edge in suffix:
            if not self._edge_allows_passenger(edge):
                return [], None, "non_passenger_edge"

        full_route = [edge_id] + list(suffix)
        return full_route, immediate, None

    def apply_route_decision(self, vehicle_id: str, edge_id: str, action_idx: int, destination: str):
        """
        Shared route application for training + inference.
        Applies one contiguous route via setRoute(...) to avoid split
        semantics from mixing short via fragments with global retargeting.
        """
        full_route, immediate, error = self.build_full_route(edge_id, action_idx, destination)
        if error:
            return None, None, error
        try:
            traci.vehicle.setRoute(vehicle_id, full_route)
        except traci.TraCIException:
            return None, None, "route_apply_failed"
        return full_route, immediate, None

    def route_matches_expected(self, pending: PendingDecision, actual_next_edge: str) -> bool:
        if pending.intended_next_edge == actual_next_edge:
            return True
        if pending.route_fragment and actual_next_edge in pending.route_fragment:
            return True
        return False

    def direction_masks(self, context: DecisionContext):
        edge_mask = [1.0 if i in context.edge_valid_actions else 0.0 for i in range(len(self.direction_choices))]
        lane_mask = [1.0 if i in context.lane_feasible_now_actions else 0.0 for i in range(len(self.direction_choices))]
        reach_mask = [1.0 if i in context.reachable_with_lane_change_actions else 0.0 for i in range(len(self.direction_choices))]
        avail_mask = [1.0 if i in context.available_actions else 0.0 for i in range(len(self.direction_choices))]
        return edge_mask, lane_mask, reach_mask, avail_mask
