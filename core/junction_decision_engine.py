from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import math
import traci
from core.route_loop_safety import transition_signal, would_worsen_distance


@dataclass
class DecisionContext:
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
class PolicyActionDiagnostics:
    hard_invalid_actions: int = 0
    lane_infeasible_available_actions: int = 0
    soft_loop_risk_actions: int = 0
    policy_mask_removed_actions: int = 0
    non_lane_available_masked: int = 0


@dataclass
class PendingDecision:
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

    def __init__(
        self,
        connection_info,
        net,
        direction_choices,
        pending_timeout_steps: int = 18,
        pending_progress_timeout_steps: int = 10,
        observe_steps_min: int = 2,
        observe_steps_max: int = 4,
        observe_low_speed_mps: float = 0.8,
        observe_stall_steps: int = 2,
    ):
        self.connection_info = connection_info
        self.net = net
        self.direction_choices = direction_choices

        self.base_reaction_distance = 25.0
        self.reaction_time_s = 1.3
        self.commit_time_s = 0.8
        self.lane_change_margin_m = 24.0
        self.commit_min_distance = 14.0
        self.default_fragment_horizon_m = 180.0
        self.pending_timeout_steps = max(int(pending_timeout_steps), 1)
        self.lane_change_defer_limit = 4
        self.observe_steps_min = max(int(observe_steps_min), 1)
        self.observe_steps_max = max(int(observe_steps_max), self.observe_steps_min)
        self.observe_low_speed_mps = max(float(observe_low_speed_mps), 0.0)
        self.observe_stall_steps = max(int(observe_stall_steps), 1)
        self.cooldown_steps = 3
        self.pending_progress_timeout_steps = max(int(pending_progress_timeout_steps), 1)
        self.loop_distance_slack = 30.0

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
        try:
            return self.net.getEdge(edge_id).allows("passenger")
        except Exception:
            return False

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
        edge_valid = [i for i, d in enumerate(self.direction_choices) if d in outgoing]
        lane_map = self.connection_info.lane_outgoing_edges_dict.get(lane_id, {})
        lane_now = [i for i, d in enumerate(self.direction_choices) if d in lane_map and d in outgoing]

        lane_ids = self.connection_info.edge_lane_ids.get(edge_id, [])
        reachable = []
        required_shift = {}
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
                reachable.append(idx)

        reaction_distance = max(self.base_reaction_distance, speed * self.reaction_time_s)
        commit_distance = max(self.commit_min_distance, speed * self.commit_time_s)
        commit_window = dist_to_end <= commit_distance

        available = []
        if commit_window:
            available = list(lane_now)
        else:
            lane_change_budget = max(dist_to_end - commit_distance, 0.0)
            for idx in edge_valid:
                if idx in lane_now:
                    available.append(idx)
                    continue
                shift = required_shift.get(idx, 999)
                dynamic_margin = self.lane_change_margin_m * (1.0 + 0.5 * max(0, shift - 1))
                low_speed = speed < 1.2
                aggressive_shift = shift >= 2 and dist_to_end < (dynamic_margin + commit_distance + reaction_distance)
                if low_speed or aggressive_shift:
                    continue
                if shift < 999 and lane_change_budget >= shift * dynamic_margin and dist_to_end >= reaction_distance:
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
        good_motion = context.speed >= self.observe_low_speed_mps and (not context.commit_window or context.dist_to_end > self.commit_min_distance)
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
        if context.commit_window and action_idx not in context.lane_feasible_now_actions:
            return "abort", "commit_window"
        if context.speed < self.observe_low_speed_mps and observe_steps >= 1:
            return "abort", "low_speed"
        if stall_steps >= self.observe_stall_steps:
            return "abort", "no_progress"
        if observe_steps >= observe_limit:
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
    ) -> List[int]:
        candidate_pool = self.lane_feasible_fallback_actions(context, blocked_action=blocked_action)
        if not candidate_pool:
            candidate_pool = self.safe_connected_fallback_actions(context, blocked_action=blocked_action)
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
            next_edge = self.get_next_edge(context.edge_id, action)
            if distance_fn is not None and next_edge is not None:
                next_dist = distance_fn(next_edge, destination)
                if math.isfinite(next_dist):
                    score += min(float(next_dist) / 250.0, 10.0)
                else:
                    score += 25.0
            score += 0.05 * float(context.required_lane_shift.get(action, 0))
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
        block_distance_worsen: bool = False,
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
        if distance_fn is not None:
            current_distance = distance_fn(context.edge_id, destination)
            next_distance = distance_fn(next_edge, destination)
            dist_worsen = would_worsen_distance(
                current_distance,
                next_distance,
                slack=self.loop_distance_slack if distance_slack is None else float(distance_slack),
            )
        blocked = bool(
            signals.get("short_cycle")
            or signals.get("aba_bounce")
            or signals.get("dead_end_reentry")
            or signals.get("long_horizon_loop")
            or signals.get("revisit_without_progress")
            or trap_like
            or (bool(block_distance_worsen) and dist_worsen)
        )
        details = dict(signals)
        details["trap_like_reversal"] = trap_like
        details["distance_worsen"] = dist_worsen
        return (not blocked), details

    def build_policy_action_set(
        self,
        context: DecisionContext,
        destination: str,
        recent_history: List[str],
        cooldown_active: bool,
        distance_fn: Optional[Callable[[str, str], float]] = None,
    ) -> Tuple[List[int], PolicyActionDiagnostics]:
        available_actions = list(context.available_actions)
        if not available_actions:
            return [], PolicyActionDiagnostics(
                hard_invalid_actions=max(len(context.edge_valid_actions), 0),
            )

        lane_now = set(context.lane_feasible_now_actions)
        diagnostics = PolicyActionDiagnostics(
            hard_invalid_actions=max(len(context.edge_valid_actions) - len(available_actions), 0),
            lane_infeasible_available_actions=sum(1 for a in available_actions if a not in lane_now),
        )

        safe_lane_now_actions: List[int] = []
        safe_non_lane_actions: List[int] = []
        filtered_available_actions: List[int] = []
        for action in available_actions:
            safe_ok, _ = self.prefilter_action_for_loops(
                context=context,
                action_idx=action,
                destination=destination,
                recent_history=recent_history,
                distance_fn=distance_fn,
            )
            if not safe_ok:
                diagnostics.soft_loop_risk_actions += 1
                continue
            filtered_available_actions.append(action)
            if action in lane_now:
                safe_lane_now_actions.append(action)
            elif (not cooldown_active) and (not context.commit_window):
                safe_non_lane_actions.append(action)

        if context.commit_window:
            policy_actions = sorted(set(safe_lane_now_actions or filtered_available_actions))
        elif cooldown_active and safe_lane_now_actions:
            policy_actions = sorted(set(safe_lane_now_actions))
        else:
            policy_actions = sorted(set(safe_lane_now_actions + safe_non_lane_actions))
            if not policy_actions:
                policy_actions = sorted(set(filtered_available_actions))
        if not policy_actions:
            policy_actions = sorted(set(available_actions))

        diagnostics.policy_mask_removed_actions = max(len(available_actions) - len(policy_actions), 0)
        diagnostics.non_lane_available_masked = sum(
            1 for action in available_actions if action not in lane_now and action not in set(policy_actions)
        )
        return policy_actions, diagnostics

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

    # Legacy route-fragment builder from split-routing approach.
    # Kept commented for reference; training + inference now use build_full_route()
    # and apply_route_decision() for strict contiguous SUMO route application.
    # def build_route_fragment(self, edge_id: str, action_idx: int, destination: str, horizon_m: Optional[float] = None):
    #     horizon = self.default_fragment_horizon_m if horizon_m is None else float(horizon_m)
    #     immediate = self.get_next_edge(edge_id, action_idx)
    #     if immediate is None:
    #         return [], None, "invalid_action"
    #     if not self._edge_allows_passenger(immediate):
    #         return [], None, "non_passenger_edge"
    #     immediate_outgoing = self.connection_info.outgoing_edges_dict.get(immediate, {})
    #     if immediate != destination and len(immediate_outgoing) == 1 and edge_id in immediate_outgoing.values():
    #         return [], None, "trap_like_reversal"
    #
    #     try:
    #         from_edge = self.net.getEdge(immediate)
    #         to_edge = self.net.getEdge(destination)
    #         path_edges, _ = self.net.getShortestPath(from_edge, to_edge, vClass="passenger")
    #     except Exception:
    #         path_edges = None
    #
    #     if not path_edges:
    #         if immediate == destination:
    #             return [immediate], immediate, None
    #         return [], None, "unreachable_destination"
    #
    #     path_ids = [edge.getID() for edge in path_edges]
    #     fragment = []
    #     cumulative = 0.0
    #     for edge in path_ids:
    #         if not self._edge_allows_passenger(edge):
    #             return [], None, "non_passenger_edge"
    #         fragment.append(edge)
    #         cumulative += float(self.connection_info.edge_length_dict.get(edge, 40.0))
    #         if cumulative >= horizon:
    #             break
    #
    #     if fragment[-1] != destination and destination not in path_ids:
    #         return [], None, "fragment_disconnected"
    #
    #     local_target = fragment[-1]
    #     return fragment, local_target, None

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

        try:
            from_edge = self.net.getEdge(immediate)
            to_edge = self.net.getEdge(destination)
            path_edges, _ = self.net.getShortestPath(from_edge, to_edge, vClass="passenger")
        except Exception:
            path_edges = None

        if not path_edges:
            if immediate == destination:
                return [edge_id, immediate], immediate, None
            return [], None, "unreachable_destination"

        suffix = [edge.getID() for edge in path_edges]
        for edge in suffix:
            if not self._edge_allows_passenger(edge):
                return [], None, "non_passenger_edge"

        full_route = [edge_id] + suffix
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
