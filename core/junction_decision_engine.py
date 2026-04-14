from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import traci


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

    def __init__(self, connection_info, net, direction_choices):
        self.connection_info = connection_info
        self.net = net
        self.direction_choices = direction_choices

        self.base_reaction_distance = 25.0
        self.reaction_time_s = 1.3
        self.commit_time_s = 0.8
        self.lane_change_margin_m = 24.0
        self.commit_min_distance = 14.0
        self.default_fragment_horizon_m = 180.0
        self.pending_timeout_steps = 6
        self.lane_change_defer_limit = 2
        self.lane_change_request_duration = 20
        self.short_wait_lane_change_steps = 2
        self.lane_change_stall_speed_mps = 1.2
        self.low_speed_no_far_lane_change_mps = 1.5
        self.max_shift_when_low_speed = 1

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
                if shift >= 999:
                    continue
                # Prefer "safe move now" over late/far lane-change plans.
                if speed <= self.low_speed_no_far_lane_change_mps and shift > self.max_shift_when_low_speed:
                    continue
                if dist_to_end <= (reaction_distance + commit_distance) and shift > 0:
                    continue
                if lane_change_budget >= shift * self.lane_change_margin_m and dist_to_end >= reaction_distance:
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

    def lane_feasible_fallback_actions(self, context: DecisionContext, blocked_action: Optional[int] = None) -> List[int]:
        candidates = sorted(set(context.lane_feasible_now_actions))
        if blocked_action is None:
            return candidates
        return [a for a in candidates if a != blocked_action] or candidates

    def try_request_lane_change(self, context: DecisionContext, action_idx: int, duration: Optional[int] = None) -> Tuple[bool, bool]:
        if duration is None:
            duration = self.lane_change_request_duration
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

    def required_lane_shift_for_action(self, context: DecisionContext, action_idx: int) -> int:
        if action_idx in context.lane_feasible_now_actions:
            return 0
        return int(context.required_lane_shift.get(action_idx, 999))

    def evaluate_lane_change_progress(self, attempt: Dict[str, object], context: DecisionContext, action_idx: int) -> Tuple[bool, Optional[str], Dict[str, object]]:
        same_edge_steps = int(attempt.get("same_edge_steps", 0)) + 1
        prev_lane_index = int(attempt.get("last_lane_index", context.lane_index))
        prev_shift = int(attempt.get("last_required_shift", self.required_lane_shift_for_action(context, action_idx)))
        prev_dist = float(attempt.get("last_dist_to_end", context.dist_to_end))
        curr_shift = self.required_lane_shift_for_action(context, action_idx)
        lane_unchanged = (context.lane_index == prev_lane_index)
        shift_not_decreasing = curr_shift >= prev_shift
        low_speed = context.speed < self.lane_change_stall_speed_mps
        dist_shrinking = context.dist_to_end < prev_dist
        in_commit_window_not_feasible = bool(context.commit_window and action_idx not in context.lane_feasible_now_actions)

        no_progress = (
            (same_edge_steps >= self.short_wait_lane_change_steps and lane_unchanged and shift_not_decreasing)
            or (same_edge_steps >= self.short_wait_lane_change_steps and low_speed and shift_not_decreasing)
            or (dist_shrinking and in_commit_window_not_feasible)
            or in_commit_window_not_feasible
        )

        reason = None
        if no_progress:
            if in_commit_window_not_feasible:
                reason = "commit_window_not_lane_feasible"
            elif low_speed and shift_not_decreasing:
                reason = "low_speed_lane_change_stall"
            elif lane_unchanged and shift_not_decreasing:
                reason = "lane_change_no_progress"
            else:
                reason = "lane_change_stall"

        updated_attempt = {
            "same_edge_steps": same_edge_steps,
            "last_lane_index": context.lane_index,
            "last_required_shift": curr_shift,
            "last_dist_to_end": context.dist_to_end,
        }
        return bool(no_progress), reason, updated_attempt

    def should_abort_pending_decision(self, pending: PendingDecision, context: DecisionContext, step: int) -> Tuple[bool, Optional[str], Dict[str, object]]:
        same_edge_steps = int(pending.metadata.get("same_edge_pending_steps", 0)) + 1
        pending.metadata["same_edge_pending_steps"] = same_edge_steps
        if not pending.lane_change_requested:
            return False, None, {"same_edge_pending_steps": same_edge_steps}

        attempt = {
            "same_edge_steps": same_edge_steps,
            "last_lane_index": pending.metadata.get("last_lane_index", context.lane_index),
            "last_required_shift": pending.metadata.get(
                "last_required_shift",
                self.required_lane_shift_for_action(context, pending.intended_action),
            ),
            "last_dist_to_end": pending.metadata.get("last_dist_to_end", context.dist_to_end),
        }
        abort, reason, updated = self.evaluate_lane_change_progress(attempt, context, pending.intended_action)
        pending.metadata.update(updated)
        return abort, reason, {"same_edge_pending_steps": same_edge_steps}

    def get_next_edge(self, edge_id: str, action_idx: int) -> Optional[str]:
        outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {})
        direction = self.direction_choices[action_idx]
        return outgoing.get(direction)

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

        try:
            from_edge = self.net.getEdge(immediate)
            to_edge = self.net.getEdge(destination)
            path_edges, _ = self.net.getShortestPath(from_edge, to_edge, vClass="passenger")
        except Exception:
            path_edges = None

        if not path_edges:
            if immediate == destination:
                return [immediate], immediate, None
            return [], None, "unreachable_destination"

        path_ids = [edge.getID() for edge in path_edges]
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
