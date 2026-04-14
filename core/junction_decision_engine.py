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
        self.change_lane_duration = 20
        self.pending_timeout_steps = 6
        self.lane_change_defer_limit = 2
        self.pending_progress_grace_steps = 1
        self.pending_low_speed_threshold_mps = 1.5
        self.pending_no_progress_lane_steps = 2
        self.commit_guard_extra_distance_m = 28.0
        self.low_speed_lane_change_limit_mps = 1.2
        self.low_speed_max_shift = 1

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

        near_commit_window = dist_to_end <= (commit_distance + self.commit_guard_extra_distance_m)
        low_speed = speed <= self.low_speed_lane_change_limit_mps
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
                if near_commit_window:
                    continue
                if low_speed and shift > self.low_speed_max_shift:
                    continue
                if shift < 999 and lane_change_budget >= shift * self.lane_change_margin_m and dist_to_end >= reaction_distance:
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
            traci.vehicle.changeLane(context.vehicle_id, target_lane, int(duration if duration is not None else self.change_lane_duration))
            return True, True
        except traci.TraCIException:
            return True, False

    def _required_shift_now(self, context: DecisionContext, action_idx: int) -> int:
        if action_idx in context.lane_feasible_now_actions:
            return 0
        return int(context.required_lane_shift.get(action_idx, 999))

    def evaluate_lane_change_progress(self, metadata: Dict[str, object], context: DecisionContext, action_idx: int, step: int) -> Dict[str, object]:
        prev_lane_idx = int(metadata.get("last_lane_index", context.lane_index))
        prev_shift = int(metadata.get("last_required_lane_shift", self._required_shift_now(context, action_idx)))
        prev_dist = float(metadata.get("last_dist_to_end", context.dist_to_end))
        same_edge_steps = int(metadata.get("same_edge_steps", 0)) + 1
        curr_shift = self._required_shift_now(context, action_idx)
        target_lane_reached = action_idx in context.lane_feasible_now_actions
        lane_changed = int(context.lane_index) != prev_lane_idx
        shift_decreased = curr_shift < prev_shift
        dist_shrinking = float(context.dist_to_end) < (prev_dist - 0.25)
        low_speed = float(context.speed) < self.pending_low_speed_threshold_mps
        grace_elapsed = same_edge_steps > self.pending_progress_grace_steps
        stalled_same_lane = grace_elapsed and (not lane_changed) and (not shift_decreased)
        stalled_low_speed = grace_elapsed and low_speed and (not shift_decreased)
        commit_window_blocked = context.commit_window and (not target_lane_reached)
        dist_to_junction_stalling = grace_elapsed and dist_shrinking and (not shift_decreased) and (not target_lane_reached)
        no_progress = commit_window_blocked or stalled_same_lane or stalled_low_speed or dist_to_junction_stalling
        should_abort = no_progress and (not target_lane_reached)
        metadata.update({
            "last_lane_index": int(context.lane_index),
            "last_required_lane_shift": int(curr_shift),
            "last_dist_to_end": float(context.dist_to_end),
            "same_edge_steps": int(same_edge_steps),
            "last_progress_step": int(step if (lane_changed or shift_decreased or target_lane_reached) else metadata.get("last_progress_step", step)),
        })
        return {
            "target_lane_reached": target_lane_reached,
            "commit_window_blocked": commit_window_blocked,
            "no_progress": no_progress,
            "should_abort": should_abort,
            "same_edge_steps": same_edge_steps,
            "stall_low_speed": stalled_low_speed,
            "stall_same_lane": stalled_same_lane,
        }

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
