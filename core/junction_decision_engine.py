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
    chosen_action: int
    executed_action: Optional[int]
    intended_next_edge: Optional[str]
    decision_edge: str
    decision_step: int
    destination: str
    context: DecisionContext
    lane_change_requested: bool
    lane_change_ok: bool = True
    intervention_type: str = "none"
    committed: bool = False
    commit_step: Optional[int] = None
    route_fragment: List[str] = field(default_factory=list)


@dataclass
class ActionExecutionResult:
    chosen_action: int
    executed_action: Optional[int]
    executed_next_edge: Optional[str]
    intervention_type: str
    lane_change_requested: bool
    lane_change_ok: bool
    committed_decision: bool
    route_fragment: List[str] = field(default_factory=list)
    local_target: Optional[str] = None
    error: Optional[str] = None


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
        self.reaction_time_s = 1.6
        self.commit_time_s = 1.0
        self.lane_change_margin_m = 28.0
        self.commit_min_distance = 14.0
        self.default_fragment_horizon_m = 180.0

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

    def route_matches_expected(self, pending: PendingDecision, actual_next_edge: str) -> bool:
        if pending.intended_next_edge == actual_next_edge:
            return True
        if pending.route_fragment and actual_next_edge in pending.route_fragment:
            return True
        return False

    def ordered_fallback_actions(self, context: DecisionContext, exclude_action: Optional[int] = None) -> List[int]:
        candidates = [a for a in context.available_actions if a != exclude_action]
        lane_now = set(context.lane_feasible_now_actions)
        return sorted(
            candidates,
            key=lambda action: (
                0 if action in lane_now else 1,
                context.required_lane_shift.get(action, 999),
                action,
            ),
        )

    def attempt_execute_action(
        self,
        context: DecisionContext,
        chosen_action: int,
        destination: str,
        fallback_actions: Optional[List[int]] = None,
    ) -> ActionExecutionResult:
        chosen_lane_change_requested = False
        chosen_lane_change_ok = True
        chosen_error = None

        candidates = [chosen_action]
        fallback_order = fallback_actions if fallback_actions is not None else self.ordered_fallback_actions(context, exclude_action=chosen_action)
        for action in fallback_order:
            if action != chosen_action:
                candidates.append(action)

        for idx, action in enumerate(candidates):
            lane_change_requested, lane_change_ok = self.try_request_lane_change(context, action)
            if idx == 0:
                chosen_lane_change_requested = lane_change_requested
                chosen_lane_change_ok = lane_change_ok
                if lane_change_requested and not lane_change_ok:
                    chosen_error = "impossible_lane_change"
                    continue
            elif lane_change_requested and not lane_change_ok:
                continue

            next_edge = self.get_next_edge(context.edge_id, action)
            if next_edge is None:
                if idx == 0:
                    chosen_error = "invalid_action"
                continue

            fragment, local_target, frag_error = self.build_route_fragment(context.edge_id, action, destination)
            if frag_error or local_target is None:
                if idx == 0:
                    chosen_error = frag_error or "fragment_error"
                continue

            intervention = "none" if action == chosen_action else "fallback_applied"
            return ActionExecutionResult(
                chosen_action=chosen_action,
                executed_action=action,
                executed_next_edge=next_edge,
                intervention_type=intervention,
                lane_change_requested=chosen_lane_change_requested,
                lane_change_ok=chosen_lane_change_ok,
                committed_decision=True,
                route_fragment=list(fragment),
                local_target=local_target,
                error=None,
            )

        return ActionExecutionResult(
            chosen_action=chosen_action,
            executed_action=None,
            executed_next_edge=None,
            intervention_type="skip_infeasible_action",
            lane_change_requested=chosen_lane_change_requested,
            lane_change_ok=chosen_lane_change_ok,
            committed_decision=False,
            route_fragment=[],
            local_target=None,
            error=chosen_error or "no_safe_fallback",
        )

    def can_commit_provisional(
        self,
        context: DecisionContext,
        executed_action: Optional[int],
        decision_step: int,
        current_step: int,
    ) -> bool:
        if executed_action is None:
            return False
        if int(current_step) <= int(decision_step):
            return False
        return executed_action in context.lane_feasible_now_actions

    def direction_masks(self, context: DecisionContext):
        edge_mask = [1.0 if i in context.edge_valid_actions else 0.0 for i in range(len(self.direction_choices))]
        lane_mask = [1.0 if i in context.lane_feasible_now_actions else 0.0 for i in range(len(self.direction_choices))]
        reach_mask = [1.0 if i in context.reachable_with_lane_change_actions else 0.0 for i in range(len(self.direction_choices))]
        avail_mask = [1.0 if i in context.available_actions else 0.0 for i in range(len(self.direction_choices))]
        return edge_mask, lane_mask, reach_mask, avail_mask
