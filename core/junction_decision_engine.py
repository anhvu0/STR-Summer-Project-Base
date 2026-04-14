"""Unified decision engine used by both training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import traci

from core.route_loop_safety import summarize_loop_risk


@dataclass
class DecisionContext:
    vehicle_id: str
    edge_id: str
    destination: str
    step: int
    lane_id: str
    lane_index: int
    lane_count: int
    dist_to_end: float
    speed: float
    edge_valid_actions: List[int]
    lane_feasible_now_actions: List[int]
    reachable_with_lane_change_actions: List[int]


@dataclass
class RankedAction:
    action_idx: int
    next_edge: Optional[str]
    total_score: float
    lane_feasible_now: bool
    reachable_with_lane_change: bool
    progress_delta: float
    congestion_cost: float
    loop_risk: float
    trap_risk: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class PendingDecision:
    state: object
    action: int
    decision_edge: str
    intended_next_edge: Optional[str]
    decision_step: int
    destination: str
    fallback_applied: bool = False
    metadata: Dict[str, object] = field(default_factory=dict)


class JunctionDecisionEngine:
    """Feasibility + action ranking + contiguous route commitment."""

    def __init__(self, connection_info, net, direction_choices):
        self.connection_info = connection_info
        self.net = net
        self.direction_choices = direction_choices
        self.distance_worsening_slack = 35.0
        self.min_lane_change_buffer = 25.0

    def _lane_data(self, vehicle_id: str, edge_id: str) -> Tuple[str, int, int, float, float]:
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        lane_idx = int(traci.vehicle.getLaneIndex(vehicle_id))
        lane_count = max(len(self.connection_info.edge_lane_ids.get(edge_id, [])), 1)
        lane_len = traci.lane.getLength(lane_id)
        lane_pos = traci.vehicle.getLanePosition(vehicle_id)
        dist_to_end = max(float(lane_len - lane_pos), 0.0)
        speed = max(float(traci.vehicle.getSpeed(vehicle_id)), 0.0)
        return lane_id, lane_idx, lane_count, dist_to_end, speed

    def _dist_to_dest(self, edge_id: str, dest_id: str) -> float:
        try:
            path = self.net.getShortestPath(self.net.getEdge(edge_id), self.net.getEdge(dest_id), vClass="passenger")
            if not path or not path[0]:
                return float("inf")
            return float(path[1])
        except Exception:
            return float("inf")

    def _estimate_eta(self, edge_id: str, dest_id: str, assumed_speed: float = 8.5) -> float:
        dist = self._dist_to_dest(edge_id, dest_id)
        if dist == float("inf"):
            return float("inf")
        return dist / max(assumed_speed, 1.0)

    def _edge_out_degree(self, edge_id: str) -> int:
        return len(self.connection_info.outgoing_edges_dict.get(edge_id, {}))

    def build_context(self, vehicle_id: str, edge_id: str, destination: str, step: int) -> DecisionContext:
        lane_id, lane_idx, lane_count, dist_to_end, speed = self._lane_data(vehicle_id, edge_id)
        outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {})
        edge_valid = [i for i, d in enumerate(self.direction_choices) if d in outgoing]

        lane_out = self.connection_info.lane_outgoing_edges_dict.get(lane_id, {})
        lane_now = [i for i, d in enumerate(self.direction_choices) if d in lane_out and d in outgoing]

        lane_ids = self.connection_info.edge_lane_ids.get(edge_id, [])
        reachable: List[int] = []
        for idx in edge_valid:
            direction = self.direction_choices[idx]
            if idx in lane_now:
                reachable.append(idx)
                continue
            for lane_candidate in lane_ids:
                if direction in self.connection_info.lane_outgoing_edges_dict.get(lane_candidate, {}):
                    reachable.append(idx)
                    break

        return DecisionContext(
            vehicle_id=str(vehicle_id),
            edge_id=edge_id,
            destination=destination,
            step=int(step),
            lane_id=lane_id,
            lane_index=lane_idx,
            lane_count=lane_count,
            dist_to_end=dist_to_end,
            speed=speed,
            edge_valid_actions=sorted(set(edge_valid)),
            lane_feasible_now_actions=sorted(set(lane_now)),
            reachable_with_lane_change_actions=sorted(set(reachable)),
        )

    def get_next_edge(self, edge_id: str, action_idx: int) -> Optional[str]:
        outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {})
        return outgoing.get(self.direction_choices[action_idx])

    def rank_actions(
        self,
        context: DecisionContext,
        recent_history: Deque[str],
        distance_cache: Dict[Tuple[str, str], float],
    ) -> List[RankedAction]:
        ranked: List[RankedAction] = []
        current_dist = self._dist_to_dest(context.edge_id, context.destination)
        edge_out_degree = {
            edge: self._edge_out_degree(edge)
            for edge in set(recent_history) | set(self.connection_info.outgoing_edges_dict.get(context.edge_id, {}).values())
        }

        for action_idx in context.edge_valid_actions:
            next_edge = self.get_next_edge(context.edge_id, action_idx)
            if not next_edge:
                continue

            lane_now = action_idx in context.lane_feasible_now_actions
            reachable = action_idx in context.reachable_with_lane_change_actions

            key = (next_edge, context.destination)
            next_dist = distance_cache.get(key)
            if next_dist is None:
                next_dist = self._dist_to_dest(next_edge, context.destination)
                distance_cache[key] = next_dist

            progress_delta = 0.0 if current_dist == float("inf") else (current_dist - next_dist)
            loop = summarize_loop_risk(recent_history, next_edge, edge_out_degree)
            loop_penalty = (4.0 if loop.aba_bounce else 0.0) + (3.0 if loop.short_cycle else 0.0) + (2.0 if loop.dead_end_reentry else 0.0)
            trap_risk = 3.0 if (self._edge_out_degree(next_edge) <= 1 and next_edge != context.destination) else 0.0
            congestion = traci.edge.getLastStepVehicleNumber(next_edge) / max(self.connection_info.edge_length_dict.get(next_edge, 10.0), 10.0)

            score = 0.0
            reasons: List[str] = []
            if lane_now:
                score += 3.0
                reasons.append("lane_now")
            elif reachable and context.dist_to_end >= self.min_lane_change_buffer:
                score += 1.5
                reasons.append("lane_change_possible")
            else:
                score -= 2.5
                reasons.append("lane_unreachable")

            if next_dist != float("inf"):
                score += min(progress_delta / 40.0, 3.0)
            else:
                score -= 8.0
                reasons.append("unreachable")

            if progress_delta < -self.distance_worsening_slack:
                score -= 2.0
                reasons.append("distance_worsening")

            score -= loop_penalty
            score -= trap_risk
            score -= 2.0 * congestion

            ranked.append(
                RankedAction(
                    action_idx=action_idx,
                    next_edge=next_edge,
                    total_score=score,
                    lane_feasible_now=lane_now,
                    reachable_with_lane_change=reachable,
                    progress_delta=progress_delta,
                    congestion_cost=float(congestion),
                    loop_risk=float(loop_penalty),
                    trap_risk=float(trap_risk),
                    reasons=reasons,
                )
            )

        ranked.sort(key=lambda x: x.total_score, reverse=True)
        return ranked

    def fallback_action(self, ranked: List[RankedAction], context: DecisionContext) -> Optional[RankedAction]:
        if not ranked:
            return None

        def pick(predicate):
            for cand in ranked:
                if predicate(cand):
                    return cand
            return None

        lane_now = pick(lambda c: c.lane_feasible_now and c.loop_risk <= 6.0 and c.next_edge is not None)
        if lane_now:
            return lane_now

        lane_change = pick(
            lambda c: c.reachable_with_lane_change and context.dist_to_end >= self.min_lane_change_buffer and c.loop_risk <= 6.0
        )
        if lane_change:
            return lane_change

        safe_progress = pick(lambda c: c.next_edge is not None and c.progress_delta > -self.distance_worsening_slack)
        if safe_progress:
            return safe_progress

        forced = pick(lambda c: c.next_edge is not None)
        return forced

    def build_full_route(self, edge_id: str, action_idx: int, destination: str) -> Tuple[List[str], Optional[str], Optional[str]]:
        immediate = self.get_next_edge(edge_id, action_idx)
        if not immediate:
            return [], None, "invalid_action"

        try:
            path_edges, _ = self.net.getShortestPath(self.net.getEdge(immediate), self.net.getEdge(destination), vClass="passenger")
        except Exception:
            path_edges = None

        if not path_edges:
            if immediate == destination:
                return [edge_id, immediate], immediate, None
            return [], None, "unreachable_destination"

        suffix = [edge.getID() for edge in path_edges]
        full_route = [edge_id] + suffix
        return full_route, immediate, None

    def apply_route_decision(self, vehicle_id: str, edge_id: str, action_idx: int, destination: str):
        full_route, immediate, error = self.build_full_route(edge_id, action_idx, destination)
        if error:
            return None, None, error
        try:
            traci.vehicle.setRoute(vehicle_id, full_route)
            return full_route, immediate, None
        except traci.TraCIException:
            return None, None, "route_apply_failed"

    def direction_masks(self, context: DecisionContext) -> Tuple[List[float], List[float], List[float]]:
        n = len(self.direction_choices)
        return (
            [1.0 if i in context.edge_valid_actions else 0.0 for i in range(n)],
            [1.0 if i in context.lane_feasible_now_actions else 0.0 for i in range(n)],
            [1.0 if i in context.reachable_with_lane_change_actions else 0.0 for i in range(n)],
        )
