import math
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import traci
import sumolib


@dataclass
class RoutingCandidate:
    slot: int
    direction: str
    next_edge: str
    path_exists: bool
    lane_supported: bool
    lane_reachable: bool
    min_lane_shift: int
    enough_distance: bool
    pressure: float
    eta: float
    deadline_deficit: float
    loop_risk: float
    fallback_only: bool
    score: float
    valid: bool


class SharedRoutingLogic:
    """Shared candidate generation, state encoding, masking, validation and recovery helpers."""

    def __init__(self, connection_info, slot_count: int = 8, candidate_feature_dim: int = 12):
        self.connection_info = connection_info
        self.slot_count = slot_count
        self.candidate_feature_dim = candidate_feature_dim
        self._net = sumolib.net.readNet(connection_info.net_filename)
        self._distance_cache: Dict[Tuple[str, str], float] = {}
        self._path_cache: Dict[Tuple[str, str], List[str]] = {}
        self._outgoing_cache: Dict[str, Dict[str, str]] = connection_info.outgoing_edges_dict

        # base features in encode_observation:
        # - edge code (3)
        # - destination code (3)
        # - scalar context features (10)
        # total = 16
        self.base_state_size = 16
        self.state_size = self.base_state_size + (self.slot_count * self.candidate_feature_dim) + self.slot_count

    def _edge_code(self, edge_id: str) -> Tuple[float, float, float]:
        idx = float(self.connection_info.edge_index_dict.get(edge_id, 0))
        n = max(len(self.connection_info.edge_list), 1)
        x = 2.0 * math.pi * (idx / n)
        return math.sin(x), math.cos(x), idx / n

    def get_shortest_path(self, from_edge: str, to_edge: str) -> List[str]:
        key = (from_edge, to_edge)
        if key in self._path_cache:
            return self._path_cache[key]
        try:
            f = self._net.getEdge(from_edge)
            t = self._net.getEdge(to_edge)
            path, _ = self._net.getShortestPath(f, t)
            edges = [e.getID() for e in path] if path else []
        except Exception:
            edges = []
        self._path_cache[key] = edges
        return edges

    def distance_to_destination(self, from_edge: str, to_edge: str) -> float:
        key = (from_edge, to_edge)
        if key in self._distance_cache:
            return self._distance_cache[key]
        try:
            f = self._net.getEdge(from_edge)
            t = self._net.getEdge(to_edge)
            _path, dist = self._net.getShortestPath(f, t)
            value = float(dist) if _path else math.inf
        except Exception:
            value = math.inf
        self._distance_cache[key] = value
        return value

    def _lane_stats(self, vehicle_id: str, edge_id: str) -> Tuple[str, int, int, float, float]:
        try:
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            lane_idx = int(traci.vehicle.getLaneIndex(vehicle_id))
            lane_count = max(int(traci.edge.getLaneNumber(edge_id)), 1)
            lane_len = float(traci.lane.getLength(lane_id))
            lane_pos = float(traci.vehicle.getLanePosition(vehicle_id))
            return lane_id, lane_idx, lane_count, max(lane_len - lane_pos, 0.0), lane_len
        except Exception:
            return "", 0, 1, 0.0, 1.0

    def _edge_density(self, edge_id: str) -> float:
        length = max(float(self.connection_info.edge_length_dict.get(edge_id, 5.0)), 5.0)
        try:
            count = float(traci.edge.getLastStepVehicleNumber(edge_id))
        except Exception:
            count = float(self.connection_info.edge_vehicle_count.get(edge_id, 0))
        return count / length

    def is_decision_point(self, vehicle_id: str, edge_id: str) -> bool:
        outgoing = self._outgoing_cache.get(edge_id, {})
        if len(outgoing) <= 1:
            return False
        _lid, _lidx, lane_count, dist_to_end, _ = self._lane_stats(vehicle_id, edge_id)
        speed = max(float(traci.vehicle.getSpeed(vehicle_id)), 1.0)
        edge_len = max(float(self.connection_info.edge_length_dict.get(edge_id, 40.0)), 40.0)
        complexity = 1.0 + (0.2 * max(len(outgoing) - 2, 0)) + (0.08 * max(lane_count - 1, 0))
        threshold = min(260.0, max(35.0, 0.65 * edge_len + speed * complexity))
        return dist_to_end <= threshold

    def build_candidates(
        self,
        vehicle,
        vehicle_id: str,
        edge_id: str,
        destination_edge: str,
        recent_edges: deque,
        recent_transitions: deque,
        mismatch_count: int,
    ) -> Tuple[List[RoutingCandidate], np.ndarray]:
        now = float(traci.simulation.getTime())
        lane_id, lane_idx, _lane_count, dist_to_end, _ = self._lane_stats(vehicle_id, edge_id)
        speed = max(float(traci.vehicle.getSpeed(vehicle_id)), 1.0)
        time_left = max(float(vehicle.deadline) - now, 0.0)
        deadline_window = max(float(vehicle.deadline) - float(vehicle.start_time), 1.0)
        urgency = 1.0 - min(time_left / deadline_window, 1.0)
        flexibility = 1.0 - urgency

        outgoing = self._outgoing_cache.get(edge_id, {})
        by_next: Dict[str, str] = {}
        for d, nxt in outgoing.items():
            by_next.setdefault(nxt, d)

        options: List[RoutingCandidate] = []
        for next_edge, direction in by_next.items():
            lane_supported = False
            supporting_lane_idxs: List[int] = []
            if lane_id and direction in self.connection_info.lane_outgoing_edges_dict.get(lane_id, {}):
                lane_supported = True
            for idx, lid in enumerate(self.connection_info.edge_lane_ids.get(edge_id, [])):
                lane_map = self.connection_info.lane_outgoing_edges_dict.get(lid, {})
                if lane_map.get(direction) == next_edge:
                    supporting_lane_idxs.append(idx)

            min_shift = min([abs(lane_idx - i) for i in supporting_lane_idxs], default=99)
            required = 8.0 + (12.0 * min(min_shift, 5)) + speed * 1.2
            enough_distance = dist_to_end >= required
            lane_reachable = lane_supported or (min_shift < 99 and enough_distance)

            dist = self.distance_to_destination(next_edge, destination_edge)
            path_exists = math.isfinite(dist)
            eta = (dist / max(speed, 7.0)) if path_exists else math.inf
            deficit = max(eta - time_left, 0.0) if math.isfinite(eta) else 999.0
            pressure = self._edge_density(next_edge)

            transition = (edge_id, next_edge)
            loop_hits = sum(1 for e in recent_edges if e == next_edge)
            transition_hits = sum(1 for t in recent_transitions if t == transition)
            loop_risk = float(loop_hits + (2 * transition_hits))

            fallback_only = (not lane_reachable) or (not path_exists)
            score = (
                8.0 * deficit
                + (1.0 + flexibility) * 4.0 * pressure
                + 2.0 * loop_risk
                + 0.3 * min(min_shift, 6)
                + (6.0 if fallback_only else 0.0)
                + 1.5 * mismatch_count
            )

            options.append(
                RoutingCandidate(
                    slot=-1,
                    direction=direction,
                    next_edge=next_edge,
                    path_exists=path_exists,
                    lane_supported=lane_supported,
                    lane_reachable=lane_reachable,
                    min_lane_shift=min_shift if min_shift < 99 else 6,
                    enough_distance=enough_distance,
                    pressure=pressure,
                    eta=eta if math.isfinite(eta) else 999.0,
                    deadline_deficit=min(deficit, 999.0),
                    loop_risk=loop_risk,
                    fallback_only=fallback_only,
                    score=score,
                    valid=(path_exists and lane_reachable),
                )
            )

        options.sort(key=lambda c: c.score)
        candidates: List[RoutingCandidate] = []
        mask = np.zeros(self.slot_count, dtype=np.float32)
        for i in range(self.slot_count):
            if i < len(options):
                c = options[i]
                c.slot = i
                candidates.append(c)
                mask[i] = 1.0 if c.valid else 0.0
            else:
                candidates.append(
                    RoutingCandidate(i, "", "", False, False, False, 0, False, 0.0, 999.0, 999.0, 0.0, True, 9999.0, False)
                )

        if mask.sum() == 0 and options:
            # force a safe fallback slot for progress; still marked with high score and penalties in reward.
            candidates[0] = options[0]
            candidates[0].slot = 0
            mask[0] = 1.0

        return candidates, mask

    def encode_observation(
        self,
        vehicle,
        vehicle_id: str,
        edge_id: str,
        destination_edge: str,
        candidates: List[RoutingCandidate],
        mask: np.ndarray,
        loop_score: float,
        mismatch_count: int,
    ) -> np.ndarray:
        now = float(traci.simulation.getTime())
        lane_id, lane_idx, lane_count, dist_to_end, lane_len = self._lane_stats(vehicle_id, edge_id)
        speed = max(float(traci.vehicle.getSpeed(vehicle_id)), 0.0)
        time_left = max(float(vehicle.deadline) - now, 0.0)
        deadline_window = max(float(vehicle.deadline) - float(vehicle.start_time), 1.0)
        urgency = 1.0 - min(time_left / deadline_window, 1.0)

        state = []
        state.extend(self._edge_code(edge_id))
        state.extend(self._edge_code(destination_edge))
        state.extend([
            min(time_left / deadline_window, 1.0),
            min(max(now - float(vehicle.start_time), 0.0) / deadline_window, 1.0),
            urgency,
            lane_idx / max(lane_count - 1, 1),
            min(lane_count, 6) / 6.0,
            min(dist_to_end / max(lane_len, 1.0), 1.0),
            min(speed / 25.0, 1.0),
            min(self._edge_density(edge_id), 2.0),
            min(loop_score / 6.0, 1.0),
            min(float(mismatch_count) / 6.0, 1.0),
        ])

        for cand in candidates:
            state.extend([
                1.0 if cand.valid else 0.0,
                1.0 if cand.path_exists else 0.0,
                1.0 if cand.lane_supported else 0.0,
                1.0 if cand.lane_reachable else 0.0,
                min(cand.min_lane_shift, 6) / 6.0,
                1.0 if cand.enough_distance else 0.0,
                min(cand.pressure, 2.0),
                min(cand.eta / 300.0, 1.0),
                min(cand.deadline_deficit / 300.0, 1.0),
                min(cand.loop_risk / 6.0, 1.0),
                1.0 if cand.fallback_only else 0.0,
                min(cand.score / 100.0, 1.0),
            ])

        state.extend(mask.tolist())
        arr = np.asarray(state, dtype=np.float32)
        if arr.shape[0] != self.state_size:
            raise ValueError(f"Encoded state size mismatch: expected {self.state_size}, got {arr.shape[0]}")
        return arr.reshape(1, -1)

    def choose_safe_candidate(self, candidates: List[RoutingCandidate], mask: np.ndarray, selected_slot: int) -> RoutingCandidate:
        if 0 <= selected_slot < len(candidates) and mask[selected_slot] > 0.0:
            return candidates[selected_slot]
        for i, c in enumerate(candidates):
            if mask[i] > 0.0:
                return c
        return candidates[0]

    def best_supporting_lane_index(self, edge_id: str, direction: str, next_edge: str, current_lane_idx: int) -> Optional[int]:
        best_idx = None
        best_shift = 999
        for idx, lane_id in enumerate(self.connection_info.edge_lane_ids.get(edge_id, [])):
            lane_map = self.connection_info.lane_outgoing_edges_dict.get(lane_id, {})
            if lane_map.get(direction) == next_edge:
                shift = abs(idx - current_lane_idx)
                if shift < best_shift:
                    best_shift = shift
                    best_idx = idx
        return best_idx

    def maintain_lane_alignment(self, vehicle_id: str, edge_id: str, candidate: RoutingCandidate) -> bool:
        try:
            # Lane control is a low-level execution helper: RL selects turn/next-edge, and this
            # method persistently nudges toward a supporting lane over multiple simulation steps.
            lane_id, lane_idx, _cnt, dist_to_end, _len = self._lane_stats(vehicle_id, edge_id)
            if candidate.lane_supported:
                return True
            target_lane = self.best_supporting_lane_index(edge_id, candidate.direction, candidate.next_edge, lane_idx)
            if target_lane is None:
                return False
            if lane_idx == target_lane:
                return True
            required_dist = 8.0 + 12.0 * abs(target_lane - lane_idx)
            if dist_to_end < required_dist:
                return False
            step_lane = lane_idx + (1 if target_lane > lane_idx else -1)
            traci.vehicle.changeLane(vehicle_id, int(step_lane), 2)
            return True
        except Exception:
            return False

    def apply_lane_alignment(self, vehicle_id: str, edge_id: str, candidate: RoutingCandidate) -> bool:
        return self.maintain_lane_alignment(vehicle_id, edge_id, candidate)

    def plan_local_target(self, current_edge: str, next_edge: str, destination_edge: str, horizon_m: float = 300.0) -> str:
        if next_edge == destination_edge:
            return destination_edge
        if not math.isfinite(self.distance_to_destination(next_edge, destination_edge)):
            return current_edge

        full_path = self.get_shortest_path(next_edge, destination_edge)
        if not full_path:
            return current_edge

        walk = [current_edge]
        if full_path[0] != next_edge:
            walk.append(next_edge)
        walk.extend(full_path)

        seen = set([current_edge])
        dist = 0.0
        target = current_edge
        for edge in walk[1:]:
            if edge in seen:
                break
            seen.add(edge)
            target = edge
            dist += float(self.connection_info.edge_length_dict.get(edge, 30.0))
            if target == destination_edge:
                break
            if dist >= horizon_m:
                break
        return target

    def compute_transition_reward(
        self,
        vehicle,
        prev_edge: str,
        current_edge: str,
        selected_candidate: RoutingCandidate,
        expected_next_edge: str,
        arrived: bool,
        teleported: bool,
        step: int,
        loop_hits: int,
        transition_hits: int,
        mismatch_happened: bool,
    ) -> Tuple[float, bool, Dict[str, float]]:
        reward = -0.5
        done = False
        terms: Dict[str, float] = {}

        prev_dist = self.distance_to_destination(prev_edge, vehicle.destination)
        curr_dist = self.distance_to_destination(current_edge, vehicle.destination)
        progress = 0.0
        if math.isfinite(prev_dist) and math.isfinite(curr_dist):
            progress = (prev_dist - curr_dist)
            reward += 0.02 * progress
        terms["progress"] = progress

        now = float(step)
        time_left = max(float(vehicle.deadline) - now, 0.0)
        curr_eta = (curr_dist / 8.0) if math.isfinite(curr_dist) else 999.0
        prev_eta = (prev_dist / 8.0) if math.isfinite(prev_dist) else 999.0
        prev_def = max(prev_eta - (time_left + 1.0), 0.0)
        curr_def = max(curr_eta - time_left, 0.0)
        deficit_term = 1.2 * (prev_def - curr_def) - 0.6 * curr_def
        reward += deficit_term
        terms["deadline"] = deficit_term

        urgency = 1.0 - min(time_left / max(float(vehicle.deadline) - float(vehicle.start_time), 1.0), 1.0)
        flexibility = 1.0 - urgency
        ext_penalty = -(0.8 + flexibility) * selected_candidate.pressure
        reward += ext_penalty
        terms["externality"] = ext_penalty

        loop_penalty = -1.8 * loop_hits - 2.5 * transition_hits
        reward += loop_penalty
        terms["loop"] = loop_penalty

        if selected_candidate.fallback_only:
            reward -= 2.0
            terms["fallback_only"] = -2.0

        if mismatch_happened:
            reward -= 4.0
            terms["mismatch"] = -4.0

        if teleported:
            reward -= 20.0
            done = True
            terms["teleport"] = -20.0

        if not math.isfinite(curr_dist):
            reward -= 15.0
            done = True
            terms["no_path"] = -15.0

        if arrived:
            done = True
            if current_edge == vehicle.destination:
                reward += 18.0
                terms["arrival"] = 18.0
                if step <= vehicle.deadline:
                    reward += 12.0
                    terms["on_time"] = 12.0
                else:
                    late = min(float(step - vehicle.deadline), 300.0)
                    reward -= 6.0 + (0.08 * late)
                    terms["late"] = -(6.0 + 0.08 * late)
            else:
                reward -= 10.0
                terms["bad_arrival"] = -10.0

        return float(np.clip(reward, -40.0, 40.0)), done, terms
