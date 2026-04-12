from collections import defaultdict, deque
import math
import os

import numpy as np
import sumolib
import traci
from keras.models import load_model
from traci import constants as tc
from xml.dom.minidom import parse

from controller.RouteController import RouteController


def parse_sumocfg(sumocfg_path):
    dom = parse(sumocfg_path)
    net_file = dom.getElementsByTagName('net-file')[0].attributes['value'].nodeValue
    return os.path.join(os.path.dirname(sumocfg_path), net_file)


class QLearningPolicy(RouteController):
    """
    Inference policy aligned with training semantics:
    - decisions are next-edge commitments (not arbitrary direction spam)
    - masked Q selection uses executable candidate edges only
    - loop/oscillation guardrails and destination reachability checks are explicit
    """

    def __init__(self, vehicles, connection_info, model_file, net_xml_file=None):
        super().__init__(connection_info)
        self.model = load_model(model_file)
        self.model_state_size = int(self.model.input_shape[-1])
        self.vehicles = vehicles
        self.net = sumolib.net.readNet(net_xml_file or parse_sumocfg("./configurations/myconfig.sumocfg"))

        # Must match training defaults.
        self.candidate_slots = 6
        self.base_feature_size = 22
        self.candidate_feature_size = 13
        self.expected_state_size = self.base_feature_size + self.candidate_slots * self.candidate_feature_size
        if self.model_state_size != self.expected_state_size:
            raise ValueError(
                f"Model input size {self.model_state_size} does not match expected {self.expected_state_size}. "
                "Retrain model with current training pipeline semantics."
            )

        self.loop_window = 10
        self.speed_norm = 20.0
        self.dist_norm = 300.0
        self.time_norm = 1200.0
        self.max_lane_shift_norm = 4.0
        self.route_retry_cooldown_steps = 20

        self._distance_cache = {}
        self._eta_cache = {}
        self._downstream_path_cache = {}
        self._route_blacklist = {}
        self._recent_edges = defaultdict(lambda: deque(maxlen=self.loop_window))
        self._recent_pairs = defaultdict(lambda: deque(maxlen=self.loop_window))
        self._metrics = defaultdict(int)

    def _edge_from_lane_id(self, lane_id):
        if lane_id and "_" in lane_id:
            return lane_id.rsplit("_", 1)[0]
        return None

    def _get_distance(self, edge_id, destination):
        key = (edge_id, destination)
        if key in self._distance_cache:
            return self._distance_cache[key]
        try:
            path, cost = self.net.getShortestPath(self.net.getEdge(edge_id), self.net.getEdge(destination))
            dist = float(cost) if path is not None else math.inf
        except Exception:
            dist = math.inf
        self._distance_cache[key] = dist
        return dist

    def _estimate_eta(self, edge_id, destination):
        key = (edge_id, destination)
        if key in self._eta_cache:
            return self._eta_cache[key]
        try:
            path, _ = self.net.getShortestPath(self.net.getEdge(edge_id), self.net.getEdge(destination))
            if path is None:
                eta = math.inf
            else:
                free_flow_eta = sum(float(e.getLength()) / max(float(e.getSpeed()), 5.0) for e in path)
                eta = free_flow_eta + max(len(path) - 1, 0) * 2.0 + 6.0 + 0.10 * free_flow_eta
        except Exception:
            eta = math.inf
        self._eta_cache[key] = eta
        return eta

    def _has_downstream_path(self, next_edge, destination):
        key = (next_edge, destination)
        if key not in self._downstream_path_cache:
            self._downstream_path_cache[key] = math.isfinite(self._get_distance(next_edge, destination))
        return self._downstream_path_cache[key]

    def _is_lane_legal_successor(self, current_lane_id, next_edge):
        try:
            for link in traci.lane.getLinks(current_lane_id):
                if not link:
                    continue
                next_lane = link[0]
                if self._edge_from_lane_id(next_lane) == next_edge:
                    return True
        except Exception:
            return False
        return False

    def _candidate_lane_metrics(self, vehicle_id, edge_id, next_edge):
        lane_ids = self.connection_info.edge_lane_ids.get(edge_id, [])
        if not lane_ids:
            return {"target_lanes": [], "min_lane_shifts": 99, "feasible": False, "score": 0.0}

        lane_idx = traci.vehicle.getLaneIndex(vehicle_id)
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        lane_len = traci.lane.getLength(lane_id)
        lane_pos = traci.vehicle.getLanePosition(vehicle_id)
        dist_to_end = max(lane_len - lane_pos, 0.0)
        speed = max(traci.vehicle.getSpeed(vehicle_id), 1.0)

        target_lanes = []
        for idx, ln in enumerate(lane_ids):
            outgoing = self.connection_info.lane_outgoing_edges_dict.get(ln, {})
            if next_edge in outgoing.values():
                target_lanes.append(idx)
        if not target_lanes:
            return {"target_lanes": [], "min_lane_shifts": 99, "feasible": False, "score": 0.0}

        min_shift = min(abs(lane_idx - t) for t in target_lanes)
        est_shift_distance = 18.0 * min_shift
        comfort_budget = max(35.0, speed * 2.3)
        feasible = dist_to_end >= est_shift_distance + 8.0
        score = float(np.clip((dist_to_end - est_shift_distance) / comfort_budget, 0.0, 1.0))
        return {"target_lanes": target_lanes, "min_lane_shifts": min_shift, "feasible": feasible, "score": score}

    def _enumerate_candidates(self, vehicle_id, edge_id, destination, step):
        candidates = []
        outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {})
        for direction, next_edge in outgoing.items():
            blacklist_key = (vehicle_id, edge_id, next_edge)
            if self._route_blacklist.get(blacklist_key, -1) >= step:
                self._metrics["blacklist_hits"] += 1
                continue
            if not self._has_downstream_path(next_edge, destination):
                self._metrics["downstream_missing"] += 1
                continue
            lane_metrics = self._candidate_lane_metrics(vehicle_id, edge_id, next_edge)
            if not lane_metrics["target_lanes"]:
                self._metrics["lane_unreachable"] += 1
                continue
            candidates.append((direction, next_edge, lane_metrics))
        return candidates[: self.candidate_slots]

    def _global_stats(self):
        densities = []
        for edge in self.connection_info.edge_list:
            count = traci.edge.getLastStepVehicleNumber(edge)
            density = count / max(self.connection_info.edge_length_dict.get(edge, 1.0), 1.0)
            densities.append(density)
        if not densities:
            return {"mean_density": 0.0, "std_density": 0.0}
        return {"mean_density": float(np.mean(densities)), "std_density": float(np.std(densities))}

    def _is_pair_oscillation(self, vehicle_id, current_edge, candidate_edge):
        pairs = self._recent_pairs[vehicle_id]
        if not pairs:
            return False
        # Detect repeated A->B->A->B alternation pattern.
        last_pair = pairs[-1]
        return last_pair[0] == candidate_edge and last_pair[1] == current_edge

    def _build_state(self, vehicle_id, vehicle, step, current_edge, candidates, global_stats):
        lane_idx = traci.vehicle.getLaneIndex(vehicle_id)
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        lane_pos = traci.vehicle.getLanePosition(vehicle_id)
        n_lanes = max(traci.edge.getLaneNumber(current_edge), 1)
        dist_to_end = max(traci.lane.getLength(lane_id) - lane_pos, 0.0)
        speed = traci.vehicle.getSpeed(vehicle_id)
        time_left = max(float(vehicle.deadline) - float(step), 0.0)
        elapsed = max(float(step) - float(vehicle.start_time), 0.0)
        window = max(float(vehicle.deadline) - float(vehicle.start_time), 1.0)
        eta_curr = self._estimate_eta(current_edge, vehicle.destination)
        slack = (time_left - eta_curr) if math.isfinite(eta_curr) else -self.time_norm
        urgency = float(np.clip(1.0 - (time_left / window), 0.0, 1.0))
        curr_density = traci.edge.getLastStepVehicleNumber(current_edge) / max(self.connection_info.edge_length_dict.get(current_edge, 1.0), 1.0)
        curr_speed = traci.edge.getLastStepMeanSpeed(current_edge)
        sp_dist = self._get_distance(current_edge, vehicle.destination)

        base = np.array([
            lane_idx / max(n_lanes - 1, 1),
            min(n_lanes, 6) / 6.0,
            min(dist_to_end, self.dist_norm) / self.dist_norm,
            min(speed, self.speed_norm) / self.speed_norm,
            min(time_left, self.time_norm) / self.time_norm,
            min(elapsed / window, 2.0) / 2.0,
            urgency,
            np.clip(slack / self.time_norm, -1.0, 1.0),
            np.clip(curr_density, 0.0, 2.0) / 2.0,
            np.clip(curr_speed / self.speed_norm, 0.0, 1.5) / 1.5,
            0.0,
            min(len(candidates), self.candidate_slots) / float(self.candidate_slots),
            1.0 if current_edge in self._recent_edges[vehicle_id] else 0.0,
            np.clip(eta_curr / self.time_norm if math.isfinite(eta_curr) else 1.0, 0.0, 2.0) / 2.0,
            np.clip(sp_dist / 3000.0 if math.isfinite(sp_dist) else 1.0, 0.0, 1.0),
            np.clip((sp_dist / 120.0) / 20.0 if math.isfinite(sp_dist) else 1.0, 0.0, 1.0),
            np.clip(global_stats["mean_density"], 0.0, 2.0) / 2.0,
            np.clip(global_stats["std_density"], 0.0, 1.0),
            0.0,
            0.0,
            0.0,
            0.0,
        ], dtype=np.float32)

        cand_vec = np.zeros((self.candidate_slots, self.candidate_feature_size), dtype=np.float32)
        mask = np.zeros((self.candidate_slots,), dtype=np.float32)
        for idx, (_direction, next_edge, lane_m) in enumerate(candidates):
            density = traci.edge.getLastStepVehicleNumber(next_edge) / max(self.connection_info.edge_length_dict.get(next_edge, 1.0), 1.0)
            mean_speed = traci.edge.getLastStepMeanSpeed(next_edge)
            eta = self._estimate_eta(next_edge, vehicle.destination)
            deficit = max((eta - time_left), 0.0) if math.isfinite(eta) else self.time_norm
            repeated = 1.0 if next_edge in self._recent_edges[vehicle_id] else 0.0
            oscillating = 1.0 if self._is_pair_oscillation(vehicle_id, current_edge, next_edge) else 0.0

            cand_vec[idx] = np.array([
                1.0,
                np.clip(density, 0.0, 2.0) / 2.0,
                np.clip(mean_speed / self.speed_norm, 0.0, 1.5) / 1.5,
                np.clip(eta / self.time_norm if math.isfinite(eta) else 1.0, 0.0, 2.0) / 2.0,
                np.clip(deficit / self.time_norm, 0.0, 1.0),
                np.clip(max(density - global_stats["mean_density"], 0.0), 0.0, 1.0),
                np.clip(lane_m["min_lane_shifts"] / self.max_lane_shift_norm, 0.0, 1.0),
                lane_m["score"],
                np.clip(repeated + oscillating, 0.0, 1.0),
                0.0,
                0.0,
                0.0,
                0.0,
            ], dtype=np.float32)

            # Mask requires lane-feasible successor and destination reachability.
            lane_ready = lane_m["feasible"] and self._is_lane_legal_successor(lane_id, next_edge)
            if lane_ready and self._has_downstream_path(next_edge, vehicle.destination):
                mask[idx] = 1.0

        return np.concatenate([base, cand_vec.reshape(-1)], axis=0).reshape(1, -1), mask

    def make_decisions(self, vehicles, connection_info):
        local_targets = {}
        now = int(traci.simulation.getTime())
        global_stats = self._global_stats()

        for vehicle in vehicles:
            vehicle_id = vehicle.vehicle_id
            edge = vehicle.current_edge
            if edge == vehicle.destination:
                continue

            self._recent_edges[vehicle_id].append(edge)
            candidates = self._enumerate_candidates(vehicle_id, edge, vehicle.destination, now)
            if not candidates:
                self._metrics["no_candidates"] += 1
                continue

            state, mask = self._build_state(vehicle_id, vehicle, now, edge, candidates, global_stats)
            valid_idx = np.flatnonzero(mask > 0)
            if len(valid_idx) == 0:
                self._metrics["all_masked"] += 1
                continue

            q_values = self.model.predict(state, verbose=0)[0]
            masked = np.full_like(q_values, -1e9)
            for idx in valid_idx:
                masked[idx] = q_values[idx]
            chosen_idx = int(np.argmax(masked))
            if chosen_idx >= len(candidates):
                chosen_idx = int(valid_idx[0])

            direction, next_edge, lane_metrics = candidates[chosen_idx]
            self._recent_pairs[vehicle_id].append((edge, next_edge))

            if lane_metrics["min_lane_shifts"] > 0 and lane_metrics["target_lanes"]:
                try:
                    curr_idx = traci.vehicle.getLaneIndex(vehicle_id)
                    target_lane = min(lane_metrics["target_lanes"], key=lambda i: abs(i - curr_idx))
                    traci.vehicle.changeLane(vehicle_id, int(target_lane), 25)
                except Exception:
                    self._metrics["lane_change_request_fail"] += 1

            local_targets[vehicle_id] = self.compute_local_target([direction], vehicle)
            self._metrics["decisions"] += 1

        return local_targets
