from collections import defaultdict, deque
from dataclasses import dataclass
import math
import os

import numpy as np
import sumolib
import traci
import traci.constants as tc
from keras.models import load_model
from xml.dom.minidom import parse

from controller.RouteController import RouteController


def parse_sumocfg(sumocfg_path):
    dom = parse(sumocfg_path)
    net_file = dom.getElementsByTagName('net-file')[0].attributes['value'].nodeValue
    return os.path.join(os.path.dirname(sumocfg_path), net_file)


net_path = parse_sumocfg("./configurations/myconfig.sumocfg")


@dataclass
class PendingLaneAlignment:
    edge: str
    chosen_next_edge: str
    lane_index: int
    distance_to_end: float


class QLearningPolicy(RouteController):
    """Inference policy aligned with training-time next-edge commitment semantics."""

    def __init__(self, vehicles, connection_info, model_file, net_xml_file=net_path):
        super().__init__(connection_info)
        self.model = load_model(model_file)
        self.vehicles = vehicles
        self.net = sumolib.net.readNet(net_xml_file)
        self.candidate_slots = 6
        self.base_feature_size = 22
        self.candidate_feature_size = 13
        self.state_size = self.base_feature_size + self.candidate_slots * self.candidate_feature_size
        self.speed_norm = 20.0
        self.dist_norm = 300.0
        self.time_norm = 1200.0
        self.max_lane_shift_norm = 4.0
        self.loop_window = 10
        self.decision_min_hold_steps = 3
        self.reconsider_min_deficit_gain = 2.5
        self.reconsider_min_pressure_gain = 0.08

        self._distance_cache = {}
        self._eta_cache = {}
        self._downstream_path_cache = {}
        self._legal_successors_cache = {}
        self._lane_length_cache = {}
        self._lane_links_cache = {}
        self._lane_allowed_cache = {}
        self._edge_lane_count_cache = {
            edge_id: max(len(lane_ids), 1)
            for edge_id, lane_ids in self.connection_info.edge_lane_ids.items()
        }
        self._pending_lane_alignment = {}
        self._route_blacklist = {}
        self._recent_edges = defaultdict(lambda: deque(maxlen=self.loop_window))
        self._last_commit_step = {}
        self._last_choice = {}
        self.apply_direct_routes = True

    def _edge_from_lane_id(self, lane_id):
        if lane_id is None or "_" not in lane_id:
            return None
        return lane_id.rsplit("_", 1)[0]

    def _lane_length(self, lane_id):
        if lane_id in self._lane_length_cache:
            return self._lane_length_cache[lane_id]
        try:
            length = float(traci.lane.getLength(lane_id))
        except Exception:
            length = 0.0
        self._lane_length_cache[lane_id] = length
        return length

    def _lane_links(self, lane_id):
        if lane_id in self._lane_links_cache:
            return self._lane_links_cache[lane_id]
        try:
            links = traci.lane.getLinks(lane_id)
        except Exception:
            links = []
        self._lane_links_cache[lane_id] = links
        return links

    def _lane_allowed(self, lane_id):
        if lane_id in self._lane_allowed_cache:
            return self._lane_allowed_cache[lane_id]
        try:
            allowed = traci.lane.getAllowed(lane_id)
        except Exception:
            allowed = ()
        self._lane_allowed_cache[lane_id] = allowed
        return allowed

    def _distance_to_destination(self, edge_id, destination):
        key = (edge_id, destination)
        if key in self._distance_cache:
            return self._distance_cache[key]
        try:
            e0 = self.net.getEdge(edge_id)
            e1 = self.net.getEdge(destination)
            path, cost = self.net.getShortestPath(e0, e1)
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
            e0 = self.net.getEdge(edge_id)
            e1 = self.net.getEdge(destination)
            path, _ = self.net.getShortestPath(e0, e1)
            if path is None:
                eta = math.inf
            else:
                free_flow_eta = 0.0
                for e in path:
                    edge_speed = max(float(e.getSpeed()), 5.0)
                    free_flow_eta += float(e.getLength()) / edge_speed
                eta = free_flow_eta + max(len(path) - 1, 0) * 2.0 + 6.0 + 0.10 * free_flow_eta
        except Exception:
            eta = math.inf
        self._eta_cache[key] = eta
        return eta

    def _vehicle_legal_successors(self, vehicle_id, edge_id):
        legal = set()
        lane_ids = self.connection_info.edge_lane_ids.get(edge_id, [])
        try:
            vclass = traci.vehicle.getVehicleClass(vehicle_id)
        except Exception:
            vclass = None
        key = (edge_id, vclass)
        cached = self._legal_successors_cache.get(key)
        if cached is not None:
            return set(cached)
        for lane_id in lane_ids:
            for link in self._lane_links(lane_id):
                if not link:
                    continue
                next_lane = link[0]
                next_edge = self._edge_from_lane_id(next_lane)
                if not next_edge:
                    continue
                if vclass is not None:
                    allowed = self._lane_allowed(next_lane)
                    if allowed and (vclass not in allowed):
                        continue
                legal.add(next_edge)
        self._legal_successors_cache[key] = tuple(sorted(legal))
        return legal

    def _is_valid_immediate_successor(self, vehicle_id, current_edge, chosen_next):
        topo = set(self.connection_info.outgoing_edges_dict.get(current_edge, {}).values())
        return (chosen_next in topo) and (chosen_next in self._vehicle_legal_successors(vehicle_id, current_edge))

    def _has_downstream_path(self, chosen_next, destination):
        key = (chosen_next, destination)
        if key in self._downstream_path_cache:
            return self._downstream_path_cache[key]
        ok = math.isfinite(self._distance_to_destination(chosen_next, destination))
        self._downstream_path_cache[key] = ok
        return ok

    def _compute_lane_metrics(self, vehicle_id, edge_id, next_edge):
        lane_ids = self.connection_info.edge_lane_ids.get(edge_id, [])
        curr_lane = traci.vehicle.getLaneIndex(vehicle_id)
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        dist_to_end = max(self._lane_length(lane_id) - traci.vehicle.getLanePosition(vehicle_id), 0.0)
        speed = max(traci.vehicle.getSpeed(vehicle_id), 1.0)

        target_lanes = []
        for idx, ln in enumerate(lane_ids):
            for _d, out_edge in self.connection_info.lane_outgoing_edges_dict.get(ln, {}).items():
                if out_edge == next_edge:
                    target_lanes.append(idx)
                    break
        if not target_lanes:
            return {"target_lanes": [], "min_lane_shifts": 99, "feasible": False, "score": 0.0}

        min_shift = min(abs(curr_lane - t) for t in target_lanes)
        est_shift_distance = 18.0 * min_shift
        comfort_budget = max(35.0, speed * 2.3)
        feasible = dist_to_end >= est_shift_distance + 8.0
        score = float(np.clip((dist_to_end - est_shift_distance) / comfort_budget, 0.0, 1.0))
        return {"target_lanes": target_lanes, "min_lane_shifts": min_shift, "feasible": feasible, "score": score}

    def _enumerate_candidates(self, vehicle_id, edge_id, destination, step):
        outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {})
        cands = []
        for _dir, nxt in outgoing.items():
            if nxt in cands:
                continue
            if self._route_blacklist.get((vehicle_id, edge_id, nxt), -1) >= step:
                continue
            if not self._is_valid_immediate_successor(vehicle_id, edge_id, nxt):
                continue
            if not self._has_downstream_path(nxt, destination):
                continue
            cands.append(nxt)
        return cands[: self.candidate_slots]

    def _global_density(self):
        vals = []
        for e in self.connection_info.edge_list:
            try:
                count = traci.edge.getLastStepVehicleNumber(e)
            except Exception:
                count = 0
            vals.append(count / max(self.connection_info.edge_length_dict.get(e, 1.0), 1.0))
        if not vals:
            return 0.0, 0.0
        return float(np.mean(vals)), float(np.std(vals))

    def _build_state(self, vehicle_id, vehicle, step, current_edge, candidates):
        mean_density, std_density = self._global_density()
        lane_idx = traci.vehicle.getLaneIndex(vehicle_id)
        n_lanes = self._edge_lane_count_cache.get(current_edge, max(traci.edge.getLaneNumber(current_edge), 1))
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        lane_pos = traci.vehicle.getLanePosition(vehicle_id)
        dist_to_end = max(self._lane_length(lane_id) - lane_pos, 0.0)
        speed = traci.vehicle.getSpeed(vehicle_id)
        time_left = max(float(vehicle.deadline) - float(step), 0.0)
        elapsed = max(float(step) - float(vehicle.start_time), 0.0)
        window = max(float(vehicle.deadline) - float(vehicle.start_time), 1.0)
        eta_curr = self._estimate_eta(current_edge, vehicle.destination)
        slack = (time_left - eta_curr) if math.isfinite(eta_curr) else -self.time_norm

        curr_density = traci.edge.getLastStepVehicleNumber(current_edge) / max(self.connection_info.edge_length_dict.get(current_edge, 1.0), 1.0)
        curr_mean_speed = traci.edge.getLastStepMeanSpeed(current_edge)
        sp_dist = self._distance_to_destination(current_edge, vehicle.destination)
        topo_hops = sp_dist / 120.0 if math.isfinite(sp_dist) else 10.0

        base = np.array([
            lane_idx / max(n_lanes - 1, 1),
            min(n_lanes, 6) / 6.0,
            min(dist_to_end, self.dist_norm) / self.dist_norm,
            min(speed, self.speed_norm) / self.speed_norm,
            min(time_left, self.time_norm) / self.time_norm,
            min(elapsed / window, 2.0) / 2.0,
            np.clip(1.0 - (time_left / window), 0.0, 1.0),
            np.clip(slack / self.time_norm, -1.0, 1.0),
            np.clip(curr_density, 0.0, 2.0) / 2.0,
            np.clip(curr_mean_speed / self.speed_norm, 0.0, 1.5) / 1.5,
            np.clip(traci.edge.getLastStepHaltingNumber(current_edge) / max(self.connection_info.edge_length_dict.get(current_edge, 5.0), 5.0), 0.0, 1.0),
            min(len(candidates), self.candidate_slots) / float(self.candidate_slots),
            1.0 if current_edge in self._recent_edges[vehicle_id] else 0.0,
            np.clip(eta_curr / self.time_norm if math.isfinite(eta_curr) else 1.0, 0.0, 2.0) / 2.0,
            np.clip(sp_dist / 3000.0 if math.isfinite(sp_dist) else 1.0, 0.0, 1.0),
            np.clip(topo_hops / 20.0, 0.0, 1.0),
            np.clip(mean_density, 0.0, 2.0) / 2.0,
            np.clip(std_density, 0.0, 1.0),
            0.0,
            0.0,
            0.0,
            0.0,
        ], dtype=np.float32)

        cand_vec = np.zeros((self.candidate_slots, self.candidate_feature_size), dtype=np.float32)
        mask = np.zeros((self.candidate_slots,), dtype=np.float32)
        for i, nxt in enumerate(candidates):
            lane_m = self._compute_lane_metrics(vehicle_id, current_edge, nxt)
            density = traci.edge.getLastStepVehicleNumber(nxt) / max(self.connection_info.edge_length_dict.get(nxt, 1.0), 1.0)
            mean_speed = traci.edge.getLastStepMeanSpeed(nxt)
            eta = self._estimate_eta(nxt, vehicle.destination)
            deficit = max((eta - time_left), 0.0) if math.isfinite(eta) else self.time_norm
            repeated = 1.0 if nxt in self._recent_edges[vehicle_id] else 0.0
            dead_end = 0.0 if self._has_downstream_path(nxt, vehicle.destination) else 1.0
            cand_vec[i] = np.array([
                1.0,
                np.clip(density, 0.0, 2.0) / 2.0,
                np.clip(mean_speed / self.speed_norm, 0.0, 1.5) / 1.5,
                np.clip(eta / self.time_norm if math.isfinite(eta) else 1.0, 0.0, 2.0) / 2.0,
                np.clip(deficit / self.time_norm, 0.0, 1.0),
                np.clip(max(density - mean_density, 0.0), 0.0, 1.0),
                np.clip(lane_m["min_lane_shifts"] / self.max_lane_shift_norm, 0.0, 1.0),
                lane_m["score"],
                repeated,
                dead_end,
                0.0,
                0.0,
                0.0,
            ], dtype=np.float32)
            if lane_m["feasible"] and dead_end < 1.0:
                mask[i] = 1.0
        return np.concatenate([base, cand_vec.reshape(-1)], axis=0).reshape(1, -1), mask

    def _select_action(self, state, mask):
        valid = np.flatnonzero(mask > 0)
        if len(valid) == 0:
            return None
        q = self.model(state, training=False).numpy()[0]
        masked = np.full_like(q, -1e9)
        masked[valid] = q[valid]
        return int(np.argmax(masked))

    def _current_lane_successors(self, vehicle_id):
        successors = set()
        for link in self._lane_links(traci.vehicle.getLaneID(vehicle_id)):
            if not link:
                continue
            nxt = self._edge_from_lane_id(link[0])
            if nxt:
                successors.add(nxt)
        return successors

    def _find_route_edges(self, from_edge, to_edge, vehicle_id):
        try:
            route = traci.simulation.findRoute(from_edge, to_edge, vType=traci.vehicle.getTypeID(vehicle_id))
        except Exception:
            return None
        route_edges = list(getattr(route, "edges", []) or [])
        return route_edges or None

    def _apply_next_edge_route(self, vehicle_id, current_edge, chosen_next, destination):
        if not self._is_valid_immediate_successor(vehicle_id, current_edge, chosen_next):
            return False, "invalid_first_hop"
        if chosen_next not in self._current_lane_successors(vehicle_id):
            return False, "lane_not_ready"
        first = self._find_route_edges(current_edge, chosen_next, vehicle_id)
        if not first:
            return False, "invalid_first_hop"
        down = self._find_route_edges(chosen_next, destination, vehicle_id)
        if not down:
            return False, "downstream_path_missing"
        try:
            traci.vehicle.setRoute(vehicle_id, first + down[1:])
            return True, "ok"
        except Exception:
            return False, "route_set_exception"

    def _should_decide(self, vehicle_id, edge_id, candidates, step):
        if len(candidates) <= 1:
            return False
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        dist_to_end = max(self._lane_length(lane_id) - traci.vehicle.getLanePosition(vehicle_id), 0.0)
        speed = max(traci.vehicle.getSpeed(vehicle_id), 3.0)
        threshold = min(0.85 * self.connection_info.edge_length_dict.get(edge_id, 100.0), max(45.0, 2.2 * speed + 12.0))
        if dist_to_end > threshold:
            return False
        last_step = self._last_commit_step.get(vehicle_id, -10**9)
        if step - last_step < self.decision_min_hold_steps:
            return False
        return True

    def make_decisions(self, vehicles, connection_info):
        decisions = {}
        step = int(traci.simulation.getTime())

        for vehicle in vehicles:
            vid = vehicle.vehicle_id
            edge = vehicle.current_edge
            if not edge or edge == vehicle.destination:
                continue
            self._recent_edges[vid].append(edge)
            candidates = self._enumerate_candidates(vid, edge, vehicle.destination, step)
            if not self._should_decide(vid, edge, candidates, step):
                continue
            state, mask = self._build_state(vid, vehicle, step, edge, candidates)
            action = self._select_action(state, mask)
            if action is None or action >= len(candidates):
                continue
            chosen_next = candidates[action]

            lane_m = self._compute_lane_metrics(vid, edge, chosen_next)
            if lane_m["target_lanes"] and lane_m["min_lane_shifts"] > 0:
                target_lane = min(lane_m["target_lanes"], key=lambda idx: abs(idx - traci.vehicle.getLaneIndex(vid)))
                try:
                    traci.vehicle.changeLane(vid, int(target_lane), 20)
                except Exception:
                    pass

            blocked = self._pending_lane_alignment.get(vid)
            if blocked and blocked.edge == edge and blocked.chosen_next_edge == chosen_next:
                curr_lane_idx = traci.vehicle.getLaneIndex(vid)
                lane_id = traci.vehicle.getLaneID(vid)
                dist_to_end = max(self._lane_length(lane_id) - traci.vehicle.getLanePosition(vid), 0.0)
                if (curr_lane_idx == blocked.lane_index) and (dist_to_end >= blocked.distance_to_end - 1.0):
                    continue

            applied, reason = self._apply_next_edge_route(vid, edge, chosen_next, vehicle.destination)
            if applied:
                self._pending_lane_alignment.pop(vid, None)
                self._last_commit_step[vid] = step
                self._last_choice[vid] = chosen_next
                decisions[vid] = chosen_next
                continue

            if reason == "lane_not_ready":
                lane_id = traci.vehicle.getLaneID(vid)
                self._pending_lane_alignment[vid] = PendingLaneAlignment(
                    edge=edge,
                    chosen_next_edge=chosen_next,
                    lane_index=traci.vehicle.getLaneIndex(vid),
                    distance_to_end=max(self._lane_length(lane_id) - traci.vehicle.getLanePosition(vid), 0.0),
                )
            else:
                self._route_blacklist[(vid, edge, chosen_next)] = step + 25
        return decisions
