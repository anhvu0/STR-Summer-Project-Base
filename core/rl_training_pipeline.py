import math
import os
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
from keras import backend as K
from keras.layers import Dense, Input, Lambda
from keras.losses import Huber
from keras.models import Model, clone_model
from keras.optimizers import Adam
from xml.dom.minidom import parse

from controller.RouteController import RouteController
from core.Util import ConnectionInfo
from core.target_vehicles_generation_protocols import target_vehicles_generator

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")

from sumolib import checkBinary
import traci
import sumolib

MAX_SIMULATION_STEPS = 3000


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, eps=1e-5):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.buffer = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, transition, priority=None):
        max_p = float(np.max(self.priorities[: len(self.buffer)]) if self.buffer else 1.0)
        p = max(priority if priority is not None else max_p, self.eps)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        size = len(self.buffer)
        if size == 0:
            return [], np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        probs = self.priorities[:size] ** self.alpha
        probs = probs / np.sum(probs)
        idx = np.random.choice(size, batch_size, p=probs, replace=(size < batch_size))
        samples = [self.buffer[i] for i in idx]
        weights = (size * probs[idx]) ** (-beta)
        weights = weights / np.max(weights)
        return samples, idx, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        for i, td in zip(indices, td_errors):
            self.priorities[int(i)] = max(abs(float(td)) + self.eps, self.eps)


class TrainingRouteHelper(RouteController):
    def __init__(self, connection_info):
        super().__init__(connection_info)

    def make_decisions(self, vehicles, connection_info):
        return {}


class DQNTrainer:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=5e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        replay_capacity=10000,
        batch_size=64,
        replay_warmup=1000,
        target_update_every=1,
        target_soft_tau=0.01,
        grad_clip_norm=10.0,
        n_step=3,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_anneal_steps=100000,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.batch_size = int(batch_size)
        self.replay_warmup = max(int(replay_warmup), self.batch_size)
        self.target_update_every = max(int(target_update_every), 1)
        self.target_soft_tau = float(np.clip(target_soft_tau, 0.0, 1.0))
        self.grad_clip_norm = float(grad_clip_norm)
        self.n_step = max(int(n_step), 1)
        self.memory = PrioritizedReplayBuffer(replay_capacity, alpha=per_alpha)
        self.per_beta_start = float(per_beta_start)
        self.per_beta_anneal_steps = max(int(per_beta_anneal_steps), 1)
        self.train_steps = 0
        self.last_td_error_stats = {"mean": 0.0, "max": 0.0, "min": 0.0}

        self.model = self.build_model(learning_rate)
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self, learning_rate):
        inp = Input(shape=(self.state_size,))
        x = Dense(128, activation='relu')(inp)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        value = Dense(1, activation='linear')(x)
        advantage = Dense(self.action_size, activation='linear')(x)
        centered_adv = Lambda(lambda a: a - K.mean(a, axis=1, keepdims=True))(advantage)
        q_vals = Lambda(lambda va: va[0] + va[1])([value, centered_adv])
        model = Model(inp, q_vals)
        model.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=self.grad_clip_norm), loss=Huber())
        return model

    def _beta(self):
        return min(1.0, self.per_beta_start + (1.0 - self.per_beta_start) * (self.train_steps / self.per_beta_anneal_steps))

    def select_action(self, state_vec, valid_mask):
        valid_idx = np.flatnonzero(valid_mask > 0)
        if len(valid_idx) == 0:
            return None
        if np.random.rand() <= self.epsilon:
            return int(random.choice(valid_idx))
        q_values = self.model(state_vec, training=False).numpy()[0]
        masked = np.full_like(q_values, -1e9)
        masked[valid_idx] = q_values[valid_idx]
        return int(np.argmax(masked))

    def remember(self, transition, priority=None):
        self.memory.add(transition, priority=priority)

    def replay(self):
        if len(self.memory) < self.replay_warmup:
            return

        samples, idxs, is_weights = self.memory.sample(self.batch_size, beta=self._beta())
        states = np.vstack([t["state"] for t in samples])
        actions = np.array([t["action"] for t in samples], dtype=np.int32)
        rewards = np.array([t["reward"] for t in samples], dtype=np.float32)
        next_states = np.vstack([t["next_state"] for t in samples])
        dones = np.array([t["done"] for t in samples], dtype=np.float32)
        next_masks = np.vstack([t["next_mask"] for t in samples])
        horizons = np.array([t.get("horizon", 1) for t in samples], dtype=np.float32)

        q_pred = self.model(states, training=False).numpy()
        q_next_online = self.model(next_states, training=False).numpy()
        q_next_target = self.target_model(next_states, training=False).numpy()

        bootstrap = np.zeros(self.batch_size, dtype=np.float32)
        for i in range(self.batch_size):
            if dones[i] >= 1.0:
                continue
            valid = np.flatnonzero(next_masks[i] > 0)
            if len(valid) == 0:
                continue
            masked_online = np.full(self.action_size, -1e9, dtype=np.float32)
            masked_online[valid] = q_next_online[i, valid]
            best_a = int(np.argmax(masked_online))
            bootstrap[i] = q_next_target[i, best_a]

        target = q_pred.copy()
        y = rewards + (1.0 - dones) * (self.gamma ** horizons) * bootstrap
        td_errors = y - q_pred[np.arange(self.batch_size), actions]
        target[np.arange(self.batch_size), actions] = q_pred[np.arange(self.batch_size), actions] + td_errors

        self.model.train_on_batch(states, target, sample_weight=is_weights)
        self.memory.update_priorities(idxs, td_errors)
        self.last_td_error_stats = {
            "mean": float(np.mean(np.abs(td_errors))),
            "max": float(np.max(np.abs(td_errors))),
            "min": float(np.min(np.abs(td_errors))),
        }

        self.train_steps += 1
        if self.train_steps % self.target_update_every == 0:
            online = self.model.get_weights()
            target_w = self.target_model.get_weights()
            tau = self.target_soft_tau
            mixed = [tau * o + (1.0 - tau) * t for o, t in zip(online, target_w)]
            self.target_model.set_weights(mixed)


@dataclass
class Commitment:
    state: np.ndarray
    action: int
    edge: str
    chosen_next_edge: str
    chosen_density: float
    prev_deficit: float
    step: int
    candidate_repeated: float


@dataclass
class RouteApplyResult:
    applied: bool
    reason: str = "ok"
    route_edges: list | None = None


class RLTrainingPipeline:
    def __init__(
        self,
        sumocfg_path,
        model_output_path,
        episodes=10,
        spawn_interval=4.0,
        seed_with_episode=True,
        candidate_slots=6,
        epsilon_decay=0.995,
        epsilon_min=0.10,
        gamma=0.99,
        replay_capacity=10000,
        batch_size=64,
        replay_warmup=1000,
        train_every=5,
        grad_steps=1,
        rolling_window=100,
        target_pattern=2,
    ):
        self.sumocfg_path = sumocfg_path
        self.model_output_path = model_output_path
        self.episodes = episodes
        self.spawn_interval = spawn_interval
        self.seed_with_episode = seed_with_episode
        self.candidate_slots = int(candidate_slots)
        self.train_every = int(train_every)
        self.grad_steps = int(grad_steps)
        self.rolling_window = int(rolling_window)
        self.target_pattern = target_pattern

        self.w_deficit_delta = 6.0
        self.w_critical_worse = 12.0
        self.w_externality = 2.5
        self.w_lane_failure = 20.0
        self.w_loop = 5.0
        self.execution_success_bonus = 8.0
        self.invalid_first_hop_penalty = -10.0
        self.downstream_path_missing_penalty = -8.0
        self.route_set_exception_penalty = -12.0
        self.teleport_penalty = -180.0
        self.arrival_on_time_reward = 220.0
        self.arrival_late_reward = 80.0
        self.wrong_target_penalty = -200.0
        self.deadline_miss_terminal_scale = 0.8

        self.loop_window = 10
        self.speed_norm = 20.0
        self.dist_norm = 300.0
        self.time_norm = 1200.0
        self.max_lane_shift_norm = 4.0

        self.sumocfg_dir = os.path.dirname(sumocfg_path)
        self.net_file, self.route_file = self.parse_sumocfg(sumocfg_path)
        self.net = sumolib.net.readNet(os.path.join(self.sumocfg_dir, self.net_file))
        self.connection_info = ConnectionInfo(os.path.join(self.sumocfg_dir, self.net_file))
        self.route_helper = TrainingRouteHelper(self.connection_info)
        self._distance_cache = {}
        self._eta_cache = {}
        self._route_blacklist = {}
        self.route_retry_cooldown_steps = 30
        self._curriculum = [
            {"until": 0.33, "target": 12, "random": 16, "slack": (1.35, 1.55)},
            {"until": 0.66, "target": 16, "random": 24, "slack": (1.20, 1.40)},
            {"until": 1.00, "target": 20, "random": 30, "slack": (1.10, 1.30)},
        ]

        self.base_feature_size = 20
        self.candidate_feature_size = 10
        self.state_size = self.base_feature_size + self.candidate_slots * self.candidate_feature_size
        self.action_size = self.candidate_slots

        self.trainer = DQNTrainer(
            self.state_size,
            self.action_size,
            gamma=gamma,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            replay_warmup=replay_warmup,
            n_step=3,
            target_soft_tau=0.01,
            target_update_every=1,
        )

    def parse_sumocfg(self, sumocfg_path):
        dom = parse(sumocfg_path)
        net_file = dom.getElementsByTagName('net-file')[0].attributes['value'].nodeValue
        route_file = dom.getElementsByTagName('route-files')[0].attributes['value'].nodeValue
        return net_file, route_file

    def get_curriculum_config(self, episode_idx):
        frac = (episode_idx + 1) / float(max(self.episodes, 1))
        for phase in self._curriculum:
            if frac <= phase["until"]:
                return phase
        return self._curriculum[-1]

    def get_distance_to_destination(self, edge_id, destination_edge):
        key = (edge_id, destination_edge)
        if key in self._distance_cache:
            return self._distance_cache[key]
        try:
            e0 = self.net.getEdge(edge_id)
            e1 = self.net.getEdge(destination_edge)
            path, cost = self.net.getShortestPath(e0, e1)
            dist = float(cost) if path is not None else math.inf
        except Exception:
            dist = math.inf
        self._distance_cache[key] = dist
        return dist

    def estimate_eta(self, edge_id, destination_edge):
        key = (edge_id, destination_edge)
        if key in self._eta_cache:
            return self._eta_cache[key]
        try:
            e0 = self.net.getEdge(edge_id)
            e1 = self.net.getEdge(destination_edge)
            path, _ = self.net.getShortestPath(e0, e1)
            if path is None:
                eta = math.inf
            else:
                free_flow_eta = 0.0
                for e in path:
                    edge_speed = max(float(e.getSpeed()), 5.0)
                    free_flow_eta += float(e.getLength()) / edge_speed
                junction_delay = max(len(path) - 1, 0) * 2.0
                congestion_allowance = 6.0 + 0.10 * free_flow_eta
                eta = free_flow_eta + junction_delay + congestion_allowance
        except Exception:
            eta = math.inf
        self._eta_cache[key] = eta
        return eta

    def get_teleport_ids(self):
        teleported = set()
        for fn in [
            lambda: traci.simulation.getStartingTeleportIDList(),
            lambda: traci.simulation.getEndingTeleportIDList(),
            lambda: traci.vehicle.getTeleportingList(),
        ]:
            try:
                teleported.update(fn())
            except Exception:
                pass
        return teleported

    def update_edge_vehicle_counts(self, step, every=1):
        if hasattr(self, "_last_density_step") and (step - self._last_density_step) < every:
            return
        counts = self.connection_info.edge_vehicle_count
        lengths = self.connection_info.edge_length_dict
        for edge in self.connection_info.edge_list:
            counts[edge] = traci.edge.getLastStepVehicleNumber(edge)
        self._density_vec = np.array([counts[e] / max(lengths[e], 1.0) for e in self.connection_info.edge_list], dtype=np.float32)
        self._global_density_mean = float(np.mean(self._density_vec)) if len(self._density_vec) else 0.0
        self._global_density_std = float(np.std(self._density_vec)) if len(self._density_vec) else 0.0
        self._last_density_step = step

    def _blacklist_key(self, vehicle_id, current_edge, next_edge):
        return (str(vehicle_id), str(current_edge), str(next_edge))

    def _edge_from_lane_id(self, lane_id):
        if lane_id is None:
            return None
        if "_" not in lane_id:
            return None
        return lane_id.rsplit("_", 1)[0]

    def _get_vehicle_legal_successors(self, vehicle_id, edge_id):
        legal = set()
        lane_ids = self.connection_info.edge_lane_ids.get(edge_id, [])
        try:
            vclass = traci.vehicle.getVehicleClass(vehicle_id)
        except Exception:
            vclass = None
        for lane_id in lane_ids:
            try:
                links = traci.lane.getLinks(lane_id)
            except Exception:
                links = []
            for link in links:
                if not link:
                    continue
                next_lane = link[0]
                next_edge = self._edge_from_lane_id(next_lane)
                if not next_edge:
                    continue
                if vclass is not None:
                    try:
                        allowed = traci.lane.getAllowed(next_lane)
                        if allowed and (vclass not in allowed):
                            continue
                    except Exception:
                        pass
                legal.add(next_edge)
        return legal

    def _is_valid_immediate_successor(self, vehicle_id, current_edge, chosen_next_edge):
        topological = set(self.connection_info.outgoing_edges_dict.get(current_edge, {}).values())
        if chosen_next_edge not in topological:
            return False
        legal = self._get_vehicle_legal_successors(vehicle_id, current_edge)
        if legal and chosen_next_edge not in legal:
            return False
        return True

    def _has_downstream_path(self, chosen_next_edge, destination):
        try:
            from_edge = self.net.getEdge(chosen_next_edge)
            to_edge = self.net.getEdge(destination)
            path, _ = self.net.getShortestPath(from_edge, to_edge)
            return path is not None
        except Exception:
            return False

    def enumerate_candidate_next_edges(self, vehicle_id, edge_id, destination, step, diag=None):
        outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {})
        seen = set()
        candidates = []
        for _d, nxt in outgoing.items():
            if nxt not in seen:
                seen.add(nxt)
                bl_key = self._blacklist_key(vehicle_id, edge_id, nxt)
                blocked_until = self._route_blacklist.get(bl_key, -1)
                if blocked_until >= step:
                    if diag is not None:
                        diag["blacklist_hits"] += 1
                    continue
                if not self._is_valid_immediate_successor(vehicle_id, edge_id, nxt):
                    if diag is not None:
                        diag["invalid_first_hop_suppressions"] += 1
                    continue
                if not self._has_downstream_path(nxt, destination):
                    if diag is not None:
                        diag["downstream_path_failures"] += 1
                    continue
                candidates.append(nxt)
        return candidates[: self.candidate_slots]

    def compute_candidate_lane_metrics(self, vehicle_id, edge_id, next_edge):
        lane_ids = self.connection_info.edge_lane_ids.get(edge_id, [])
        curr_lane = traci.vehicle.getLaneIndex(vehicle_id)
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        lane_len = traci.lane.getLength(lane_id)
        lane_pos = traci.vehicle.getLanePosition(vehicle_id)
        dist_to_end = max(lane_len - lane_pos, 0.0)
        speed = max(traci.vehicle.getSpeed(vehicle_id), 1.0)

        target_lanes = []
        for idx, ln in enumerate(lane_ids):
            for _direction, out_edge in self.connection_info.lane_outgoing_edges_dict.get(ln, {}).items():
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
        return {
            "target_lanes": target_lanes,
            "min_lane_shifts": min_shift,
            "feasible": feasible,
            "score": score,
        }

    def _queue_proxy(self, edge_id):
        edge_len = max(self.connection_info.edge_length_dict.get(edge_id, 5.0), 5.0)
        halted = traci.edge.getLastStepHaltingNumber(edge_id)
        return float(halted) / edge_len

    def build_state(self, vehicle_id, vehicle, step, current_edge, candidates, recent_edges, global_stats):
        lane_idx = traci.vehicle.getLaneIndex(vehicle_id)
        n_lanes = max(traci.edge.getLaneNumber(current_edge), 1)
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        dist_to_end = max(traci.lane.getLength(lane_id) - traci.vehicle.getLanePosition(vehicle_id), 0.0)
        speed = traci.vehicle.getSpeed(vehicle_id)
        time_left = max(float(vehicle.deadline) - float(step), 0.0)
        elapsed = max(float(step) - float(vehicle.start_time), 0.0)
        window = max(float(vehicle.deadline) - float(vehicle.start_time), 1.0)
        eta_curr = self.estimate_eta(current_edge, vehicle.destination)
        slack = (time_left - eta_curr) if math.isfinite(eta_curr) else -self.time_norm
        urgency = float(np.clip(1.0 - (time_left / window), 0.0, 1.0))

        curr_density = self.connection_info.edge_vehicle_count.get(current_edge, 0) / max(self.connection_info.edge_length_dict.get(current_edge, 1.0), 1.0)
        curr_mean_speed = traci.edge.getLastStepMeanSpeed(current_edge)
        sp_dist = self.get_distance_to_destination(current_edge, vehicle.destination)
        topo_hops = sp_dist / 120.0 if math.isfinite(sp_dist) else 10.0

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
            np.clip(curr_mean_speed / self.speed_norm, 0.0, 1.5) / 1.5,
            np.clip(self._queue_proxy(current_edge), 0.0, 1.0),
            min(len(candidates), self.candidate_slots) / float(self.candidate_slots),
            1.0 if current_edge in recent_edges else 0.0,
            np.clip(eta_curr / self.time_norm if math.isfinite(eta_curr) else 1.0, 0.0, 2.0) / 2.0,
            np.clip(sp_dist / 3000.0 if math.isfinite(sp_dist) else 1.0, 0.0, 1.0),
            np.clip(topo_hops / 20.0, 0.0, 1.0),
            np.clip(global_stats["mean_density"], 0.0, 2.0) / 2.0,
            np.clip(global_stats["std_density"], 0.0, 1.0),
            np.clip(global_stats["frac_behind"], 0.0, 1.0),
            np.clip(global_stats["frac_failed"], 0.0, 1.0),
        ], dtype=np.float32)

        cand_vec = np.zeros((self.candidate_slots, self.candidate_feature_size), dtype=np.float32)
        mask = np.zeros((self.candidate_slots,), dtype=np.float32)
        for i, next_edge in enumerate(candidates[: self.candidate_slots]):
            lane_m = self.compute_candidate_lane_metrics(vehicle_id, current_edge, next_edge)
            density = self.connection_info.edge_vehicle_count.get(next_edge, 0) / max(self.connection_info.edge_length_dict.get(next_edge, 1.0), 1.0)
            mean_speed = traci.edge.getLastStepMeanSpeed(next_edge)
            eta = self.estimate_eta(next_edge, vehicle.destination)
            deficit = max((eta - time_left), 0.0) if math.isfinite(eta) else self.time_norm
            repeated = 1.0 if next_edge in recent_edges else 0.0
            dead_end = 0.0 if math.isfinite(self.get_distance_to_destination(next_edge, vehicle.destination)) else 1.0
            cand_vec[i] = np.array([
                1.0,
                np.clip(density, 0.0, 2.0) / 2.0,
                np.clip(mean_speed / self.speed_norm, 0.0, 1.5) / 1.5,
                np.clip(eta / self.time_norm if math.isfinite(eta) else 1.0, 0.0, 2.0) / 2.0,
                np.clip(deficit / self.time_norm, 0.0, 1.0),
                np.clip(max(density - global_stats["mean_density"], 0.0), 0.0, 1.0),
                np.clip(lane_m["min_lane_shifts"] / self.max_lane_shift_norm, 0.0, 1.0),
                lane_m["score"],
                repeated,
                dead_end,
            ], dtype=np.float32)
            # keep invalid/dead candidates masked out
            immediate_ok = self._is_valid_immediate_successor(vehicle_id, current_edge, next_edge)
            downstream_ok = dead_end < 1.0
            if lane_m["min_lane_shifts"] < 99 and lane_m["feasible"] and downstream_ok and immediate_ok:
                mask[i] = 1.0

        return np.concatenate([base, cand_vec.reshape(-1)], axis=0).reshape(1, -1), mask, cand_vec

    def _commitment_threshold(self, vehicle_id, edge_id, candidates):
        edge_len = float(self.connection_info.edge_length_dict.get(edge_id, 100.0))
        speed = max(traci.vehicle.getSpeed(vehicle_id), 3.0)
        lanes = max(traci.edge.getLaneNumber(edge_id), 1)
        lane_cost = 0.0
        for c in candidates:
            lane_cost = max(lane_cost, self.compute_candidate_lane_metrics(vehicle_id, edge_id, c)["min_lane_shifts"])
        t = min(0.85 * edge_len, max(45.0, 2.2 * speed + 12.0 * lane_cost + 4.0 * lanes))
        return float(t)

    def should_make_decision(self, vehicle_id, edge_id, candidates, commitment, latest_density, current_density, candidate_densities, committed_deficit=None, candidate_deficits=None):
        if len(candidates) <= 1:
            return False
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        dist_to_end = max(traci.lane.getLength(lane_id) - traci.vehicle.getLanePosition(vehicle_id), 0.0)
        in_zone = dist_to_end <= self._commitment_threshold(vehicle_id, edge_id, candidates)
        if not in_zone:
            return False
        if commitment is None:
            return True
        if commitment.edge != edge_id:
            return True
        if abs(current_density - latest_density) > 0.10:
            return True
        if candidate_densities:
            best_idx = int(np.argmin(candidate_densities))
            best_candidate_density = candidate_densities[best_idx]
            best_candidate_deficit = candidate_deficits[best_idx] if candidate_deficits and best_idx < len(candidate_deficits) else math.inf
            if (current_density - best_candidate_density) > 0.12 and (committed_deficit is None or best_candidate_deficit <= committed_deficit + 2.0):
                return True
        lane_m = self.compute_candidate_lane_metrics(vehicle_id, edge_id, commitment.chosen_next_edge)
        return not lane_m["feasible"]

    def apply_route_commitment(self, vehicle_id, current_edge, chosen_next_edge, destination):
        if not self._is_valid_immediate_successor(vehicle_id, current_edge, chosen_next_edge):
            return RouteApplyResult(applied=False, reason="invalid_first_hop")
        try:
            from_edge = self.net.getEdge(chosen_next_edge)
            to_edge = self.net.getEdge(destination)
            path, _ = self.net.getShortestPath(from_edge, to_edge)
            if path is None:
                return RouteApplyResult(applied=False, reason="downstream_path_missing")
            route_edges = [current_edge] + [e.getID() for e in path]
            traci.vehicle.setRoute(vehicle_id, route_edges)
            return RouteApplyResult(applied=True, reason="ok", route_edges=route_edges)
        except Exception:
            return RouteApplyResult(applied=False, reason="route_set_exception")

    def _global_stats(self, controlled_ids, vehicles, step, teleported, failed):
        behind = 0
        for vid in controlled_ids:
            if vid not in vehicles:
                continue
            v = vehicles[vid]
            edge = v.current_edge
            if not edge:
                continue
            tleft = max(v.deadline - step, 0.0)
            eta = self.estimate_eta(edge, v.destination)
            if math.isfinite(eta) and eta > tleft:
                behind += 1
        total = max(len(controlled_ids), 1)
        return {
            "mean_density": self._global_density_mean,
            "std_density": self._global_density_std,
            "frac_behind": behind / float(total),
            "frac_failed": (len(teleported) + len(failed)) / float(total),
        }

    def compute_reward(self, prev_deficit, curr_deficit, chosen_density, global_mean_density, lane_failed=False, repeated=0.0, arrived=False, on_time=False, wrong_target=False, teleported=False, deadline_missed=False, lateness=0.0, remain_dist=0.0):
        reward = 0.0
        reward += self.w_deficit_delta * (prev_deficit - curr_deficit)
        if prev_deficit > 0 and curr_deficit > prev_deficit:
            reward -= self.w_critical_worse * (curr_deficit - prev_deficit)
        reward -= self.w_externality * max(chosen_density - global_mean_density, 0.0)
        if lane_failed:
            reward -= self.w_lane_failure
        reward -= self.w_loop * repeated
        if teleported:
            return reward + self.teleport_penalty, True
        if wrong_target:
            return reward + self.wrong_target_penalty, True
        if arrived:
            if on_time:
                return reward + self.arrival_on_time_reward, True
            return reward + self.arrival_late_reward - min(lateness, 300.0) * 0.2, True
        if deadline_missed:
            return reward - (40.0 + self.deadline_miss_terminal_scale * (lateness + 0.05 * remain_dist)), True
        return reward, False

    def generate_episode_vehicles(self, episode_seed=None, curriculum_cfg=None):
        generator = target_vehicles_generator(os.path.join(self.sumocfg_dir, self.net_file))
        route_path = os.path.join(self.sumocfg_dir, self.route_file)
        c = curriculum_cfg or self._curriculum[-1]
        vlist = generator.generate_vehicles(
            num_target_vehicles=c["target"],
            num_random_vehicles=c["random"],
            pattern=self.target_pattern,
            target_xml_file=route_path,
            net_xml_file=os.path.join(self.sumocfg_dir, self.net_file),
            spawn_interval=self.spawn_interval,
            seed=episode_seed,
            deadline_slack_range=c["slack"],
            curriculum_phase=c,
        )
        if vlist is None:
            raise RuntimeError("Failed to generate vehicles")
        return {str(v.vehicle_id): v for v in vlist}

    def run(self):
        sumo_binary = checkBinary('sumo')
        rolling = defaultdict(lambda: deque(maxlen=self.rolling_window))
        rolling_baseline = deque(maxlen=30)
        metrics_history = []

        for episode in range(self.episodes):
            self._route_blacklist = {}
            if self.seed_with_episode:
                random.seed(episode)
                np.random.seed(episode)
            curriculum_cfg = self.get_curriculum_config(episode)
            vehicles = self.generate_episode_vehicles(episode_seed=(episode if self.seed_with_episode else None), curriculum_cfg=curriculum_cfg)

            traci.start([sumo_binary, '-c', self.sumocfg_path, '--tripinfo-output', os.path.join(self.sumocfg_dir, 'trips.trips.xml'), '--quit-on-end'])

            commitments = {}
            nstep_buffers = defaultdict(lambda: deque(maxlen=self.trainer.n_step))
            recent_edges = defaultdict(lambda: deque(maxlen=self.loop_window))
            teleported_controlled = set()
            failed_ids = set()
            arrived_true_dest = set()
            arrived_on_time = set()
            arrived_any = set()
            wrong_target = set()
            lane_failures = 0
            deficit_improvements = []
            loops = 0
            episode_return = 0.0
            removal_causes = defaultdict(int)
            terminal_step = {}
            final_step = 0
            route_diag = defaultdict(int)

            try:
                for step in range(MAX_SIMULATION_STEPS):
                    pre_step_on_destination = set()
                    final_step = step
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        break
                    self.update_edge_vehicle_counts(step, every=1)
                    vehicle_ids = list(traci.vehicle.getIDList())
                    global_stats = self._global_stats(set(vehicles.keys()), vehicles, step, teleported_controlled, failed_ids)

                    def push_nstep_transition(local_vid, transition):
                        buf = nstep_buffers[local_vid]
                        buf.append(transition)
                        terminal = transition["done"] >= 1.0
                        while buf and (len(buf) >= self.trainer.n_step or terminal):
                            horizon = min(len(buf), self.trainer.n_step)
                            first = buf[0]
                            R = 0.0
                            for i in range(horizon):
                                R += (self.trainer.gamma ** i) * buf[i]["reward"]
                            last = buf[horizon - 1]
                            out = {
                                "state": first["state"],
                                "action": first["action"],
                                "reward": R,
                                "next_state": last["next_state"],
                                "done": last["done"],
                                "next_mask": last["next_mask"],
                                "horizon": horizon,
                            }
                            self.trainer.remember(out)
                            buf.popleft()
                            if not terminal and len(buf) < self.trainer.n_step:
                                break
                        if terminal:
                            buf.clear()

                    for vid in vehicle_ids:
                        if vid not in vehicles:
                            continue
                        v = vehicles[vid]
                        edge = traci.vehicle.getRoadID(vid)
                        if edge not in self.connection_info.edge_index_dict:
                            continue
                        v.current_edge = edge
                        recent_edges[vid].append(edge)

                        if edge == v.destination:
                            pre_step_on_destination.add(vid)
                            continue

                        candidates = self.enumerate_candidate_next_edges(vid, edge, v.destination, step, diag=route_diag)
                        state, mask, cand_feat = self.build_state(vid, v, step, edge, candidates, recent_edges[vid], global_stats)

                        # Close commitments when execution outcome is known.
                        c = commitments.get(vid)
                        if c is not None:
                            curr_eta = self.estimate_eta(edge, v.destination)
                            time_left = max(v.deadline - step, 0.0)
                            curr_deficit = max(curr_eta - time_left, 0.0) if math.isfinite(curr_eta) else self.time_norm
                            entered_chosen = (edge == c.chosen_next_edge)
                            lane_m = self.compute_candidate_lane_metrics(vid, c.edge, c.chosen_next_edge) if c.edge == edge else {"feasible": True}
                            late_impossible = (not lane_m.get("feasible", True)) and not entered_chosen
                            repeated = 1.0 if edge in list(recent_edges[vid])[:-1] else 0.0
                            if repeated > 0:
                                loops += 1

                            if entered_chosen or late_impossible:
                                if entered_chosen:
                                    reward_base = self.execution_success_bonus
                                else:
                                    reward_base = 0.0
                                    lane_failures += 1
                                reward, done = self.compute_reward(
                                    c.prev_deficit,
                                    curr_deficit,
                                    c.chosen_density,
                                    self._global_density_mean,
                                    lane_failed=late_impossible,
                                    repeated=repeated + c.candidate_repeated,
                                )
                                reward += reward_base
                                deficit_improvements.append(c.prev_deficit - curr_deficit)
                                next_state = state
                                transition = {
                                    "state": c.state,
                                    "action": c.action,
                                    "reward": reward,
                                    "next_state": next_state,
                                    "done": float(done),
                                    "next_mask": mask.reshape(1, -1)[0],
                                }
                                push_nstep_transition(vid, transition)
                                commitments.pop(vid, None)
                                episode_return += reward

                        candidate_densities = [
                            self.connection_info.edge_vehicle_count.get(nxt, 0) / max(self.connection_info.edge_length_dict.get(nxt, 1.0), 1.0)
                            for nxt in candidates
                        ]
                        time_left = max(v.deadline - step, 0.0)
                        candidate_deficits = []
                        for nxt in candidates:
                            eta_next = self.estimate_eta(nxt, v.destination)
                            candidate_deficits.append(max(eta_next - time_left, 0.0) if math.isfinite(eta_next) else self.time_norm)
                        prev_density = commitments[vid].chosen_density if vid in commitments else self.connection_info.edge_vehicle_count.get(edge, 0) / max(self.connection_info.edge_length_dict.get(edge, 1.0), 1.0)
                        curr_density = self.connection_info.edge_vehicle_count.get((commitments[vid].chosen_next_edge if vid in commitments else edge), 0) / max(self.connection_info.edge_length_dict.get((commitments[vid].chosen_next_edge if vid in commitments else edge), 1.0), 1.0)
                        committed_eta = self.estimate_eta((commitments[vid].chosen_next_edge if vid in commitments else edge), v.destination)
                        committed_deficit = max(committed_eta - time_left, 0.0) if math.isfinite(committed_eta) else self.time_norm
                        if not self.should_make_decision(vid, edge, candidates, commitments.get(vid), prev_density, curr_density, candidate_densities, committed_deficit=committed_deficit, candidate_deficits=candidate_deficits):
                            continue

                        if np.sum(mask) <= 0:
                            continue
                        action = self.trainer.select_action(state, mask)
                        if action is None or action >= len(candidates):
                            continue
                        chosen_next = candidates[action]
                        lane_m = self.compute_candidate_lane_metrics(vid, edge, chosen_next)
                        if lane_m["target_lanes"]:
                            target_lane = min(lane_m["target_lanes"], key=lambda idx: abs(idx - traci.vehicle.getLaneIndex(vid)))
                            try:
                                traci.vehicle.changeLane(vid, int(target_lane), 60)
                            except Exception:
                                pass

                        apply_result = self.apply_route_commitment(vid, edge, chosen_next, v.destination)
                        if not apply_result.applied:
                            route_diag[apply_result.reason] += 1
                            if apply_result.reason == "downstream_path_missing":
                                route_diag["downstream_path_failures"] += 1
                            if apply_result.reason == "route_set_exception":
                                route_diag["route_set_exceptions"] += 1
                            bl_key = self._blacklist_key(vid, edge, chosen_next)
                            self._route_blacklist[bl_key] = step + self.route_retry_cooldown_steps
                            fail_penalty = 0.0
                            if apply_result.reason == "invalid_first_hop":
                                fail_penalty = self.invalid_first_hop_penalty
                            elif apply_result.reason == "downstream_path_missing":
                                fail_penalty = self.downstream_path_missing_penalty
                            elif apply_result.reason == "route_set_exception":
                                fail_penalty = self.route_set_exception_penalty
                            transition = {
                                "state": state,
                                "action": action,
                                "reward": fail_penalty,
                                "next_state": state,
                                "done": 0.0,
                                "next_mask": mask.reshape(1, -1)[0],
                            }
                            push_nstep_transition(vid, transition)
                            episode_return += fail_penalty
                            continue

                        eta_now = self.estimate_eta(edge, v.destination)
                        time_left = max(v.deadline - step, 0.0)
                        prev_deficit = max(eta_now - time_left, 0.0) if math.isfinite(eta_now) else self.time_norm
                        chosen_density = self.connection_info.edge_vehicle_count.get(chosen_next, 0) / max(self.connection_info.edge_length_dict.get(chosen_next, 1.0), 1.0)
                        commitments[vid] = Commitment(
                            state=state,
                            action=action,
                            edge=edge,
                            chosen_next_edge=chosen_next,
                            chosen_density=chosen_density,
                            prev_deficit=prev_deficit,
                            step=step,
                            candidate_repeated=(1.0 if chosen_next in recent_edges[vid] else 0.0),
                        )

                    traci.simulationStep()

                    for aid in traci.simulation.getArrivedIDList():
                        if aid not in vehicles:
                            continue
                        if aid in terminal_step:
                            continue
                        v = vehicles[aid]
                        arrived_any.add(aid)
                        is_true = (aid in pre_step_on_destination)
                        if is_true:
                            arrived_true_dest.add(aid)
                            if step <= v.deadline:
                                arrived_on_time.add(aid)
                            removal_causes["reached_true_destination"] += 1
                        else:
                            wrong_target.add(aid)
                            failed_ids.add(aid)
                            removal_causes["reached_wrong_target"] += 1
                        if aid in commitments:
                            c = commitments.pop(aid)
                            lateness = max(step - v.deadline, 0.0)
                            last_edge = v.current_edge or c.edge
                            curr_eta = self.estimate_eta(last_edge, v.destination)
                            curr_deficit = max(curr_eta - max(v.deadline - step, 0.0), 0.0) if math.isfinite(curr_eta) else self.time_norm
                            reward, done = self.compute_reward(c.prev_deficit, curr_deficit, c.chosen_density, self._global_density_mean, wrong_target=(not is_true), arrived=is_true, on_time=(step <= v.deadline), lateness=lateness)
                            transition = {"state": c.state, "action": c.action, "reward": reward, "next_state": np.zeros((1, self.state_size), dtype=np.float32), "done": float(done), "next_mask": np.zeros((self.action_size,), dtype=np.float32)}
                            push_nstep_transition(aid, transition)
                            episode_return += reward
                        terminal_step[aid] = step

                    tele = self.get_teleport_ids()
                    for tid in tele:
                        if tid not in vehicles:
                            continue
                        if tid in terminal_step:
                            continue
                        teleported_controlled.add(tid)
                        failed_ids.add(tid)
                        removal_causes["teleport"] += 1
                        if tid in commitments:
                            c = commitments.pop(tid)
                            reward, done = self.compute_reward(c.prev_deficit, c.prev_deficit + 5.0, c.chosen_density, self._global_density_mean, teleported=True)
                            transition = {"state": c.state, "action": c.action, "reward": reward, "next_state": np.zeros((1, self.state_size), dtype=np.float32), "done": float(done), "next_mask": np.zeros((self.action_size,), dtype=np.float32)}
                            push_nstep_transition(tid, transition)
                            episode_return += reward
                        terminal_step[tid] = step

                    if step % self.train_every == 0:
                        for _ in range(self.grad_steps):
                            self.trainer.replay()

                # unresolved vehicles
                controlled_ids = set(vehicles.keys())
                disappeared = controlled_ids - set(traci.vehicle.getIDList()) - arrived_any - teleported_controlled
                for vid in disappeared:
                    if vid not in arrived_any and vid not in teleported_controlled and vid not in terminal_step:
                        failed_ids.add(vid)
                        removal_causes["disappeared"] += 1
                        terminal_step[vid] = final_step

                for vid, c in list(commitments.items()):
                    if vid in terminal_step:
                        commitments.pop(vid, None)
                        continue
                    v = vehicles[vid]
                    late = max(final_step - v.deadline, 0.0)
                    remain_dist = self.get_distance_to_destination(v.current_edge or c.edge, v.destination)
                    reward, done = self.compute_reward(c.prev_deficit, c.prev_deficit + 2.0, c.chosen_density, self._global_density_mean, deadline_missed=True, lateness=late, remain_dist=(remain_dist if math.isfinite(remain_dist) else 1000.0))
                    transition = {"state": c.state, "action": c.action, "reward": reward, "next_state": np.zeros((1, self.state_size), dtype=np.float32), "done": float(done), "next_mask": np.zeros((self.action_size,), dtype=np.float32)}
                    push_nstep_transition(vid, transition)
                    episode_return += reward
                    commitments.pop(vid, None)
                    failed_ids.add(vid)
                    removal_causes["dead_end_trapped"] += 1
                    terminal_step[vid] = final_step

                for vid, buf in list(nstep_buffers.items()):
                    while buf:
                        horizon = min(len(buf), self.trainer.n_step)
                        first = buf[0]
                        R = 0.0
                        for i in range(horizon):
                            R += (self.trainer.gamma ** i) * buf[i]["reward"]
                        last = buf[horizon - 1]
                        out = {
                            "state": first["state"],
                            "action": first["action"],
                            "reward": R,
                            "next_state": last["next_state"],
                            "done": last["done"],
                            "next_mask": last["next_mask"],
                            "horizon": horizon,
                        }
                        self.trainer.remember(out)
                        buf.popleft()

            finally:
                traci.close()

            total = max(len(vehicles), 1)
            on_time = len(arrived_on_time)
            completion_before_deadline = on_time / float(total)
            true_arrival_rate = len(arrived_true_dest) / float(total)
            wrong_target_rate = len(wrong_target) / float(total)
            teleport_rate = len(teleported_controlled) / float(total)
            exit_wo_dest_rate = len((set(vehicles.keys()) - arrived_true_dest - teleported_controlled)) / float(total)
            avg_lateness = float(np.mean([max(terminal_step.get(vid, final_step) - vehicles[vid].deadline, 0.0) for vid in vehicles])) if vehicles else 0.0
            late_vehicles = [vid for vid in vehicles if max(terminal_step.get(vid, final_step) - vehicles[vid].deadline, 0.0) > 0]
            avg_lateness_late_only = float(np.mean([max(terminal_step.get(vid, final_step) - vehicles[vid].deadline, 0.0) for vid in late_vehicles])) if late_vehicles else 0.0
            avg_deficit_improvement = float(np.mean(deficit_improvements)) if deficit_improvements else 0.0
            lane_failure_rate = lane_failures / float(max(len(deficit_improvements), 1))
            loop_rate = loops / float(max(len(deficit_improvements), 1))

            rolling_baseline.append(completion_before_deadline)
            base = float(np.mean(rolling_baseline)) if rolling_baseline else completion_before_deadline
            shared_bonus_triggered = completion_before_deadline > base

            metrics = {
                "episode": episode,
                "true_destination_arrival_rate": true_arrival_rate,
                "completion_before_deadline_rate": completion_before_deadline,
                "wrong_target_arrival_rate": wrong_target_rate,
                "exit_without_destination_rate": exit_wo_dest_rate,
                "non_true_destination_non_teleport_rate": exit_wo_dest_rate,
                "teleport_rate": teleport_rate,
                "average_deadline_lateness": avg_lateness,
                "average_deadline_lateness_late_only": avg_lateness_late_only,
                "average_deficit_improvement": avg_deficit_improvement,
                "lane_execution_failure_rate": lane_failure_rate,
                "loop_oscillation_rate": loop_rate,
                "replay_td_error_stats": dict(self.trainer.last_td_error_stats),
                "removals_by_cause": dict(removal_causes),
                "route_failure_diagnostics": dict(route_diag),
                "shared_bonus_triggered": shared_bonus_triggered,
                "episode_return": episode_return / float(total),
            }
            metrics_history.append(metrics)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    rolling[k].append(float(v))

            self.trainer.epsilon = max(self.trainer.epsilon_min, self.trainer.epsilon * self.trainer.epsilon_decay)
            print(
                f"Ep {episode} | arr_true={true_arrival_rate:.3f} on_time={completion_before_deadline:.3f} "
                f"wrong_target={wrong_target_rate:.3f} exit_wo_dest={exit_wo_dest_rate:.3f} tele={teleport_rate:.3f} "
                f"lane_fail={lane_failure_rate:.3f} loop={loop_rate:.3f} td_mean={self.trainer.last_td_error_stats['mean']:.4f}"
            )
            print(f"Removal causes: {dict(removal_causes)}")
            print(f"Route diagnostics: {dict(route_diag)}")

        self.trainer.model.save(self.model_output_path)
        return metrics_history
