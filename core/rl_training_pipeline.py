import math
import os
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

"""
Migration note (old -> new pipeline):
- Old design coupled "choose next edge" with immediate setRoute and ad-hoc route stitching.
  This violated SUMO lane/routing semantics and caused recurring command 0xc4 failures
  ("Invalid route replacement").
- New design introduces a two-stage controller:
  (1) Strategic intent selection over a feasibility-filtered action mask.
  (2) Tactical execution gate that commits only when lane/junction/signal constraints are safe.
- Route application now uses SUMO-native route construction (traci.simulation.findRoute)
  and explicit pre-commit validation, preventing invalid route replacements by design.
- Learning objective remains SELFLESS: reward is dominated by fleet-level deadline deficit
  improvement, with local safety/tactical penalties as secondary shaping.

Diagnostics checklist (if warnings remain):
1) Inspect diagnostics counters for pre-blocked invalid proposals (should be high before any 0xc4).
2) Verify tactical defer reasons near junctions / red zones / best-lane infeasibility.
3) Check route proposal logs: current edge/lane, intent, SUMO route, route-valid flag.
4) Confirm emergency-brake counter trends downward as defer-zone tuning is adjusted.
5) Check oscillation and repeated-intent cooldown counters for excessive re-decisions.
"""


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
class StrategicIntent:
    intent_type: str  # keep, delay, reroute
    target_edge: Optional[str]
    route_edges: List[str]
    reason: str = "ok"


@dataclass
class IntentProposal:
    intent: StrategicIntent
    features: np.ndarray
    valid: bool


@dataclass
class PendingIntent:
    state: np.ndarray
    action: int
    intent: StrategicIntent
    created_step: int
    prev_global_deficit: float


class FleetMetricComputer:
    def __init__(self, eta_fn, time_norm=1200.0):
        self.eta_fn = eta_fn
        self.time_norm = float(time_norm)

    def global_deadline_deficit(self, controlled_ids, vehicles, step):
        total = 0.0
        per_vehicle = {}
        for vid in controlled_ids:
            if vid not in vehicles:
                continue
            v = vehicles[vid]
            edge = v.current_edge
            if not edge:
                continue
            eta = self.eta_fn(edge, v.destination)
            tleft = max(float(v.deadline) - float(step), 0.0)
            deficit = max(eta - tleft, 0.0) if math.isfinite(eta) else self.time_norm
            per_vehicle[vid] = deficit
            total += deficit
        return total, per_vehicle


class RLTrainingPipeline:
    def __init__(
        self,
        sumocfg_path,
        model_output_path,
        episodes=10,
        spawn_interval=4.0,
        seed_with_episode=True,
        candidate_slots=8,
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

        # Reward weights: global/selfless objective dominates.
        self.w_global = 12.0
        self.w_local_externality = 1.0
        self.w_invalid_blocked = 1.5
        self.w_tactical_defer = 0.3
        self.w_oscillation = 2.0
        self.w_harsh_brake = 5.0
        self.teleport_penalty = -180.0
        self.arrival_on_time_reward = 180.0
        self.arrival_late_reward = 20.0
        self.deadline_miss_penalty = -50.0

        self.speed_norm = 20.0
        self.dist_norm = 300.0
        self.time_norm = 1200.0
        self.max_lane_shift_norm = 4.0

        # Tactical gating configuration.
        self.min_commit_distance = 35.0
        self.red_stop_zone_distance = 25.0
        self.intent_ttl = 12
        self.oscillation_window = 20
        self.route_retry_cooldown_steps = 30

        self.sumocfg_dir = os.path.dirname(sumocfg_path)
        self.net_file, self.route_file = self.parse_sumocfg(sumocfg_path)
        self.net = sumolib.net.readNet(os.path.join(self.sumocfg_dir, self.net_file))
        self.connection_info = ConnectionInfo(os.path.join(self.sumocfg_dir, self.net_file))
        self.route_helper = TrainingRouteHelper(self.connection_info)

        self._distance_cache = {}
        self._eta_cache = {}
        self._route_feasibility_cache = {}
        self._failed_intent_cooldown = {}
        self._curriculum = [
            {"until": 0.33, "target": 12, "random": 16, "slack": (1.35, 1.55)},
            {"until": 0.66, "target": 16, "random": 24, "slack": (1.20, 1.40)},
            {"until": 1.00, "target": 20, "random": 30, "slack": (1.10, 1.30)},
        ]

        self.base_feature_size = 22
        self.intent_feature_size = 10
        self.state_size = self.base_feature_size + self.candidate_slots * self.intent_feature_size
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
        self.metric_computer = FleetMetricComputer(self.estimate_eta, time_norm=self.time_norm)

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
        # Dynamic ETA via SUMO-native route computation (refresh frequently).
        key = (edge_id, destination_edge, int(traci.simulation.getTime() // 10 if traci.isLoaded() else 0))
        if key in self._eta_cache:
            return self._eta_cache[key]
        try:
            route = traci.simulation.findRoute(edge_id, destination_edge)
            edges = list(route.edges) if route and route.edges else []
            if not edges:
                eta = math.inf
            else:
                # Use SUMO route travel time when available; fall back to edge means.
                tt = float(getattr(route, "travelTime", 0.0) or 0.0)
                if tt <= 0:
                    tt = 0.0
                    for e in edges:
                        ms = max(float(traci.edge.getLastStepMeanSpeed(e)), 3.0)
                        ln = max(float(self.connection_info.edge_length_dict.get(e, 25.0)), 5.0)
                        tt += ln / ms
                eta = tt
        except Exception:
            eta = math.inf
        self._eta_cache[key] = eta
        return eta

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

    def _edge_from_lane_id(self, lane_id):
        if lane_id is None or "_" not in lane_id:
            return None
        return lane_id.rsplit("_", 1)[0]

    def _lane_successors(self, lane_id):
        succ = set()
        try:
            links = traci.lane.getLinks(lane_id)
        except Exception:
            links = []
        for link in links:
            if not link:
                continue
            nxt_lane = link[0]
            nxt_edge = self._edge_from_lane_id(nxt_lane)
            if nxt_edge:
                succ.add(nxt_edge)
        return succ

    def _vehicle_legal_next_edges(self, vehicle_id, edge_id):
        legal = set()
        for lane_id in self.connection_info.edge_lane_ids.get(edge_id, []):
            legal.update(self._lane_successors(lane_id))
        return legal

    def _distance_to_lane_end(self, vehicle_id):
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        return max(traci.lane.getLength(lane_id) - traci.vehicle.getLanePosition(vehicle_id), 0.0)

    def _best_lane_reachable(self, vehicle_id, required_next_edge):
        # SUMO best-lane guidance can signal if continuation is realistic from current tactical state.
        try:
            best = traci.vehicle.getBestLanes(vehicle_id)
        except Exception:
            best = []
        if not best:
            return True
        for entry in best:
            # entry schema may vary by SUMO version; last field often list of reachable continuations.
            if len(entry) >= 6 and isinstance(entry[5], (list, tuple)):
                cont = set(entry[5])
                if required_next_edge in cont:
                    return True
        return False

    def _route_cache_key(self, vehicle_id, current_edge, lane_id, target_edge, destination):
        return (vehicle_id, current_edge, lane_id, target_edge or "", destination)

    def _cooldown_key(self, vehicle_id, current_edge, target_edge):
        return (vehicle_id, current_edge, target_edge or "")

    def _build_route_via_target(self, vehicle_id, current_edge, target_edge, destination):
        # Route construction is SUMO-native to avoid illegal ad-hoc concatenation.
        try:
            r1 = traci.simulation.findRoute(current_edge, target_edge, vType=traci.vehicle.getTypeID(vehicle_id))
            if not r1 or not r1.edges:
                return [], "no_valid_sumo_route_from_current"
            r2 = traci.simulation.findRoute(target_edge, destination, vType=traci.vehicle.getTypeID(vehicle_id))
            if not r2 or not r2.edges:
                return [], "no_valid_sumo_route_to_destination"
            merged = list(r1.edges)
            tail = list(r2.edges)
            if merged and tail and merged[-1] == tail[0]:
                merged.extend(tail[1:])
            else:
                merged.extend(tail)
            if not merged:
                return [], "empty_route"
            return merged, "ok"
        except Exception:
            return [], "find_route_exception"

    def _build_keep_route(self, vehicle_id, current_edge, destination):
        try:
            r = traci.simulation.findRoute(current_edge, destination, vType=traci.vehicle.getTypeID(vehicle_id))
            return list(r.edges) if r and r.edges else []
        except Exception:
            return []

    def _first_actionable_next_edge(self, route_edges, current_edge):
        if not route_edges:
            return None
        if route_edges[0] == current_edge:
            return route_edges[1] if len(route_edges) > 1 else None
        return route_edges[0]

    def _lane_shift_estimate(self, vehicle_id, edge_id, next_edge):
        lane_ids = self.connection_info.edge_lane_ids.get(edge_id, [])
        curr_lane = traci.vehicle.getLaneIndex(vehicle_id)
        targets = []
        for idx, lane in enumerate(lane_ids):
            for _d, out_edge in self.connection_info.lane_outgoing_edges_dict.get(lane, {}).items():
                if out_edge == next_edge:
                    targets.append(idx)
                    break
        if not targets:
            return 99
        return min(abs(curr_lane - t) for t in targets)

    def _propose_intents(self, vehicle_id, vehicle, edge_id, destination, step, diag):
        proposals = []
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        legal_next = self._vehicle_legal_next_edges(vehicle_id, edge_id)

        keep_route = self._build_keep_route(vehicle_id, edge_id, destination)
        proposals.append(IntentProposal(StrategicIntent("keep", None, keep_route, reason="keep"), np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), valid=bool(keep_route)))

        # Delay action explicitly represents strategic hold when tactical state is not ready.
        proposals.append(IntentProposal(StrategicIntent("delay", None, [], reason="delay"), np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), valid=True))

        outgoing = list(dict.fromkeys(self.connection_info.outgoing_edges_dict.get(edge_id, {}).values()))
        for target in outgoing:
            ckey = self._cooldown_key(vehicle_id, edge_id, target)
            if self._failed_intent_cooldown.get(ckey, -1) >= step:
                diag["filtered_tactical_cooldown"] += 1
                continue

            cache_key = self._route_cache_key(vehicle_id, edge_id, lane_id, target, destination)
            if cache_key in self._route_feasibility_cache:
                route_edges, reason = self._route_feasibility_cache[cache_key]
            else:
                route_edges, reason = self._build_route_via_target(vehicle_id, edge_id, target, destination)
                self._route_feasibility_cache[cache_key] = (route_edges, reason)
            if not route_edges:
                diag[f"filtered_{reason}"] += 1
                continue

            first_next = self._first_actionable_next_edge(route_edges, edge_id)
            if first_next is None or first_next not in legal_next:
                diag["filtered_no_legal_lane_successor"] += 1
                continue

            eta = self.estimate_eta(target, destination)
            tleft = max(vehicle.deadline - step, 0.0)
            deficit = max(eta - tleft, 0.0) if math.isfinite(eta) else self.time_norm
            density = self.connection_info.edge_vehicle_count.get(target, 0) / max(self.connection_info.edge_length_dict.get(target, 1.0), 1.0)
            shift = self._lane_shift_estimate(vehicle_id, edge_id, first_next)
            feat = np.array([
                0,
                0,
                1,
                np.clip(density, 0.0, 2.0) / 2.0,
                np.clip((eta if math.isfinite(eta) else self.time_norm) / self.time_norm, 0.0, 2.0) / 2.0,
                np.clip(deficit / self.time_norm, 0.0, 1.0),
                np.clip(shift / self.max_lane_shift_norm, 0.0, 1.0),
                1.0 if self._best_lane_reachable(vehicle_id, first_next) else 0.0,
                1.0 if first_next == target else 0.5,
                1.0,
            ], dtype=np.float32)
            proposals.append(IntentProposal(StrategicIntent("reroute", target, route_edges, reason="ok"), feat, valid=True))

        return proposals[: self.candidate_slots]

    def _signal_or_stop_zone(self, vehicle_id):
        # Conservative signal/stop-zone gating to avoid last-second unsafe reroutes.
        try:
            tls = traci.vehicle.getNextTLS(vehicle_id)
        except Exception:
            tls = []
        if not tls:
            return False
        for item in tls:
            if len(item) < 3:
                continue
            dist = float(item[2])
            if dist <= self.red_stop_zone_distance:
                return True
        return False

    def _tactical_gate(self, vehicle_id, intent: StrategicIntent, step, diag) -> Tuple[bool, str]:
        if intent.intent_type in ("keep", "delay"):
            return True, "safe_hold"

        dist_to_end = self._distance_to_lane_end(vehicle_id)
        if dist_to_end < self.min_commit_distance:
            diag["filtered_too_close_to_junction"] += 1
            return False, "too_close_to_junction"

        if self._signal_or_stop_zone(vehicle_id):
            diag["filtered_red_stop_zone"] += 1
            return False, "red_stop_zone"

        current_edge = traci.vehicle.getRoadID(vehicle_id)
        first_next = self._first_actionable_next_edge(intent.route_edges, current_edge)
        if first_next is None:
            diag["filtered_empty_actionable_hop"] += 1
            return False, "empty_actionable_hop"

        lane_succ = self._lane_successors(traci.vehicle.getLaneID(vehicle_id))
        if first_next not in lane_succ:
            diag["filtered_current_lane_no_successor"] += 1
            return False, "lane_not_ready"

        if not self._best_lane_reachable(vehicle_id, first_next):
            diag["filtered_best_lane_infeasible"] += 1
            return False, "best_lane_infeasible"

        return True, "executable"

    def _apply_intent(self, vehicle_id, intent: StrategicIntent, diag):
        if intent.intent_type in ("keep", "delay"):
            return True, "noop"
        if not intent.route_edges:
            return False, "empty_route"
        try:
            traci.vehicle.setRoute(vehicle_id, intent.route_edges)
            try:
                if not traci.vehicle.isRouteValid(vehicle_id):
                    diag["route_application_invalid_after_set"] += 1
                    return False, "route_invalid_after_set"
            except Exception:
                pass
            return True, "applied"
        except Exception:
            diag["route_application_failures"] += 1
            return False, "set_route_exception"

    def _global_stats(self, controlled_ids, vehicles, step, teleported, failed):
        total = max(len(controlled_ids), 1)
        global_deficit, per_vehicle = self.metric_computer.global_deadline_deficit(controlled_ids, vehicles, step)
        behind = sum(1 for _vid, d in per_vehicle.items() if d > 0.0)
        return {
            "mean_density": self._global_density_mean,
            "std_density": self._global_density_std,
            "frac_behind": behind / float(total),
            "frac_failed": (len(teleported) + len(failed)) / float(total),
            "global_deficit": global_deficit,
        }

    def _build_state(self, vehicle_id, vehicle, step, proposals, global_stats, recent_actions):
        edge = traci.vehicle.getRoadID(vehicle_id)
        lane_idx = traci.vehicle.getLaneIndex(vehicle_id)
        n_lanes = max(traci.edge.getLaneNumber(edge), 1)
        dist_to_end = self._distance_to_lane_end(vehicle_id)
        speed = traci.vehicle.getSpeed(vehicle_id)
        time_left = max(float(vehicle.deadline) - float(step), 0.0)
        eta_curr = self.estimate_eta(edge, vehicle.destination)
        slack = (time_left - eta_curr) if math.isfinite(eta_curr) else -self.time_norm
        base = np.array([
            lane_idx / max(n_lanes - 1, 1),
            min(n_lanes, 6) / 6.0,
            np.clip(dist_to_end / self.dist_norm, 0.0, 1.0),
            np.clip(speed / self.speed_norm, 0.0, 1.5) / 1.5,
            np.clip(time_left / self.time_norm, 0.0, 1.0),
            np.clip((eta_curr if math.isfinite(eta_curr) else self.time_norm) / self.time_norm, 0.0, 2.0) / 2.0,
            np.clip(slack / self.time_norm, -1.0, 1.0),
            np.clip(global_stats["mean_density"], 0.0, 2.0) / 2.0,
            np.clip(global_stats["std_density"], 0.0, 1.0),
            np.clip(global_stats["frac_behind"], 0.0, 1.0),
            np.clip(global_stats["frac_failed"], 0.0, 1.0),
            np.clip(global_stats["global_deficit"] / (self.time_norm * max(1, len(proposals))), 0.0, 1.0),
            1.0 if self._signal_or_stop_zone(vehicle_id) else 0.0,
            1.0 if dist_to_end < self.min_commit_distance else 0.0,
            np.clip(self.connection_info.edge_vehicle_count.get(edge, 0) / max(self.connection_info.edge_length_dict.get(edge, 1.0), 1.0), 0.0, 2.0) / 2.0,
            1.0 if len(recent_actions) >= 2 and recent_actions[-1] != recent_actions[-2] else 0.0,
            np.clip(len(proposals) / float(max(self.candidate_slots, 1)), 0.0, 1.0),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ], dtype=np.float32)

        cand = np.zeros((self.candidate_slots, self.intent_feature_size), dtype=np.float32)
        mask = np.zeros((self.candidate_slots,), dtype=np.float32)
        for i, p in enumerate(proposals[: self.candidate_slots]):
            cand[i] = p.features
            if p.valid:
                mask[i] = 1.0
        return np.concatenate([base, cand.reshape(-1)], axis=0).reshape(1, -1), mask

    def _reward(self, prev_global_deficit, new_global_deficit, blocked_invalid=0, tactical_defer=0, oscillation=0, harsh_brake=0, arrived=False, on_time=False, teleported=False, deadline_missed=False):
        reward = self.w_global * (prev_global_deficit - new_global_deficit)
        reward -= self.w_invalid_blocked * float(blocked_invalid)
        reward -= self.w_tactical_defer * float(tactical_defer)
        reward -= self.w_oscillation * float(oscillation)
        reward -= self.w_harsh_brake * float(harsh_brake)
        if teleported:
            return reward + self.teleport_penalty, True
        if arrived:
            return reward + (self.arrival_on_time_reward if on_time else self.arrival_late_reward), True
        if deadline_missed:
            return reward + self.deadline_miss_penalty, True
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
        metrics_history = []

        for episode in range(self.episodes):
            if self.seed_with_episode:
                random.seed(episode)
                np.random.seed(episode)
            curriculum_cfg = self.get_curriculum_config(episode)
            vehicles = self.generate_episode_vehicles(episode_seed=(episode if self.seed_with_episode else None), curriculum_cfg=curriculum_cfg)

            traci.start([sumo_binary, '-c', self.sumocfg_path, '--tripinfo-output', os.path.join(self.sumocfg_dir, 'trips.trips.xml'), '--quit-on-end'])

            pending: Dict[str, PendingIntent] = {}
            nstep_buffers = defaultdict(lambda: deque(maxlen=self.trainer.n_step))
            teleported_controlled = set()
            failed_ids = set()
            arrived_on_time = set()
            arrived_any = set()
            removal_causes = defaultdict(int)
            route_diag = defaultdict(int)
            recent_actions = defaultdict(lambda: deque(maxlen=self.oscillation_window))
            prev_speeds = {}
            terminal_step = {}
            episode_return = 0.0
            final_step = 0

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

            try:
                for step in range(MAX_SIMULATION_STEPS):
                    final_step = step
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        break
                    self.update_edge_vehicle_counts(step, every=1)
                    ids = list(traci.vehicle.getIDList())
                    for vid in ids:
                        if vid in vehicles:
                            vehicles[vid].current_edge = traci.vehicle.getRoadID(vid)
                    g = self._global_stats(set(vehicles.keys()), vehicles, step, teleported_controlled, failed_ids)

                    for vid in ids:
                        if vid not in vehicles:
                            continue
                        v = vehicles[vid]
                        edge = v.current_edge
                        if edge not in self.connection_info.edge_index_dict:
                            continue

                        # Emergency braking monitor.
                        curr_speed = traci.vehicle.getSpeed(vid)
                        prev_speed = prev_speeds.get(vid, curr_speed)
                        harsh_brake = 1 if (prev_speed - curr_speed) > 4.5 else 0
                        prev_speeds[vid] = curr_speed
                        if harsh_brake:
                            route_diag["emergency_braking_events"] += 1

                        proposals = self._propose_intents(vid, v, edge, v.destination, step, route_diag)
                        state, mask = self._build_state(vid, v, step, proposals, g, recent_actions[vid])
                        if np.sum(mask) <= 0:
                            route_diag["no_actions_after_filter"] += 1
                            continue

                        # Resolve pending intent first (stage 2 tactical executor).
                        if vid in pending:
                            p = pending[vid]
                            if step - p.created_step > self.intent_ttl:
                                route_diag["pending_intent_expired"] += 1
                                pending.pop(vid, None)
                            else:
                                ok, reason = self._tactical_gate(vid, p.intent, step, route_diag)
                                if ok:
                                    # Log route proposal before application.
                                    route_diag["route_proposals_attempted"] += 1
                                    success, app_reason = self._apply_intent(vid, p.intent, route_diag)
                                    if success:
                                        if p.intent.intent_type == "reroute":
                                            route_diag["reroute_applied"] += 1
                                        now_global, _ = self.metric_computer.global_deadline_deficit(set(vehicles.keys()), vehicles, step)
                                        rew, done = self._reward(p.prev_global_deficit, now_global, harsh_brake=harsh_brake)
                                        transition = {
                                            "state": p.state,
                                            "action": p.action,
                                            "reward": rew,
                                            "next_state": state,
                                            "done": float(done),
                                            "next_mask": mask.reshape(1, -1)[0],
                                        }
                                        push_nstep_transition(vid, transition)
                                        episode_return += rew
                                        pending.pop(vid, None)
                                    else:
                                        route_diag["route_application_failures"] += 1
                                        ck = self._cooldown_key(vid, edge, p.intent.target_edge)
                                        self._failed_intent_cooldown[ck] = step + self.route_retry_cooldown_steps
                                        pending.pop(vid, None)
                                        rew, _done = self._reward(p.prev_global_deficit, p.prev_global_deficit, blocked_invalid=1, harsh_brake=harsh_brake)
                                        transition = {
                                            "state": p.state,
                                            "action": p.action,
                                            "reward": rew,
                                            "next_state": state,
                                            "done": 0.0,
                                            "next_mask": mask.reshape(1, -1)[0],
                                        }
                                        push_nstep_transition(vid, transition)
                                        episode_return += rew
                                else:
                                    route_diag[f"tactical_defer_{reason}"] += 1
                            continue

                        action = self.trainer.select_action(state, mask)
                        if action is None or action >= len(proposals):
                            continue
                        selected = proposals[action].intent
                        recent_actions[vid].append(selected.intent_type + ":" + str(selected.target_edge))
                        oscillation = 1 if len(recent_actions[vid]) >= 2 and recent_actions[vid][-1] != recent_actions[vid][-2] else 0
                        if oscillation:
                            route_diag["oscillation_events"] += 1

                        prev_global, _ = self.metric_computer.global_deadline_deficit(set(vehicles.keys()), vehicles, step)
                        if selected.intent_type in ("keep", "delay"):
                            new_global, _ = self.metric_computer.global_deadline_deficit(set(vehicles.keys()), vehicles, step)
                            rew, done = self._reward(prev_global, new_global, tactical_defer=(1 if selected.intent_type == "delay" else 0), oscillation=oscillation, harsh_brake=harsh_brake)
                            transition = {
                                "state": state,
                                "action": action,
                                "reward": rew,
                                "next_state": state,
                                "done": float(done),
                                "next_mask": mask.reshape(1, -1)[0],
                            }
                            push_nstep_transition(vid, transition)
                            episode_return += rew
                            continue

                        ok, reason = self._tactical_gate(vid, selected, step, route_diag)
                        if not ok:
                            pending[vid] = PendingIntent(state=state, action=action, intent=selected, created_step=step, prev_global_deficit=prev_global)
                            route_diag[f"pending_due_to_{reason}"] += 1
                            continue

                        route_diag["route_proposals_attempted"] += 1
                        success, app_reason = self._apply_intent(vid, selected, route_diag)
                        if not success:
                            ck = self._cooldown_key(vid, edge, selected.target_edge)
                            self._failed_intent_cooldown[ck] = step + self.route_retry_cooldown_steps
                            rew, done = self._reward(prev_global, prev_global, blocked_invalid=1, oscillation=oscillation, harsh_brake=harsh_brake)
                            transition = {
                                "state": state,
                                "action": action,
                                "reward": rew,
                                "next_state": state,
                                "done": float(done),
                                "next_mask": mask.reshape(1, -1)[0],
                            }
                            push_nstep_transition(vid, transition)
                            episode_return += rew
                            continue

                        now_global, _ = self.metric_computer.global_deadline_deficit(set(vehicles.keys()), vehicles, step)
                        rew, done = self._reward(prev_global, now_global, oscillation=oscillation, harsh_brake=harsh_brake)
                        transition = {
                            "state": state,
                            "action": action,
                            "reward": rew,
                            "next_state": state,
                            "done": float(done),
                            "next_mask": mask.reshape(1, -1)[0],
                        }
                        push_nstep_transition(vid, transition)
                        episode_return += rew

                    traci.simulationStep()

                    for aid in traci.simulation.getArrivedIDList():
                        if aid not in vehicles or aid in terminal_step:
                            continue
                        arrived_any.add(aid)
                        v = vehicles[aid]
                        on_time = (step <= v.deadline)
                        if on_time:
                            arrived_on_time.add(aid)
                        removal_causes["arrived"] += 1
                        terminal_step[aid] = step

                    for tid in traci.simulation.getStartingTeleportIDList():
                        if tid not in vehicles or tid in terminal_step:
                            continue
                        teleported_controlled.add(tid)
                        failed_ids.add(tid)
                        removal_causes["teleport"] += 1
                        terminal_step[tid] = step

                    if step % self.train_every == 0:
                        for _ in range(self.grad_steps):
                            self.trainer.replay()

                controlled_ids = set(vehicles.keys())
                for vid in controlled_ids:
                    if vid not in terminal_step:
                        if vid not in traci.vehicle.getIDList():
                            failed_ids.add(vid)
                            terminal_step[vid] = final_step
                            removal_causes["disappeared"] += 1

                for vid, p in list(pending.items()):
                    rew, done = self._reward(p.prev_global_deficit, p.prev_global_deficit, deadline_missed=True)
                    transition = {
                        "state": p.state,
                        "action": p.action,
                        "reward": rew,
                        "next_state": np.zeros((1, self.state_size), dtype=np.float32),
                        "done": float(done),
                        "next_mask": np.zeros((self.action_size,), dtype=np.float32),
                    }
                    push_nstep_transition(vid, transition)
                    episode_return += rew
                    pending.pop(vid, None)

            finally:
                traci.close()

            total = max(len(vehicles), 1)
            on_time_rate = len(arrived_on_time) / float(total)
            teleport_rate = len(teleported_controlled) / float(total)
            miss_rate = len([vid for vid in vehicles if max(terminal_step.get(vid, final_step) - vehicles[vid].deadline, 0.0) > 0]) / float(total)
            metrics = {
                "episode": episode,
                "on_time_rate": on_time_rate,
                "teleport_rate": teleport_rate,
                "deadline_miss_rate": miss_rate,
                "episode_return": episode_return / float(total),
                "replay_td_error_stats": dict(self.trainer.last_td_error_stats),
                "route_failure_diagnostics": dict(route_diag),
                "removals_by_cause": dict(removal_causes),
            }
            metrics_history.append(metrics)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    rolling[k].append(float(v))

            self.trainer.epsilon = max(self.trainer.epsilon_min, self.trainer.epsilon * self.trainer.epsilon_decay)
            print(
                f"Ep {episode} | on_time={on_time_rate:.3f} tele={teleport_rate:.3f} "
                f"miss={miss_rate:.3f} td_mean={self.trainer.last_td_error_stats['mean']:.4f}"
            )
            print(f"Diagnostics: {dict(route_diag)}")

        self.trainer.model.save(self.model_output_path)
        return metrics_history
