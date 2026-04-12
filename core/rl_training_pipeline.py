import numpy as np
import os
import sys
import math
import csv

import os, psutil

from xml.dom.minidom import parse
from keras.layers import Dense
from keras.models import Sequential, clone_model
from keras.losses import Huber
from keras.optimizers import Adam
from collections import defaultdict, deque
import random
from controller.RouteController import RouteController
from core.junction_decision_engine import JunctionDecisionEngine, PendingDecision
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

"""
In this file, we build a DQN network
"""

MAX_SIMULATION_STEPS = 2000 # This is the limit for each episode. Because vehicle might be stuck in infinite loop

class ReplayBuffer:
    """
    This is the replay buffer mechanism in DQN
    """
    def __init__(self, capacity):
        """
        :param capacity: Maximum number of transitions
        """
        self.buffer = deque(maxlen=capacity) # Use deque here so you can pop the first element later easily
        self.capacity = capacity

    def add(self, state, action, reward, next_state, done, next_valid_actions=None, metadata=None):
        """      
        Store one transition into the buffer
        """
        self.buffer.append((state, action, reward, next_state, done, next_valid_actions, metadata or {}))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class TrainingRouteHelper(RouteController):
    """
    Helper class to reuse compute_local_target during RL training and use connection_info
    """

    def __init__(self, connection_info):
        super().__init__(connection_info)

    def make_decisions(self, vehicles, connection_info):
        return {}

class DQNTrainer:
    """
    Deep Q-Network trainer for routing decisions
    """

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.05,
        replay_capacity=2000,
        batch_size=64,
        replay_warmup=1000,
        target_update_every=200,
        target_soft_tau=1.0,
    ):
        """
        :param learning_rate: Can be adjusted for further optimization
        :param gamma: Can be adjusted for further optimization
        :param epsilon: 1.0 allows free exploration
        :param epsilon_decay: epsilon value in next episode
        :param epsilon_min: minimum epsilon to ensure that there's still some chance for free exploration later
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.replay_warmup = max(int(replay_warmup), self.batch_size)
        self.target_update_every = max(int(target_update_every), 1)
        self.target_soft_tau = float(np.clip(target_soft_tau, 0.0, 1.0))
        self.memory = ReplayBuffer(replay_capacity)
        self.model = self.build_model(learning_rate)
        self.target_model = self._build_target_model()
        self.train_steps = 0
        self.last_loss = None

    def _build_target_model(self):
        target_model = clone_model(self.model)
        target_model.set_weights(self.model.get_weights())
        return target_model

    def update_target_network(self, force=False):
        """
        Synchronize online-network weights into target network.
        - Hard update when target_soft_tau=1.0.
        - Polyak averaging when target_soft_tau is in (0, 1).
        """
        if not force and (self.train_steps % self.target_update_every != 0):
            return

        online_weights = self.model.get_weights()
        if self.target_soft_tau >= 1.0:
            self.target_model.set_weights(online_weights)
            return

        target_weights = self.target_model.get_weights()
        tau = self.target_soft_tau
        mixed_weights = [
            tau * online_w + (1.0 - tau) * target_w
            for online_w, target_w in zip(online_weights, target_weights)
        ]
        self.target_model.set_weights(mixed_weights)

    def build_model(self, learning_rate):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))      #May increase Dense for bigger network
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=Huber(delta=1.0), optimizer=Adam(learning_rate = learning_rate))
        return model
    
    def select_action(self, state, valid_actions):
        """
        Select an action with epsilon-greedy exploration
        :param valid_actions: List of valid actions at a specific edge
        """
        if not valid_actions:
            return None
        if np.random.rand() <= self.epsilon: # Random to see if the agent should choose a new path
            return random.choice(valid_actions)
        q_values = self.model(state, training=False).numpy()[0]
        masked_values = np.full_like(q_values, -1e9)    #Make all q-values -1e9, then valid actions will update their according value, invalid actions will not be updated and stay negative
        for action in valid_actions:
            masked_values[action] = q_values[action]
        return int(np.argmax(masked_values))
    
    def remember(self, state, action, reward, next_state, done, next_valid_actions=None, metadata=None):
        """
        Store 1 transition for replay
        """
        self.memory.add(state, action, reward, next_state, done, next_valid_actions, metadata=metadata)
    
    def replay(self):
        """
        Train the Q-network from replayed experiences. Update q-values of previous state based on the most recent one.
        """
        if len(self.memory) < self.replay_warmup:
            return
        minibatch = self.memory.sample(self.batch_size)
        states      = np.vstack([s[0] for s in minibatch])
        actions     = np.array([s[1] for s in minibatch], dtype=np.int32)
        rewards     = np.array([s[2] for s in minibatch], dtype=np.float32)
        next_states = np.vstack([s[3] for s in minibatch])
        dones       = np.array([s[4] for s in minibatch], dtype=np.bool_)
        next_valid_actions_batch = [s[5] for s in minibatch]
        batch_len = states.shape[0]

        # NOTE: This keeps Double-DQN's two-network setup intact.
        # We only fuse ONLINE model inference calls (q(s), q_online(s'))
        # for speed. Target values are still evaluated by TARGET model.
        # So this is not switching from two networks to one network.
        stacked_states = np.vstack((states, next_states))
        q_all_online = self.model(stacked_states, training=False).numpy()
        q = q_all_online[:batch_len]
        q_next_online = q_all_online[batch_len:]
        q_next_target = self.target_model(next_states, training=False).numpy()

        valid_action_mask = np.zeros((batch_len, self.action_size), dtype=np.bool_)
        for idx, valid_actions in enumerate(next_valid_actions_batch):
            if dones[idx] or not valid_actions:
                continue
            valid_action_mask[idx, valid_actions] = True

        masked_online = np.where(valid_action_mask, q_next_online, -1e9)
        best_next_actions = np.argmax(masked_online, axis=1)
        bootstrap_values = q_next_target[np.arange(batch_len), best_next_actions]
        bootstrap_values[~valid_action_mask.any(axis=1)] = 0.0

        target = q.copy()
        target[np.arange(batch_len), actions] = (
            rewards + (1.0 - dones.astype(np.float32)) * self.gamma * bootstrap_values
        )

        loss = self.model.train_on_batch(states, target)
        self.last_loss = float(loss) if loss is not None else None
        self.train_steps += 1
        self.update_target_network()

        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
            
class RLTrainingPipeline:
    """
    Pipeline for training a routing policy with Deep Q-Learning.
    """

    def __init__(
        self,
        sumocfg_path,
        model_output_path,
        episodes=10,
        spawn_interval=4.0,
        seed_with_episode=True,
        decision_horizon=6,
        destination_reward=24.0,
        deadline_penalty=20.0,
        on_time_arrival_bonus=36.0,
        teleport_penalty=-20.0,
        epsilon_decay=0.99,
        epsilon_min=0.10,
        gamma=0.97,
        replay_capacity=2000,
        batch_size=64,
        replay_warmup=1000,
        train_every=40,
        grad_steps=1,
        rolling_window=100,
        target_pattern=2,
        debug_exit_diagnostics=False,
        debug_exit_diagnostics_limit=20,
        step_log_every=100,
    ):
        """
        Args:
            sumocfg_path: SUMO config file path.
            model_output_path: Path to save the trained model.
            episodes: Number of training episodes.
            spawn_interval: Interval between vehicle spawns.
            seed_with_episode: Whether to use the episode number as random seed.
            decision_horizon: Number of actions to pad a decision list.
            destination_reward: Reward when reaching the destination.
            deadline_penalty: Penalty when missing the deadline.
            on_time_arrival_bonus: Extra reward for arriving before deadline.
            teleport_penalty: Terminal penalty for teleport events.
            target_pattern: Vehicle generation pattern. 2 means varied origins
                and one shared destination (helps controlled deadline comparison).
        """
        self.sumocfg_path = sumocfg_path
        self.model_output_path = model_output_path
        self.episodes = episodes
        self.spawn_interval = spawn_interval
        self.seed_with_episode = seed_with_episode
        self.decision_horizon = decision_horizon
        self.destination_reward = destination_reward
        self.deadline_penalty = deadline_penalty
        self.on_time_arrival_bonus = on_time_arrival_bonus
        self.teleport_penalty = teleport_penalty
        self.train_every = train_every
        self.grad_steps = grad_steps
        self.rolling_window = rolling_window
        self.target_pattern = target_pattern
        self.debug_exit_diagnostics = debug_exit_diagnostics
        self.debug_exit_diagnostics_limit = max(int(debug_exit_diagnostics_limit), 0)
        self.step_log_every = max(int(step_log_every), 1)
        self._distance_cache = {}
        self.progress_reward_scale = 0.80  # or 0.0 to disable progress term cheaply
        self.system_congestion_scale = 0.03  # scales marginal congestion penalty on busy edges
        self.loop_window = 12
        self.loop_repeat_penalty = 1.5
        # Objective priority:
        # 1) deadline feasibility (dominant)
        # 2) congestion externality (secondary)
        # 3) shortest-path distance only as weak tie-breaker
        self.deadline_deficit_scale = 1.5
        self.deadline_deficit_delta_scale = 0.8
        self.deadline_critical_buffer = 20.0
        self.distance_tiebreak_scale = 0.02
        self.reward_clip_low = -20.0
        self.reward_clip_high = 20.0

        self.sumocfg_dir = os.path.dirname(sumocfg_path)
        self.net_file, self.route_file = self.parse_sumocfg(sumocfg_path)

        self.net = sumolib.net.readNet(os.path.join(self.sumocfg_dir, self.net_file))

        self.connection_info = ConnectionInfo(os.path.join(self.sumocfg_dir, self.net_file))
        self.route_helper = TrainingRouteHelper(self.connection_info)
        self.decision_engine = JunctionDecisionEngine(
            self.connection_info,
            self.net,
            self.route_helper.direction_choices,
        )

        # state = [edge_embedding, destination_embedding]
        #         + edge/lane/reachable/available feasibility masks (4*6)
        #         + commit flag + 3 lane features + 3 deadline/time features
        #         + local congestion summary
        self.edge_embedding_dim = 8
        self.local_congestion_k = 6
        self._init_edge_embeddings(seed=1337)
        self.state_size = (2 * self.edge_embedding_dim) + 24 + 1 + 3 + 3 + self.local_congestion_k
        self.action_size = 6
        self.metrics_csv_path = os.path.join(self.sumocfg_dir, "rl_episode_metrics.csv")
        self.trainer = DQNTrainer(
            self.state_size,
            self.action_size,
            gamma=gamma,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            replay_warmup=replay_warmup,
            target_update_every=200,
            target_soft_tau=1.0,
        )

    def _print_step_progress(
        self,
        episode,
        step,
        total_controlled,
        arrived_ids,
        arrived_before_deadline_ids,
        decision_metrics,
    ):
        completion = (len(arrived_ids) / float(total_controlled)) if total_controlled > 0 else 0.0
        on_time = (len(arrived_before_deadline_ids) / float(total_controlled)) if total_controlled > 0 else 0.0
        failed = max(total_controlled - len(arrived_ids), 0)
        print(
            "[EP {:03d} | STEP {:04d}] eps={:.3f} replay={} train={} loss={} "
            "done={}/{} ontime={}/{} fail={} open/final/skip={:.0f}/{:.0f}/{:.0f} forced={:.0f}".format(
                episode,
                step,
                self.trainer.epsilon,
                len(self.trainer.memory),
                self.trainer.train_steps,
                "n/a" if self.trainer.last_loss is None else f"{self.trainer.last_loss:.4f}",
                len(arrived_ids),
                total_controlled,
                len(arrived_before_deadline_ids),
                total_controlled,
                failed,
                decision_metrics["decisions_opened"],
                decision_metrics["decisions_finalized"],
                decision_metrics["decisions_skipped"],
                decision_metrics["forced_actions"],
            )
        )
        print(
            "  rates: completion={:.1%} on_time={:.1%} override={:.1%} mismatch={} teleports={}".format(
                completion,
                on_time,
                decision_metrics["safety_overrides"] / max(decision_metrics["decisions_opened"], 1.0),
                int(decision_metrics["route_mismatch"]),
                int(decision_metrics["teleports"]),
            )
        )

    def _init_edge_embeddings(self, seed=1337):
        """
        Fixed edge embeddings avoid fake ordinal structure from raw edge indices.
        """
        rng = np.random.default_rng(seed)
        self._edge_embeddings = {}
        for edge_id in self.connection_info.edge_list:
            emb = rng.normal(loc=0.0, scale=0.1, size=self.edge_embedding_dim).astype(np.float32)
            self._edge_embeddings[edge_id] = emb

    def _get_edge_embedding(self, edge_id):
        return self._edge_embeddings.get(
            edge_id,
            np.zeros(self.edge_embedding_dim, dtype=np.float32),
        )

    def _local_congestion_features(self, edge_id):
        """
        Compact congestion summary around current edge to reduce input noise.
        """
        counts = self.connection_info.edge_vehicle_count
        lengths = self.connection_info.edge_length_dict

        current_density = counts.get(edge_id, 0) / max(lengths.get(edge_id, 5.0), 5.0)
        outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {})
        outgoing_densities = [
            counts.get(next_edge, 0) / max(lengths.get(next_edge, 5.0), 5.0)
            for next_edge in outgoing.values()
        ]

        mean_out = float(np.mean(outgoing_densities)) if outgoing_densities else current_density
        max_out = float(np.max(outgoing_densities)) if outgoing_densities else current_density
        min_out = float(np.min(outgoing_densities)) if outgoing_densities else current_density
        mean_global = float(np.mean(self._density_vec)) if len(self._density_vec) > 0 else 0.0
        std_global = float(np.std(self._density_vec)) if len(self._density_vec) > 0 else 0.0

        return np.array(
            [
                current_density,
                mean_out,
                max_out,
                min_out,
                current_density - mean_global,
                std_global,
            ],
            dtype=np.float32,
        )

    def _deadline_window(self, vehicle):
        """
        Return a strictly positive scheduling window based on
        (deadline - start_time).
        """
        return max(float(vehicle.deadline) - float(vehicle.start_time), 1.0)

    def _deadline_urgency(self, vehicle, step):
        """
        Convert deadline flexibility into an urgency score in [0, 1].

        Vehicles with smaller (deadline - start_time) or little time left
        have higher urgency and should be prioritized.
        """
        deadline_window = self._deadline_window(vehicle)
        time_left = max(float(vehicle.deadline) - float(step), 0.0)
        # 1.0 means no slack left, 0.0 means fully relaxed.
        return 1.0 - min(time_left / deadline_window, 1.0)

    def parse_sumocfg(self, sumocfg_path):
        """
        Parse the SUMO config file and return net and route filenames.
        """
        dom = parse(sumocfg_path)
        net_file_node = dom.getElementsByTagName('net-file')
        route_file_node = dom.getElementsByTagName('route-files')
        net_file = net_file_node[0].attributes['value'].nodeValue
        route_file = route_file_node[0].attributes['value'].nodeValue
        return net_file, route_file

    def encode_state(self, vehicle_id, edge_id, destination_edge, context=None, vehicle=None, step=None):
        """
        Build a state vector for the given edge using cached per-step densities.
        """
        state = np.zeros(self.state_size, dtype=np.float32)
        state[0:self.edge_embedding_dim] = self._get_edge_embedding(edge_id)
        state[self.edge_embedding_dim:(2 * self.edge_embedding_dim)] = self._get_edge_embedding(destination_edge)

        if context is None:
            context = self.decision_engine.build_context(vehicle_id, edge_id, destination_edge, int(step or 0))
        edge_mask, lane_mask, reach_mask, avail_mask = self.decision_engine.direction_masks(context)
        base = 2 * self.edge_embedding_dim
        state[base:base + 6] = np.array(edge_mask, dtype=np.float32)
        state[base + 6:base + 12] = np.array(lane_mask, dtype=np.float32)
        state[base + 12:base + 18] = np.array(reach_mask, dtype=np.float32)
        state[base + 18:base + 24] = np.array(avail_mask, dtype=np.float32)
        state[base + 24] = 1.0 if context.commit_window else 0.0

        # lane features
        lane_base = base + 25
        state[lane_base + 0] = context.lane_index / max(context.lane_count - 1, 1)
        state[lane_base + 1] = min(context.lane_count, 6) / 6.0
        dist_to_end = context.dist_to_end
        state[lane_base + 2] = min(dist_to_end, 200.0) / 200.0

        # deadline/time features (normalized)
        deadline_base = lane_base + 3
        if vehicle is not None:
            if step is None:
                step = traci.simulation.getTime()
            deadline_window = self._deadline_window(vehicle)
            time_left = max(float(vehicle.deadline) - float(step), 0.0)
            elapsed = max(float(step) - float(vehicle.start_time), 0.0)
            urgency = self._deadline_urgency(vehicle, step)

            state[deadline_base + 0] = min(time_left / deadline_window, 1.0)
            state[deadline_base + 1] = min(elapsed / deadline_window, 1.0)
            state[deadline_base + 2] = urgency

        state[deadline_base + 3:] = self._local_congestion_features(edge_id)
        return state.reshape(1, -1)

    def valid_actions(self, edge_id):
        """
        Return action indices that are valid from the current edge.
        """
        valid = []
        for idx, choice in enumerate(self.route_helper.direction_choices):
            if choice in self.connection_info.outgoing_edges_dict[edge_id]:
                valid.append(idx)
        return valid
    
    def valid_actions_for_vehicle(self, vehicle_id, edge_id, destination_edge, step):
        context = self.decision_engine.build_context(vehicle_id, edge_id, destination_edge, step)
        return context.available_actions
    
    def dist_to_end(self, vehicle_id):
        """
        Distance (meters) from the vehicle to the end of its current lane.
        """
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        lane_len = traci.lane.getLength(lane_id)
        lane_pos = traci.vehicle.getLanePosition(vehicle_id)
        return max(lane_len - lane_pos, 0.0)

    def is_decision_point(self, edge_id, vehicle_id, dist_threshold=80.0):
        """
        Make routing decisions only when it matters:
        - the edge has > 1 outgoing option (real branch), AND
        - the vehicle is close enough to the junction (within dist_threshold meters)
        """
        outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {})
        if outgoing is None or len(outgoing) <= 1:
            return False
        return self.dist_to_end(vehicle_id) <= dist_threshold
    
    def adaptive_dist_threshold(self, edge_id, max_dist=200.0, ratio=0.6, min_dist=30.0):
        """
        Adaptive threshold: decide when within min(max_dist, ratio * edge_length),
        clamped to at least min_dist.
        """
        edge_len = self.connection_info.edge_length_dict.get(edge_id, None)
        if edge_len is None:
            return max_dist
        return max(min_dist, min(max_dist, ratio * float(edge_len)))

    def is_decision_point_adaptive(self, edge_id, vehicle_id, max_dist=200.0, ratio=0.6, min_dist=30.0):
        """
        Decide near junctions, but adapt threshold based on edge length so short edges are safe.
        """
        outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {})
        if not outgoing or len(outgoing) <= 1:
            return False

        dist_th = self.adaptive_dist_threshold(edge_id, max_dist=max_dist, ratio=ratio, min_dist=min_dist)
        return self.dist_to_end(vehicle_id) <= dist_th

    # =========================
    # Teleport detection helpers
    # =========================
    def get_teleport_ids(self):
        """
        Return a set of vehicle IDs that teleported this step.

        SUMO/TraCI API differs by version, so we try multiple methods.
        """
        teleported = set()

        # Most common in many SUMO versions:
        try:
            teleported.update(traci.simulation.getStartingTeleportIDList())
        except Exception:
            pass
        try:
            teleported.update(traci.simulation.getEndingTeleportIDList())
        except Exception:
            pass

        # Some versions expose a vehicle-level list:
        try:
            teleported.update(traci.vehicle.getTeleportingList())
        except Exception:
            pass

        return teleported

    def make_terminal_next_state(self, vehicle_id, edge_id, destination_edge):
        """
        Build a next_state even if the vehicle is in a weird edge after teleport.
        If edge_id is unknown, fall back to a zero state (safe for training).
        """
        try:
            if edge_id in self.connection_info.edge_index_dict and destination_edge in self.connection_info.edge_index_dict:
                return self.encode_state(vehicle_id, edge_id, destination_edge)
        except Exception:
            pass
        return np.zeros((1, self.state_size), dtype=np.float32)

    def _score_next_edge(self, next_edge, destination_edge, recent_edges, direction, vehicle, step):
        """
        Lower score is better with lexicographic-style priorities:
        1) deadline feasibility deficit
        2) congestion externality
        3) distance tie-breaker
        """
        distance = self.get_distance_to_destination(next_edge, destination_edge)
        if not math.isfinite(distance):
            return math.inf

        time_left = max(float(vehicle.deadline) - float(step), 0.0)
        eta = self._estimate_remaining_eta(next_edge, destination_edge)
        deadline_deficit = max(eta - time_left, 0.0) if math.isfinite(eta) else 9999.0

        edge_count = self.connection_info.edge_vehicle_count.get(next_edge, 0)
        edge_len = max(self.connection_info.edge_length_dict.get(next_edge, 5.0), 5.0)
        edge_density = edge_count / edge_len
        mean_density = float(np.mean(self._density_vec)) if len(self._density_vec) > 0 else 0.0
        marginal_pressure = max(edge_density - mean_density, 0.0)

        recent_penalty = 0.0
        if recent_edges:
            recent_penalty += 40.0 * sum(1 for e in recent_edges if e == next_edge)

        turnaround_penalty = 25.0 if direction == 't' else 0.0
        return (
            (self.deadline_deficit_scale * deadline_deficit)
            + (self.system_congestion_scale * 100.0 * marginal_pressure)
            + (self.distance_tiebreak_scale * float(distance))
            + recent_penalty
            + turnaround_penalty
        )

    def _estimate_remaining_eta(self, edge_id, destination_edge):
        """
        Estimate travel time from edge_id to destination using shortest-path
        distance and a conservative minimum speed floor.
        """
        distance = self.get_distance_to_destination(edge_id, destination_edge)
        if not math.isfinite(distance):
            return math.inf
        return float(distance) / 8.0

    def build_decision_list(self, edge_id, initial_action, vehicle, recent_edges, sim_step):
        """
        Build a decision list with RL-driven control.
        We intentionally avoid heuristic horizon takeover during training.
        """
        decision_list = []
        current_edge = edge_id

        for _ in range(1):
            outgoing = self.connection_info.outgoing_edges_dict.get(current_edge, {})
            if not outgoing:
                break

            action = initial_action
            valid_actions = self.valid_actions(current_edge)
            if action not in valid_actions:
                break
            direction = self.route_helper.direction_choices[action]

            if direction not in outgoing:
                break

            decision_list.append(direction)
            current_edge = outgoing[direction]

            if current_edge == vehicle.destination:
                break

        return decision_list
    

    def get_distance_to_destination(self, edge_id, destination_edge):
        """
        Return shortest-path cost from edge_id to destination_edge.
        Uses a cache because this is called frequently during training.
        """
        key = (edge_id, destination_edge)
        if key in self._distance_cache:
            return self._distance_cache[key]

        try:
            from_edge = self.net.getEdge(edge_id)
            to_edge = self.net.getEdge(destination_edge)
        except Exception:
            self._distance_cache[key] = math.inf
            return math.inf

        path_edges, path_cost = self.net.getShortestPath(from_edge, to_edge, vClass="passenger")
        distance = path_cost if path_edges is not None else math.inf
        self._distance_cache[key] = distance
        return distance
    
    def ensure_lane_for_direction(self, vehicle_id, edge_id, direction, min_dist=40.0, duration=50):
        """
        If current lane cannot do 'direction', try to change into a lane that can,
        as long as we aren't too close to the junction end.
        """
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        lane_pos = traci.vehicle.getLanePosition(vehicle_id)
        lane_len = traci.lane.getLength(lane_id)
        dist_to_end = lane_len - lane_pos

        if dist_to_end < min_dist:
            return  # too late

        lane_ids = self.connection_info.edge_lane_ids.get(edge_id, [])
        for target_lane_index, ln_id in enumerate(lane_ids):
            lane_map = self.connection_info.lane_outgoing_edges_dict.get(ln_id, {})
            if direction in lane_map:
                curr_idx = traci.vehicle.getLaneIndex(vehicle_id)
                if curr_idx != target_lane_index:
                    traci.vehicle.changeLane(vehicle_id, target_lane_index, duration)
                return


    def compute_reward(
        self,
        vehicle,
        prev_edge,
        current_edge,
        step,
        arrived,
        repeated_recent_edges=0,
        delta_t=1.0,
        reached_global_destination=False,
        route_mismatch=False,
        invalid_late_turn=False,
        route_apply_failed=False,
        uturn_repeat=False,
        externality_penalty=0.0,
    ):
        """
        Compute a bounded reward with clear objective priority:
        1) arrive at true destination before deadline
        2) avoid deadline misses / unreachable states
        3) prefer feasible progress with mild congestion awareness.
        """
        deadline_window = self._deadline_window(vehicle)
        urgency = self._deadline_urgency(vehicle, step)
        time_left = max(float(vehicle.deadline) - float(step), 0.0)
        elapsed = max(float(delta_t), 1.0)

        prev_distance = self.get_distance_to_destination(prev_edge, vehicle.destination)
        curr_distance = self.get_distance_to_destination(current_edge, vehicle.destination)
        prev_eta = self._estimate_remaining_eta(prev_edge, vehicle.destination)
        curr_eta = self._estimate_remaining_eta(current_edge, vehicle.destination)

        reward = 0.0
        done = False

        # Dense shaping: small living/time and congestion costs.
        reward -= 0.15 * elapsed
        congestion = self.connection_info.edge_vehicle_count.get(current_edge, 0)
        edge_len = max(self.connection_info.edge_length_dict.get(current_edge, 5.0), 5.0)
        edge_density = congestion / edge_len
        reward -= 0.05 * edge_density * elapsed
        reward -= float(np.clip(externality_penalty, 0.0, 4.0))

        mean_density = float(np.mean(self._density_vec)) if len(self._density_vec) > 0 else 0.0
        marginal_pressure = max(edge_density - mean_density, 0.0)
        reward -= self.system_congestion_scale * marginal_pressure * elapsed

        # Primary shaping: deadline deficit and improvement.
        prev_deficit = max(prev_eta - (time_left + 1.0), 0.0) if math.isfinite(prev_eta) else 6.0
        curr_deficit = max(curr_eta - time_left, 0.0) if math.isfinite(curr_eta) else 6.0
        reward -= self.deadline_deficit_scale * min(curr_deficit, 8.0)
        reward += self.deadline_deficit_delta_scale * np.clip(prev_deficit - curr_deficit, -3.0, 3.0)

        if (time_left < self.deadline_critical_buffer) and (curr_deficit > 0.0):
            critical_scale = 1.0 + 0.5 * urgency
            reward -= critical_scale * min(curr_deficit, 3.0)

        # Tertiary tie-breaker: shortest-path distance progress.
        if math.isfinite(prev_distance) and math.isfinite(curr_distance):
            progress = (prev_distance - curr_distance) * (self.distance_tiebreak_scale * self.progress_reward_scale)
            reward += float(np.clip(progress, -1.0, 1.0))

        # Safety and control quality penalties.
        if repeated_recent_edges > 0:
            reward -= self.loop_repeat_penalty * min(repeated_recent_edges, 3)
        if uturn_repeat:
            reward -= 3.0
        if route_mismatch:
            reward -= 4.0
        if invalid_late_turn:
            reward -= 2.0
        if route_apply_failed:
            reward -= 6.0

        # Unreachable transition after a decision is strongly terminal-negative.
        if math.isfinite(prev_distance) and not math.isfinite(curr_distance):
            reward -= 12.0
            done = True

        if arrived:
            if reached_global_destination:
                reward += self.destination_reward
                if step <= vehicle.deadline:
                    reward += self.on_time_arrival_bonus
                else:
                    reward -= 0.75 * self.deadline_penalty
            else:
                reward -= 0.4 * self.deadline_penalty
            done = True
            return self._clip_reward(reward), done

        outgoing = self.connection_info.outgoing_edges_dict.get(current_edge, {})
        if (not outgoing or len(outgoing) == 0) and current_edge != vehicle.destination:
            reward -= 12.0
            done = True

        if step > vehicle.deadline:
            reward -= self.deadline_penalty
            done = True
        else:
            remaining_ratio = max(float(vehicle.deadline) - float(step), 0.0) / deadline_window
            if remaining_ratio < 0.25:
                reward -= (0.25 - remaining_ratio) * 2.0

        return self._clip_reward(reward), done

    def _clip_reward(self, reward_value):
        return float(np.clip(reward_value, self.reward_clip_low, self.reward_clip_high))

    def generate_episode_vehicles(self, episode_seed=None):
        """
        Generate controlled and uncontrolled vehicles for one training episode.
        """
        generator = target_vehicles_generator(os.path.join(self.sumocfg_dir, self.net_file))
        route_path = os.path.join(self.sumocfg_dir, self.route_file)
        vehicle_list = generator.generate_vehicles(
            num_target_vehicles=20,
            num_random_vehicles=30,
            pattern=self.target_pattern,
            target_xml_file=route_path,
            net_xml_file=os.path.join(self.sumocfg_dir, self.net_file),
            spawn_interval=self.spawn_interval,
            seed=episode_seed,
        )
        if vehicle_list is None:
            raise RuntimeError(
                "Failed to generate vehicles. Check randomTrips.py output for errors."
            )
        return {str(vehicle.vehicle_id): vehicle for vehicle in vehicle_list}

    def run(self):
        """Run RL training with shared junction-aware feasibility + pending-decision finalization."""
        sumo_binary = checkBinary('sumo')
        rolling_teleport_events = deque(maxlen=self.rolling_window)
        rolling_teleported_controlled = deque(maxlen=self.rolling_window)
        rolling_completion_rate = deque(maxlen=self.rolling_window)
        rolling_avg_return = deque(maxlen=self.rolling_window)
        rolling_mismatch = deque(maxlen=self.rolling_window)
        csv_fields = [
            "episode", "epsilon", "replay", "train_steps", "mean_loss", "episode_return",
            "completion_rate", "on_time_rate", "avg_tardiness", "teleports", "teleported_controlled",
            "forced_actions", "decisions_opened", "decisions_finalized", "decisions_skipped",
            "route_mismatch", "loop_events", "uturn_events", "safety_overrides", "override_ratio"
        ]
        if not os.path.exists(self.metrics_csv_path):
            with open(self.metrics_csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=csv_fields).writeheader()

        # MAX_CACHE_SIZE = 5000

        for episode in range(self.episodes):
            episode_seed = episode if self.seed_with_episode else None
            if episode_seed is not None:
                random.seed(episode_seed)
                np.random.seed(episode_seed)

            # if len(self._distance_cache) > MAX_CACHE_SIZE:
            #     self._distance_cache.clear()

            vehicles = self.generate_episode_vehicles(episode_seed=episode_seed)

            traci.start([
                sumo_binary,
                "-c", self.sumocfg_path,
                "--tripinfo-output", os.path.join(self.sumocfg_dir, "trips.trips.xml"),
                "--quit-on-end",
            ])

            pending_decisions = {}
            recent_edge_history = defaultdict(lambda: deque(maxlen=self.loop_window))
            prev_edge_by_vehicle = {}
            decision_metrics = defaultdict(float)

            episode_return = 0.0
            episode_teleport_events = 0
            teleported_controlled_ids = set()
            arrived_ids = set()
            arrived_before_deadline_ids = set()
            arrived_global_destination_ids = set()
            exited_without_destination_ids = set()
            total_controlled = len(vehicles)
            controlled_ids = set(vehicles.keys())
            last_seen_edge_by_vehicle = {}
            last_target_by_vehicle = {}
            tardiness_values = []
            arrived_debug_records = []

            try:
                for step in range(MAX_SIMULATION_STEPS):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        break

                    # Keep density features fresh for routing choices and reward.
                    self.update_edge_vehicle_counts(step, every=1)
                    vehicle_ids = list(traci.vehicle.getIDList())

                    for vehicle_id in vehicle_ids:
                        if vehicle_id not in vehicles:
                            continue

                        current_edge = traci.vehicle.getRoadID(vehicle_id)
                        if current_edge not in self.connection_info.edge_index_dict:
                            continue

                        vehicle = vehicles[vehicle_id]
                        vehicle.current_edge = current_edge
                        vehicle.current_speed = traci.vehicle.getSpeed(vehicle_id)
                        last_seen_edge_by_vehicle[vehicle_id] = current_edge
                        recent_edge_history[vehicle_id].append(current_edge)

                        # arrived
                        if current_edge == vehicle.destination:
                            if vehicle_id not in arrived_ids:
                                arrived_ids.add(vehicle_id)
                                if step <= vehicle.deadline:
                                    arrived_before_deadline_ids.add(vehicle_id)
                            pending_decisions.pop(vehicle_id, None)
                            prev_edge_by_vehicle.pop(vehicle_id, None)
                            continue

                        prev_edge = prev_edge_by_vehicle.get(vehicle_id)
                        if vehicle_id in pending_decisions and prev_edge is not None and prev_edge != current_edge:
                            pending = pending_decisions.pop(vehicle_id)
                            repeated_recent_edges = sum(1 for e in recent_edge_history[vehicle_id] if e == current_edge)
                            mismatch = not self.decision_engine.route_matches_expected(pending, current_edge)
                            if mismatch:
                                decision_metrics["route_mismatch"] += 1
                            uturn_repeat = len(recent_edge_history[vehicle_id]) >= 2 and recent_edge_history[vehicle_id][-2] == current_edge
                            if uturn_repeat:
                                decision_metrics["uturn_events"] += 1
                            ext_pen = max(self.connection_info.edge_vehicle_count.get(current_edge, 0) / max(self.connection_info.edge_length_dict.get(current_edge, 10.0), 10.0), 0.0)
                            reward, done = self.compute_reward(
                                vehicle, pending.decision_edge, current_edge, step, arrived=False,
                                repeated_recent_edges=repeated_recent_edges,
                                delta_t=max(step - pending.decision_step, 1),
                                route_mismatch=mismatch,
                                uturn_repeat=uturn_repeat,
                                externality_penalty=ext_pen,
                            )
                            next_ctx = self.decision_engine.build_context(vehicle_id, current_edge, vehicle.destination, step)
                            next_state = self.encode_state(vehicle_id, current_edge, vehicle.destination, context=next_ctx, vehicle=vehicle, step=step)
                            self.trainer.remember(
                                pending.state,
                                pending.intended_action,
                                reward,
                                next_state,
                                done,
                                next_valid_actions=next_ctx.available_actions,
                                metadata={"forced": pending.context.forced_action is not None, "mismatch": mismatch},
                            )
                            decision_metrics["decisions_finalized"] += 1
                            episode_return += reward
                            if repeated_recent_edges > 1:
                                decision_metrics["loop_events"] += 1

                        context = self.decision_engine.build_context(vehicle_id, current_edge, vehicle.destination, step)
                        if context.skip_reason:
                            decision_metrics["decisions_skipped"] += 1
                            if context.forced_action is not None:
                                decision_metrics["forced_actions"] += 1
                            prev_edge_by_vehicle[vehicle_id] = current_edge
                            continue
                        if not self.decision_engine.is_decision_open(context):
                            prev_edge_by_vehicle[vehicle_id] = current_edge
                            continue

                        state = self.encode_state(vehicle_id, current_edge, vehicle.destination, context=context, vehicle=vehicle, step=step)
                        action = self.trainer.select_action(state, context.available_actions)
                        if action is None:
                            decision_metrics["decisions_skipped"] += 1
                            prev_edge_by_vehicle[vehicle_id] = current_edge
                            continue

                        next_edge = self.decision_engine.get_next_edge(current_edge, action)
                        if next_edge is None:
                            decision_metrics["safety_overrides"] += 1
                            prev_edge_by_vehicle[vehicle_id] = current_edge
                            continue

                        lane_change_requested, lane_change_ok = self.decision_engine.try_request_lane_change(context, action)
                        if lane_change_requested:
                            decision_metrics["lane_change_attempts"] += 1
                            if lane_change_ok:
                                decision_metrics["lane_change_success"] += 1
                            else:
                                decision_metrics["lane_change_fail"] += 1

                        fragment, local_target, frag_error = self.decision_engine.build_route_fragment(current_edge, action, vehicle.destination)
                        if frag_error or local_target is None:
                            decision_metrics["route_apply_fail"] += 1
                            prev_edge_by_vehicle[vehicle_id] = current_edge
                            continue

                        try:
                            traci.vehicle.setVia(vehicle_id, fragment[:3] if local_target != vehicle.destination else [])
                            traci.vehicle.changeTarget(vehicle_id, vehicle.destination)
                            last_target_by_vehicle[vehicle_id] = local_target
                        except traci.exceptions.TraCIException:
                            decision_metrics["route_apply_fail"] += 1
                            prev_edge_by_vehicle[vehicle_id] = current_edge
                            continue

                        pending_decisions[vehicle_id] = PendingDecision(
                            state=state,
                            intended_action=action,
                            intended_next_edge=next_edge,
                            decision_edge=current_edge,
                            decision_step=step,
                            destination=vehicle.destination,
                            context=context,
                            lane_change_requested=lane_change_requested,
                        )
                        decision_metrics["decisions_opened"] += 1
                        prev_edge_by_vehicle[vehicle_id] = current_edge

                    traci.simulationStep()

                    # Vehicles removed by SUMO because they reached their current route target.
                    # This is where we can tell whether they ended at the global destination
                    # or were terminated at an intermediate/local target.
                    arrived_this_step = traci.simulation.getArrivedIDList()
                    for arrived_vehicle_id in arrived_this_step:
                        if arrived_vehicle_id not in vehicles:
                            continue

                        vehicle = vehicles[arrived_vehicle_id]
                        arrived_ids.add(arrived_vehicle_id)

                        last_seen_edge = last_seen_edge_by_vehicle.get(arrived_vehicle_id, "<unknown>")
                        reached_global_destination = (last_seen_edge == vehicle.destination)

                        if reached_global_destination:
                            arrived_global_destination_ids.add(arrived_vehicle_id)
                            if step <= vehicle.deadline:
                                arrived_before_deadline_ids.add(arrived_vehicle_id)
                            tardiness_values.append(max(step - vehicle.deadline, 0.0))
                        else:
                            exited_without_destination_ids.add(arrived_vehicle_id)

                        debug_record = {
                            "vehicle_id": arrived_vehicle_id,
                            "global_destination": vehicle.destination,
                            "last_seen_edge": last_seen_edge,
                            "last_local_target": last_target_by_vehicle.get(arrived_vehicle_id, "<unset>"),
                            "reached_global_destination": reached_global_destination,
                        }
                        arrived_debug_records.append(debug_record)

                        # Close any open transition as a terminal arrival transition so
                        # destination reward / on-time bonus are learned explicitly.
                        if arrived_vehicle_id in pending_decisions:
                            pending = pending_decisions.pop(arrived_vehicle_id)
                            repeated_recent_edges = sum(
                                1 for edge in recent_edge_history[arrived_vehicle_id]
                                if edge == last_seen_edge
                            )
                            decision_delta_t = max(step - pending.decision_step, 1)
                            reward, done = self.compute_reward(
                                vehicle,
                                pending.decision_edge,
                                last_seen_edge,
                                step,
                                arrived=True,
                                repeated_recent_edges=repeated_recent_edges,
                                delta_t=decision_delta_t,
                                reached_global_destination=reached_global_destination,
                            )
                            next_state = self.make_terminal_next_state(
                                arrived_vehicle_id,
                                last_seen_edge,
                                vehicle.destination,
                            )
                            self.trainer.remember(
                                pending.state,
                                pending.intended_action,
                                reward,
                                next_state,
                                done,
                                next_valid_actions=[],
                            )
                            episode_return += reward

                        # No more transitions should be open once SUMO removes the vehicle.
                        prev_edge_by_vehicle.pop(arrived_vehicle_id, None)

                    # =========================
                    # Teleport detection + terminal penalty
                    # =========================
                    teleported_ids = self.get_teleport_ids()
                    if teleported_ids:
                        episode_teleport_events += len(teleported_ids)
                        decision_metrics["teleports"] += len(teleported_ids)
                        teleported_controlled_ids.update(tid for tid in teleported_ids if tid in vehicles)
                        for tid in list(teleported_ids):
                            if tid not in vehicles:
                                continue

                            # If we have an open transition for this vehicle, close it as terminal
                            if tid in pending_decisions:
                                pending = pending_decisions.pop(tid)
                                v = vehicles[tid]

                                # Try to get where it ended up; may fail if removed, so guard
                                try:
                                    tele_edge = traci.vehicle.getRoadID(tid)
                                except Exception:
                                    tele_edge = pending.decision_edge

                                # Big penalty so agent learns to avoid situations leading to teleports
                                next_state = self.make_terminal_next_state(tid, tele_edge, v.destination)

                                self.trainer.remember(
                                    pending.state,
                                    pending.intended_action,
                                    self.teleport_penalty,
                                    next_state,
                                    True,
                                    next_valid_actions=[],
                                )
                                episode_return += self.teleport_penalty

                            prev_edge_by_vehicle.pop(tid, None)

                    if step % self.train_every == 0:
                        for _ in range(self.grad_steps):
                            self.trainer.replay()

                    if step % self.step_log_every == 0:
                        self._print_step_progress(
                            episode=episode,
                            step=step,
                            total_controlled=total_controlled,
                            arrived_ids=arrived_ids,
                            arrived_before_deadline_ids=arrived_before_deadline_ids,
                            decision_metrics=decision_metrics,
                        )

                    # process = psutil.Process(os.getpid())

                    # if step % 100 == 0:
                    #     print(
                    #         "RAM_MB=", process.memory_info().rss / 1024 / 1024,
                    #         " replay=", len(self.trainer.memory),
                    #         " dist_cache=", len(self._distance_cache),
                    #     )

            finally:
                completion_rate = (
                    len(arrived_before_deadline_ids) / float(total_controlled)
                    if total_controlled > 0 else 0.0
                )
                on_time_rate = len(arrived_before_deadline_ids) / max(len(arrived_ids), 1)
                avg_tardiness = float(np.mean(tardiness_values)) if tardiness_values else 0.0
                avg_return = episode_return / float(total_controlled) if total_controlled > 0 else 0.0

                rolling_teleport_events.append(float(episode_teleport_events))
                rolling_teleported_controlled.append(float(len(teleported_controlled_ids)))
                rolling_completion_rate.append(float(completion_rate))
                rolling_avg_return.append(float(avg_return))
                rolling_mismatch.append(float(decision_metrics["route_mismatch"]))

                roll_tele_events = sum(rolling_teleport_events) / len(rolling_teleport_events)
                roll_tele_ctrl = sum(rolling_teleported_controlled) / len(rolling_teleported_controlled)
                roll_completion = sum(rolling_completion_rate) / len(rolling_completion_rate)
                roll_return = sum(rolling_avg_return) / len(rolling_avg_return)
                roll_mismatch = sum(rolling_mismatch) / len(rolling_mismatch)

                self.trainer.epsilon = max(
                    self.trainer.epsilon_min,
                    self.trainer.epsilon * self.trainer.epsilon_decay
                )
                print(
                    f"\n[EP {episode:03d} DONE] eps={self.trainer.epsilon:.4f} train={self.trainer.train_steps} "
                    f"replay={len(self.trainer.memory)} ret={avg_return:.3f} "
                    f"done={len(arrived_ids)}/{total_controlled} ontime={len(arrived_before_deadline_ids)}/{total_controlled} "
                    f"failed={max(total_controlled-len(arrived_ids),0)} avg_tardy={avg_tardiness:.2f}"
                )
                print(
                    f"  rolling({len(rolling_teleport_events)}): completion={roll_completion:.1%} "
                    f"avg_return={roll_return:.3f} teleports/ep={roll_tele_events:.2f} "
                    f"teleported_ctrl/ep={roll_tele_ctrl:.2f} mismatch/ep={roll_mismatch:.2f}"
                )
                print(
                    "  decisions: opened={:.0f} finalized={:.0f} skipped={:.0f} forced={:.0f} "
                    "lane_change(a/s/f)={:.0f}/{:.0f}/{:.0f} loops={:.0f} uturn={:.0f} "
                    "apply_fail={:.0f} overrides={:.0f} override_ratio={:.1%}".format(
                        decision_metrics["decisions_opened"],
                        decision_metrics["decisions_finalized"],
                        decision_metrics["decisions_skipped"],
                        decision_metrics["forced_actions"],
                        decision_metrics["lane_change_attempts"],
                        decision_metrics["lane_change_success"],
                        decision_metrics["lane_change_fail"],
                        decision_metrics["loop_events"],
                        decision_metrics["uturn_events"],
                        decision_metrics["route_apply_fail"],
                        decision_metrics["safety_overrides"],
                        decision_metrics["safety_overrides"] / max(decision_metrics["decisions_opened"], 1.0),
                    )
                )

                # Controlled vehicles that left simulation without being marked
                # as arrived (global destination) or teleported.
                exited_without_destination_ids.update(
                    controlled_ids - arrived_ids - teleported_controlled_ids
                )
                print(
                    f"Controlled exit diagnostics | "
                    f"arrived={len(arrived_global_destination_ids)}/{total_controlled}, "
                    f"arrived_any_target={len(arrived_ids)}/{total_controlled}, "
                    f"arrived_before_deadline={len(arrived_before_deadline_ids)}/{total_controlled}, "
                    f"teleported_controlled={len(teleported_controlled_ids)}/{total_controlled}, "
                    f"exited_without_destination={len(exited_without_destination_ids)}/{total_controlled}"
                )

                if self.debug_exit_diagnostics:
                    mismatched_arrivals = [
                        record for record in arrived_debug_records
                        if not record["reached_global_destination"]
                    ]
                    print(
                        f"Arrival debug | total_arrived={len(arrived_debug_records)}, "
                        f"arrived_at_non_global_target={len(mismatched_arrivals)}"
                    )

                    for record in mismatched_arrivals[:self.debug_exit_diagnostics_limit]:
                        print(
                            "  ARRIVED_NON_GLOBAL "
                            f"vehicle={record['vehicle_id']} "
                            f"last_seen_edge={record['last_seen_edge']} "
                            f"last_local_target={record['last_local_target']} "
                            f"global_destination={record['global_destination']}"
                        )

                    if len(mismatched_arrivals) > self.debug_exit_diagnostics_limit:
                        print(
                            "  ARRIVED_NON_GLOBAL ... "
                            f"{len(mismatched_arrivals) - self.debug_exit_diagnostics_limit} more vehicles"
                        )

                    for vehicle_id in sorted(exited_without_destination_ids)[:self.debug_exit_diagnostics_limit]:
                        vehicle = vehicles[vehicle_id]
                        print(
                            "  EXITED_WITHOUT_DEST "
                            f"vehicle={vehicle_id} "
                            f"last_seen_edge={last_seen_edge_by_vehicle.get(vehicle_id, '<unknown>')} "
                            f"last_local_target={last_target_by_vehicle.get(vehicle_id, '<unset>')} "
                            f"global_destination={vehicle.destination}"
                        )

                    if len(exited_without_destination_ids) > self.debug_exit_diagnostics_limit:
                        print(
                            "  EXITED_WITHOUT_DEST ... "
                            f"{len(exited_without_destination_ids) - self.debug_exit_diagnostics_limit} more vehicles"
                        )
                traci.close()
                with open(self.metrics_csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fields)
                    writer.writerow({
                        "episode": episode,
                        "epsilon": self.trainer.epsilon,
                        "replay": len(self.trainer.memory),
                        "train_steps": self.trainer.train_steps,
                        "mean_loss": self.trainer.last_loss if self.trainer.last_loss is not None else "",
                        "episode_return": avg_return,
                        "completion_rate": completion_rate,
                        "on_time_rate": on_time_rate,
                        "avg_tardiness": avg_tardiness,
                        "teleports": episode_teleport_events,
                        "teleported_controlled": len(teleported_controlled_ids),
                        "forced_actions": decision_metrics["forced_actions"],
                        "decisions_opened": decision_metrics["decisions_opened"],
                        "decisions_finalized": decision_metrics["decisions_finalized"],
                        "decisions_skipped": decision_metrics["decisions_skipped"],
                        "route_mismatch": decision_metrics["route_mismatch"],
                        "loop_events": decision_metrics["loop_events"],
                        "uturn_events": decision_metrics["uturn_events"],
                        "safety_overrides": decision_metrics["safety_overrides"],
                        "override_ratio": decision_metrics["safety_overrides"] / max(decision_metrics["decisions_opened"], 1.0),
                    })

        self.trainer.model.save(self.model_output_path)

    def update_edge_vehicle_counts(self, step, every=10):
        if hasattr(self, "_last_density_step") and (step - self._last_density_step) < every:
            return  # reuse cached self._density_vec

        counts = self.connection_info.edge_vehicle_count
        edge_list = self.connection_info.edge_list
        lengths = self.connection_info.edge_length_dict

        for edge in edge_list:
            counts[edge] = traci.edge.getLastStepVehicleNumber(edge)

        self._density_vec = np.array(
            [counts[e] / max(lengths[e], 1e-6) for e in edge_list],
            dtype=np.float32
        )
        self._last_density_step = step
