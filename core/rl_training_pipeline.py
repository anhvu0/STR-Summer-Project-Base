import numpy as np
import os
import sys
import math
import csv
import json

from xml.dom.minidom import parse
from keras.layers import Dense
from keras.models import Sequential, clone_model
from keras.losses import Huber
from keras.optimizers import Adam
from collections import defaultdict, deque
import random
from controller.RouteController import RouteController
from core.junction_decision_engine import JunctionDecisionEngine, PendingDecision, VehicleSnapshot
from core.Util import ConnectionInfo
from core.target_vehicles_generation_protocols import target_vehicles_generator
from core.route_loop_safety import transition_signal

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
        epsilon_min=0.08,
        replay_capacity=5000,
        batch_size=128,
        replay_warmup=1000,
        target_update_every=200,
        target_soft_tau=1.0,
        use_double_dqn=True,
    ):
        """
        :param learning_rate: Can be adjusted for further optimization
        :param gamma: Can be adjusted for further optimization
        :param epsilon: 1.0 allows free exploration
        :param epsilon_decay: epsilon value in next episode
        :param epsilon_min: minimum epsilon to ensure that there's still some chance for free exploration later
        :param use_double_dqn: If True, use online argmax + target evaluation for bootstrapping.
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
        self.use_double_dqn = bool(use_double_dqn)
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
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))      #May increase Dense for bigger network
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=Huber(delta=1.0), optimizer=Adam(learning_rate = learning_rate))
        return model
    
    def select_action(self, state, valid_actions, return_source=False):
        """
        Select an action with epsilon-greedy exploration
        :param valid_actions: List of valid actions at a specific edge
        """
        if not valid_actions:
            return (None, "none") if return_source else None
        if np.random.rand() <= self.epsilon: # Random to see if the agent should choose a new path
            action = random.choice(valid_actions)
            return (action, "explore") if return_source else action
        q_values = self.model(state, training=False).numpy()[0]
        masked_values = np.full_like(q_values, -1e9)    #Make all q-values -1e9, then valid actions will update their according value, invalid actions will not be updated and stay negative
        for action in valid_actions:
            masked_values[action] = q_values[action]
        selected = int(np.argmax(masked_values))
        return (selected, "policy") if return_source else selected

    def select_actions_batch(self, states, valid_actions_batch):
        """
        Batch epsilon-greedy selection for multiple vehicles in one model pass.
        Returns list of (action, source).
        """
        n = len(states)
        results = [(None, "none") for _ in range(n)]
        policy_indices = []
        policy_states = []
        for idx, (state, valid_actions) in enumerate(zip(states, valid_actions_batch)):
            if not valid_actions:
                continue
            if np.random.rand() <= self.epsilon:
                results[idx] = (random.choice(valid_actions), "explore")
            else:
                policy_indices.append(idx)
                policy_states.append(state[0])
        if policy_states:
            q_batch = self.model(np.array(policy_states, dtype=np.float32), training=False).numpy()
            for local_idx, global_idx in enumerate(policy_indices):
                valid_actions = valid_actions_batch[global_idx]
                masked_values = np.full_like(q_batch[local_idx], -1e9)
                masked_values[valid_actions] = q_batch[local_idx][valid_actions]
                results[global_idx] = (int(np.argmax(masked_values)), "policy")
        return results
    
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

        # Keep keras inference pattern fast:
        # - Fuse ONLINE model calls for q(s) and q_online(s') in one pass.
        # - Use TARGET model only for bootstrap values.
        # This preserves Double-DQN behavior when enabled.
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

        selection_q = q_next_online if self.use_double_dqn else q_next_target
        masked_selection_q = np.where(valid_action_mask, selection_q, -1e9)
        best_next_actions = np.argmax(masked_selection_q, axis=1)
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
        destination_reward=50.0,
        teleport_penalty=-40.0,
        epsilon_decay=0.99,
        epsilon_min=0.08,
        gamma=0.97,
        replay_capacity=5000,
        batch_size=128,
        replay_warmup=256,
        train_every=6,
        grad_steps=1,
        rolling_window=100,
        use_double_dqn=True,
        target_pattern=2,
        debug_exit_diagnostics=False,
        debug_exit_diagnostics_limit=20,
        step_log_every=100,
        density_refresh_every=5,
        normalize_per_step_cost_by_route_difficulty=False,
        route_difficulty_eta_floor=60.0,
        route_difficulty_scale_min=0.35,
        route_difficulty_scale_max=1.0,
        decision_debug_csv_path=None,
        fast_training_profile=False,
    ):
        """
        Args:
            sumocfg_path: SUMO config file path.
            model_output_path: Path to save the trained model.
            episodes: Number of training episodes.
            spawn_interval: Interval between vehicle spawns.
            seed_with_episode: Whether to use the episode number as random seed.
            destination_reward: Reward when reaching the destination.
            teleport_penalty: Terminal penalty for teleport events.
            use_double_dqn: Enable Double-DQN bootstrap action selection.
            target_pattern: Vehicle generation pattern. 2 means varied origins
                and one shared destination (helps controlled travel-time comparison).
            normalize_per_step_cost_by_route_difficulty: If True, scales only the
                per-step travel-time cost by estimated O-D ETA so very long routes
                are not structurally over-penalized.
        """
        self.sumocfg_path = sumocfg_path
        self.model_output_path = model_output_path
        self.episodes = episodes
        self.spawn_interval = spawn_interval
        self.seed_with_episode = seed_with_episode
        self.destination_reward = destination_reward
        self.teleport_penalty = teleport_penalty
        self.train_every = train_every
        self.grad_steps = grad_steps
        self.rolling_window = rolling_window
        self.target_pattern = target_pattern
        self.debug_exit_diagnostics = debug_exit_diagnostics
        self.debug_exit_diagnostics_limit = max(int(debug_exit_diagnostics_limit), 0)
        self.step_log_every = max(int(step_log_every), 1)
        self.density_refresh_every = max(int(density_refresh_every), 1)
        self.normalize_per_step_cost_by_route_difficulty = bool(normalize_per_step_cost_by_route_difficulty)
        self.route_difficulty_eta_floor = max(float(route_difficulty_eta_floor), 1.0)
        self.route_difficulty_scale_min = float(np.clip(route_difficulty_scale_min, 0.05, 1.0))
        self.route_difficulty_scale_max = float(np.clip(route_difficulty_scale_max, self.route_difficulty_scale_min, 1.0))
        self.fast_training_profile = bool(fast_training_profile)
        self.decision_debug_csv_path = decision_debug_csv_path
        if self.fast_training_profile and self.decision_debug_csv_path:
            self.decision_debug_csv_path = None
        if self.fast_training_profile:
            self.density_refresh_every = max(self.density_refresh_every, 4)
        self._distance_cache = {}
        self._cache_metrics = defaultdict(float)
        self.progress_reward_scale = 1.00
        self.system_congestion_scale = 0.015
        self.loop_window = 12
        self.loop_repeat_penalty = 1.5
        # Objective priority:
        # 1) minimize travel time (dominant)
        # 2) congestion externality (secondary)
        # 3) shortest-path distance as tie-breaker
        self.travel_time_penalty = 0.05
        self.eta_progress_scale = 0.65
        self.distance_tiebreak_scale = 0.06
        self.reward_clip_low = -20.0
        self.reward_clip_high = 20.0
        self.pending_timeout_penalty = -8.0
        self.pending_latency_penalty_per_step = 0.015
        self.pending_replan_penalty = -1.0
        self.stale_disappeared_penalty = -14.0
        self.non_global_arrival_penalty = -8.0
        self.observe_no_progress_penalty = -1.2
        self.observe_low_speed_penalty = -1.0
        self.observe_commit_window_miss_penalty = -1.6
        self.same_edge_repeat_chase_penalty = -1.2
        self.fallback_missed_lane_penalty = -1.0
        self.loop_trap_override_penalty = -1.4

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
        #         + commit flag + 3 lane features + 3 travel-time features
        #         + local congestion summary
        self.edge_embedding_dim = 8
        self.local_congestion_k = 6
        self._init_edge_embeddings(seed=1337)
        self.state_size = (2 * self.edge_embedding_dim) + 24 + 1 + 3 + 3 + self.local_congestion_k
        self.action_size = 6
        self.metrics_csv_path = os.path.join(self.sumocfg_dir, "rl_episode_metrics.csv")
        self._density_vec = np.zeros(len(self.connection_info.edge_list), dtype=np.float32)
        self._density_mean = 0.0
        self._density_std = 0.0
        self._last_density_step = -10**9
        self._lane_length_cache = {}
        self._passenger_edge_set = set(self.connection_info.edge_list)
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
            use_double_dqn=use_double_dqn,
        )
        self._decision_debug_fields = [
            "episode", "step", "vehicle_id", "decision_edge", "action", "action_source", "available_actions",
            "forced_action", "lane_feasible_now_actions", "reachable_with_lane_change_actions",
            "commit_window", "dist_to_end", "intended_next_edge", "actual_next_edge", "finalized",
            "finalize_delay_steps", "reward", "done", "route_mismatch", "teleported",
            "reached_global_destination", "prev_eta", "curr_eta", "prev_distance", "curr_distance",
            "edge_density", "mean_density", "externality_penalty", "marginal_pressure",
            "terminal_outcome", "last_confirmed_edge", "destination", "in_arrived_ids", "in_teleport_ids",
        ]

    def _ensure_decision_debug_csv_header(self):
        if not self.decision_debug_csv_path:
            return
        if not os.path.isabs(self.decision_debug_csv_path):
            self.decision_debug_csv_path = os.path.join(self.sumocfg_dir, self.decision_debug_csv_path)
        if not os.path.exists(self.decision_debug_csv_path):
            with open(self.decision_debug_csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._decision_debug_fields).writeheader()

    def _append_decision_debug_row(self, row):
        if not self.decision_debug_csv_path:
            return
        with open(self.decision_debug_csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._decision_debug_fields).writerow(row)

    def _append_decision_debug_rows(self, rows):
        if not self.decision_debug_csv_path or not rows:
            return
        with open(self.decision_debug_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._decision_debug_fields)
            writer.writerows(rows)

    def _build_decision_debug_row(
        self,
        episode,
        step,
        pending,
        actual_next_edge,
        finalized,
        finalize_delay_steps,
        reward="",
        done="",
        route_mismatch="",
        teleported=0,
        reached_global_destination=0,
        prev_eta="",
        curr_eta="",
        prev_distance="",
        curr_distance="",
        edge_density="",
        mean_density="",
        externality_penalty="",
        marginal_pressure="",
        terminal_outcome="",
        last_confirmed_edge="",
        destination="",
        in_arrived_ids="",
        in_teleport_ids="",
    ):
        context = pending.context
        return {
            "episode": episode,
            "step": step,
            "vehicle_id": context.vehicle_id,
            "decision_edge": pending.decision_edge,
            "action": pending.intended_action,
            "action_source": pending.metadata.get("action_source", ""),
            "available_actions": json.dumps(context.available_actions),
            "forced_action": context.forced_action if context.forced_action is not None else "",
            "lane_feasible_now_actions": json.dumps(context.lane_feasible_now_actions),
            "reachable_with_lane_change_actions": json.dumps(context.reachable_with_lane_change_actions),
            "commit_window": int(bool(context.commit_window)),
            "dist_to_end": float(context.dist_to_end),
            "intended_next_edge": pending.intended_next_edge,
            "actual_next_edge": actual_next_edge,
            "finalized": int(bool(finalized)),
            "finalize_delay_steps": finalize_delay_steps,
            "reward": reward,
            "done": done,
            "route_mismatch": route_mismatch,
            "teleported": teleported,
            "reached_global_destination": reached_global_destination,
            "prev_eta": prev_eta,
            "curr_eta": curr_eta,
            "prev_distance": prev_distance,
            "curr_distance": curr_distance,
            "edge_density": edge_density,
            "mean_density": mean_density,
            "externality_penalty": externality_penalty,
            "marginal_pressure": marginal_pressure,
            "terminal_outcome": terminal_outcome,
            "last_confirmed_edge": last_confirmed_edge,
            "destination": destination,
            "in_arrived_ids": in_arrived_ids,
            "in_teleport_ids": in_teleport_ids,
        }

    def _get_route_difficulty_scale(self, vehicle, reference_edge):
        """
        Return a multiplicative scale in [route_difficulty_scale_min, route_difficulty_scale_max]
        used for the per-step travel-time component only.
        """
        if not self.normalize_per_step_cost_by_route_difficulty:
            return 1.0

        cached_scale = getattr(vehicle, "route_difficulty_scale", None)
        if cached_scale is not None:
            return float(cached_scale)

        eta = self._estimate_remaining_eta(reference_edge, vehicle.destination)
        if not math.isfinite(eta):
            scale = 1.0
        else:
            # Harder/longer O-D pairs (larger ETA) receive a smaller per-step weight.
            normalized = self.route_difficulty_eta_floor / max(float(eta), self.route_difficulty_eta_floor)
            scale = float(np.clip(normalized, self.route_difficulty_scale_min, self.route_difficulty_scale_max))

        vehicle.route_difficulty_scale = scale
        return scale

    def _print_step_progress(
        self,
        episode,
        step,
        total_controlled,
        arrived_ids,
        decision_metrics,
    ):
        override_total = (
            decision_metrics["safety_overrides"]
            + decision_metrics["fallback_overrides"]
            + decision_metrics["route_apply_fail"]
        )
        completion = (len(arrived_ids) / float(total_controlled)) if total_controlled > 0 else 0.0
        failed = max(total_controlled - len(arrived_ids), 0)
        print(
            "[EP {:03d} | STEP {:04d}] eps={:.3f} replay={} train={} loss={} "
            "done={}/{} fail={} open/final/skip={:.0f}/{:.0f}/{:.0f} forced={:.0f}".format(
                episode,
                step,
                self.trainer.epsilon,
                len(self.trainer.memory),
                self.trainer.train_steps,
                "n/a" if self.trainer.last_loss is None else f"{self.trainer.last_loss:.4f}",
                len(arrived_ids),
                total_controlled,
                failed,
                decision_metrics["decisions_opened"],
                decision_metrics["decisions_finalized"],
                decision_metrics["decisions_skipped"],
                decision_metrics["forced_actions"],
            )
        )
        print(
            "  rates: completion={:.1%} override={:.1%} mismatch={} teleports={}".format(
                completion,
                override_total / max(decision_metrics["decisions_opened"], 1.0),
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
        mean_global = float(self._density_mean)
        std_global = float(self._density_std)

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

    def encode_state(self, vehicle_id, edge_id, destination_edge, context=None, vehicle=None, step=None, snapshot=None):
        """
        Build a state vector for the given edge using cached per-step densities.
        """
        state = np.zeros(self.state_size, dtype=np.float32)
        state[0:self.edge_embedding_dim] = self._get_edge_embedding(edge_id)
        state[self.edge_embedding_dim:(2 * self.edge_embedding_dim)] = self._get_edge_embedding(destination_edge)

        if context is None:
            if snapshot is not None:
                self._cache_metrics["snapshot_cache_hits"] += 1
            context = self.decision_engine.build_context(
                vehicle_id,
                edge_id,
                destination_edge,
                int(step or 0),
                snapshot=snapshot,
            )
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

        # Travel-time objective features (normalized)
        objective_base = lane_base + 3
        if vehicle is not None:
            if step is None:
                step = int(snapshot.step) if snapshot is not None else 0
            elapsed = max(float(step) - float(vehicle.start_time), 0.0)
            remaining_eta = self._estimate_remaining_eta(edge_id, destination_edge)
            density = self.connection_info.edge_vehicle_count.get(edge_id, 0) / max(
                self.connection_info.edge_length_dict.get(edge_id, 5.0), 5.0
            )

            state[objective_base + 0] = min(elapsed / float(MAX_SIMULATION_STEPS), 1.0)
            state[objective_base + 1] = (
                min(float(remaining_eta) / float(MAX_SIMULATION_STEPS), 1.0)
                if math.isfinite(remaining_eta) else 1.0
            )
            state[objective_base + 2] = min(float(density), 1.0)

        state[objective_base + 3:] = self._local_congestion_features(edge_id)
        return state.reshape(1, -1)
    
    def valid_actions_for_vehicle(self, vehicle_id, edge_id, destination_edge, step):
        context = self.decision_engine.build_context(vehicle_id, edge_id, destination_edge, step)
        return context.available_actions

    def _policy_action_candidates(self, context, recent_history, cooldown_active, destination):
        """
        Build a stricter action subset for policy selection only.
        NOTE:
        - context.available_actions remains the full safety/feasibility action set.
        - fallback machinery still relies on available_actions and ranked fallback behavior.
        """
        available_actions = list(context.available_actions)
        if not available_actions:
            return []

        lane_now = set(context.lane_feasible_now_actions)
        recent_history = list(recent_history or [])

        commit_distance = max(
            float(self.decision_engine.commit_min_distance),
            float(context.speed) * float(self.decision_engine.commit_time_s),
        )
        extra_buffer = max(10.0, 0.5 * float(self.decision_engine.lane_change_margin_m))
        comfortable_dist_threshold = commit_distance + extra_buffer

        safe_lane_now_actions = []
        strict_non_lane_actions = []
        filtered_available_actions = []

        for action in available_actions:
            safe_ok, _ = self.decision_engine.prefilter_action_for_loops(
                context=context,
                action_idx=action,
                destination=destination,
                recent_history=recent_history,
                distance_fn=self.get_distance_to_destination,
            )
            if not safe_ok:
                continue
            filtered_available_actions.append(action)
            if action in lane_now:
                safe_lane_now_actions.append(action)
                continue

            # Non-lane-feasible actions are exposed only in exceptional cases.
            if cooldown_active:
                continue
            if context.commit_window:
                continue
            if float(context.speed) < 1.2:
                continue
            if int(context.required_lane_shift.get(action, 99)) != 1:
                continue
            if float(context.dist_to_end) <= comfortable_dist_threshold:
                continue
            strict_non_lane_actions.append(action)

        # Dominant learning space: lane-feasible-now actions if any safe options exist.
        policy_actions = sorted(set(safe_lane_now_actions)) if safe_lane_now_actions else sorted(set(strict_non_lane_actions))
        if not policy_actions:
            policy_actions = sorted(set(filtered_available_actions))
        if not policy_actions:
            return available_actions
        return policy_actions
    
    def dist_to_end(self, vehicle_id, snapshot=None):
        """
        Distance (meters) from the vehicle to the end of its current lane.
        """
        if snapshot is not None:
            return max(float(snapshot.dist_to_end), 0.0)
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        lane_len = traci.lane.getLength(lane_id)
        lane_pos = traci.vehicle.getLanePosition(vehicle_id)
        return max(lane_len - lane_pos, 0.0)


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

    def _lane_length(self, lane_id):
        if lane_id in self._lane_length_cache:
            return self._lane_length_cache[lane_id]
        lane_len = float(traci.lane.getLength(lane_id))
        self._lane_length_cache[lane_id] = lane_len
        return lane_len

    def collect_vehicle_snapshots(self, vehicle_ids, step):
        snapshots = {}
        for vehicle_id in vehicle_ids:
            edge_id = traci.vehicle.getRoadID(vehicle_id)
            if edge_id not in self._passenger_edge_set:
                continue
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            lane_index = int(traci.vehicle.getLaneIndex(vehicle_id))
            lane_position = float(traci.vehicle.getLanePosition(vehicle_id))
            lane_length = self._lane_length(lane_id)
            lane_count = max(len(self.connection_info.edge_lane_ids.get(edge_id, [])), 1)
            speed = max(float(traci.vehicle.getSpeed(vehicle_id)), 0.0)
            dist_to_end = max(lane_length - lane_position, 0.0)
            snapshots[vehicle_id] = VehicleSnapshot(
                vehicle_id=vehicle_id,
                step=int(step),
                edge_id=edge_id,
                lane_id=lane_id,
                lane_index=lane_index,
                lane_count=lane_count,
                lane_position=lane_position,
                lane_length=lane_length,
                dist_to_end=dist_to_end,
                speed=speed,
            )
        return snapshots

    def cleanup_vehicle_state(
        self,
        vehicle_id,
        pending_decisions,
        prev_edge_by_vehicle,
        last_seen_edge_by_vehicle,
        last_planned_terminal_edge_by_vehicle,
        recent_edge_history,
        last_snapshot_by_vehicle,
    ):
        pending_decisions.pop(vehicle_id, None)
        prev_edge_by_vehicle.pop(vehicle_id, None)
        last_seen_edge_by_vehicle.pop(vehicle_id, None)
        last_planned_terminal_edge_by_vehicle.pop(vehicle_id, None)
        last_snapshot_by_vehicle.pop(vehicle_id, None)
        recent_edge_history.pop(vehicle_id, None)

    def make_terminal_next_state_from_snapshot(self, snapshot, destination_edge, vehicle=None, step=None):
        if snapshot is None:
            return self.make_terminal_next_state_from_edge(None, destination_edge, vehicle=vehicle, step=step)
        return self.encode_state(
            snapshot.vehicle_id,
            snapshot.edge_id,
            destination_edge,
            context=self.decision_engine.build_context(
                snapshot.vehicle_id,
                snapshot.edge_id,
                destination_edge,
                int(step if step is not None else snapshot.step),
                snapshot=snapshot,
            ),
            vehicle=vehicle,
            step=step if step is not None else snapshot.step,
            snapshot=snapshot,
        )

    def make_terminal_next_state_from_edge(self, edge_id, destination_edge, vehicle=None, step=None):
        if (
            edge_id in self.connection_info.edge_index_dict
            and destination_edge in self.connection_info.edge_index_dict
        ):
            terminal_context = self.decision_engine.build_context(
                vehicle_id="__terminal__",
                edge_id=edge_id,
                destination=destination_edge,
                step=int(step or 0),
                snapshot=VehicleSnapshot(
                    vehicle_id="__terminal__",
                    step=int(step or 0),
                    edge_id=edge_id,
                    lane_id=self.connection_info.edge_lane_ids.get(edge_id, [""])[0] if self.connection_info.edge_lane_ids.get(edge_id) else "",
                    lane_index=0,
                    lane_count=max(len(self.connection_info.edge_lane_ids.get(edge_id, [])), 1),
                    lane_position=0.0,
                    lane_length=float(self.connection_info.edge_length_dict.get(edge_id, 0.0)),
                    dist_to_end=float(self.connection_info.edge_length_dict.get(edge_id, 0.0)),
                    speed=0.0,
                ),
            )
            return self.encode_state(
                "__terminal__",
                edge_id,
                destination_edge,
                context=terminal_context,
                vehicle=vehicle,
                step=step,
            )
        return np.zeros((1, self.state_size), dtype=np.float32)

    def _classify_terminal_outcome(self, vehicle_id, arrived_ids, teleport_ids, is_live):
        """
        Classify terminal outcome for a controlled vehicle removed from the live set.
        Arrival classification is strictly based on SUMO arrived ids.
        """
        if vehicle_id in arrived_ids:
            return "global_arrival"
        if (vehicle_id in teleport_ids) and (not is_live):
            return "teleport"
        return "removed_nonarrival"

    def _finalize_terminal_transition(
        self,
        vehicle_id,
        vehicle,
        outcome,
        step,
        pending_decisions,
        terminal_recorded_ids,
        last_snapshot_by_vehicle,
        last_seen_edge_by_vehicle,
        decision_metrics,
        decision_latency_steps,
        finalized_decision_rewards,
        decision_debug_rows,
        episode,
        in_arrived_ids=False,
        in_teleport_ids=False,
    ):
        if vehicle_id in terminal_recorded_ids:
            return 0.0
        pending = pending_decisions.get(vehicle_id)
        if pending is None:
            last_confirmed_edge = last_seen_edge_by_vehicle.get(vehicle_id, vehicle.destination)
            snapshot = last_snapshot_by_vehicle.get(vehicle_id)
            base_state = self.make_terminal_next_state_from_snapshot(
                snapshot,
                vehicle.destination,
                vehicle=vehicle,
                step=step,
            )
            if outcome == "global_arrival":
                reward, done = self.compute_reward(
                    vehicle,
                    vehicle.destination,
                    vehicle.destination,
                    step,
                    arrived=True,
                    reached_global_destination=True,
                    terminal_outcome=outcome,
                )
                next_state = self.make_terminal_next_state_from_edge(
                    vehicle.destination,
                    vehicle.destination,
                    vehicle=vehicle,
                    step=step,
                )
            elif outcome == "teleport":
                reward = self._clip_reward(self.teleport_penalty)
                done = True
                next_state = base_state
                decision_metrics["fail_teleport"] += 1
            elif outcome == "timeout":
                reward = self._clip_reward(self.pending_timeout_penalty)
                done = True
                next_state = base_state
                decision_metrics["fail_timeout"] += 1
            elif outcome == "removed_nonarrival":
                reward = self._clip_reward(self.stale_disappeared_penalty)
                done = True
                next_state = base_state
                decision_metrics["fail_removed_non_destination"] += 1
            else:
                raise ValueError(f"Unknown terminal outcome: {outcome}")
            self.trainer.remember(
                base_state,
                0,
                reward,
                next_state,
                done,
                next_valid_actions=[],
                metadata={"terminal_outcome": outcome, "synthetic_terminal_no_pending": True},
            )
            decision_metrics["decisions_finalized"] += 1
            finalized_decision_rewards.append(float(reward))
            decision_debug_rows.append({
                "episode": episode,
                "step": step,
                "vehicle_id": vehicle_id,
                "decision_edge": last_confirmed_edge,
                "action": 0,
                "action_source": "synthetic_terminal",
                "available_actions": "[]",
                "forced_action": "",
                "lane_feasible_now_actions": "[]",
                "reachable_with_lane_change_actions": "[]",
                "commit_window": "",
                "dist_to_end": "",
                "intended_next_edge": "",
                "actual_next_edge": vehicle.destination if outcome == "global_arrival" else last_confirmed_edge,
                "finalized": 1,
                "finalize_delay_steps": 0,
                "reward": reward,
                "done": 1,
                "route_mismatch": "",
                "teleported": 1 if outcome == "teleport" else 0,
                "reached_global_destination": 1 if outcome == "global_arrival" else 0,
                "prev_eta": "",
                "curr_eta": "",
                "prev_distance": "",
                "curr_distance": "",
                "edge_density": "",
                "mean_density": "",
                "externality_penalty": "",
                "marginal_pressure": "",
                "terminal_outcome": outcome,
                "last_confirmed_edge": last_confirmed_edge,
                "destination": vehicle.destination,
                "in_arrived_ids": int(bool(in_arrived_ids)),
                "in_teleport_ids": int(bool(in_teleport_ids)),
            })
            terminal_recorded_ids.add(vehicle_id)
            return float(reward)

        last_confirmed_edge = last_seen_edge_by_vehicle.get(vehicle_id, pending.last_credit_edge)
        snapshot = last_snapshot_by_vehicle.get(vehicle_id)
        if outcome == "global_arrival":
            reward, done = self.compute_reward(
                vehicle,
                pending.last_credit_edge,
                vehicle.destination,
                step,
                arrived=True,
                delta_t=max(step - pending.last_credit_step, 1),
                reached_global_destination=True,
                terminal_outcome=outcome,
            )
            next_state = self.make_terminal_next_state_from_edge(
                vehicle.destination,
                vehicle.destination,
                vehicle=vehicle,
                step=step,
            )
            decision_metrics["arrived_global_destination"] += 1
        elif outcome == "teleport":
            reward = self._clip_reward(self.teleport_penalty)
            done = True
            next_state = self.make_terminal_next_state_from_snapshot(
                snapshot,
                vehicle.destination,
                vehicle=vehicle,
                step=step,
            )
            decision_metrics["fail_teleport"] += 1
        elif outcome == "timeout":
            timeout_elapsed = max(step - pending.last_credit_step, 0)
            reward = self.compute_pending_step_reward(
                vehicle,
                last_confirmed_edge,
                elapsed=timeout_elapsed,
                pending_age=max(step - pending.decision_step, 0),
                lane_change_deferrals=pending.metadata.get("lane_change_deferrals", 0),
            ) + self.pending_timeout_penalty
            reward = self._clip_reward(reward)
            done = True
            next_state = self.make_terminal_next_state_from_snapshot(
                snapshot,
                vehicle.destination,
                vehicle=vehicle,
                step=step,
            )
            decision_metrics["fail_timeout"] += 1
        elif outcome == "removed_nonarrival":
            reward = self._clip_reward(self.stale_disappeared_penalty)
            done = True
            next_state = self.make_terminal_next_state_from_snapshot(
                snapshot,
                vehicle.destination,
                vehicle=vehicle,
                step=step,
            )
            decision_metrics["fail_removed_non_destination"] += 1
        else:
            raise ValueError(f"Unknown terminal outcome: {outcome}")

        self.trainer.remember(
            pending.state,
            pending.intended_action,
            reward,
            next_state,
            done,
            next_valid_actions=[],
            metadata={"terminal_outcome": outcome},
        )
        decision_metrics["decisions_finalized"] += 1
        decision_latency_steps.append(float(max(step - pending.decision_step, 0)))
        finalized_decision_rewards.append(float(reward))
        terminal_recorded_ids.add(vehicle_id)
        decision_debug_rows.append(self._build_decision_debug_row(
            episode=episode,
            step=step,
            pending=pending,
            actual_next_edge=vehicle.destination if outcome == "global_arrival" else last_confirmed_edge,
            finalized=1,
            finalize_delay_steps=max(step - pending.decision_step, 0),
            reward=reward,
            done=1,
            teleported=1 if outcome == "teleport" else 0,
            reached_global_destination=1 if outcome == "global_arrival" else 0,
            terminal_outcome=outcome,
            last_confirmed_edge=last_confirmed_edge,
            destination=vehicle.destination,
            in_arrived_ids=int(bool(in_arrived_ids)),
            in_teleport_ids=int(bool(in_teleport_ids)),
        ))
        return float(reward)

    def _edge_out_degree_map(self, edges):
        return {
            edge: len(self.connection_info.outgoing_edges_dict.get(edge, {}))
            for edge in set(edges)
        }

    def _select_fallback_action(self, context, blocked_action, destination, recent_history):
        ranked = self.decision_engine.ranked_fallback_actions(
            context=context,
            destination=destination,
            recent_history=recent_history,
            blocked_action=blocked_action,
            distance_fn=self.get_distance_to_destination,
        )
        if ranked:
            return ranked[0]
        return None

    def _estimate_remaining_eta(self, edge_id, destination_edge):
        """
        Estimate travel time from edge_id to destination using shortest-path
        distance and a conservative minimum speed floor.
        """
        distance = self.get_distance_to_destination(edge_id, destination_edge)
        if not math.isfinite(distance):
            return math.inf
        return float(distance) / 8.0

    def get_distance_to_destination(self, edge_id, destination_edge):
        """
        Return shortest-path cost from edge_id to destination_edge.
        Uses a cache because this is called frequently during training.
        """
        key = (edge_id, destination_edge)
        if key in self._distance_cache:
            self._cache_metrics["shortest_path_cache_hits"] += 1
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
        long_horizon_loop=False,
        externality_penalty=0.0,
        terminal_outcome=None,
    ):
        """
        Compute a bounded reward with clear objective priority:
        1) minimize travel time (per-step cost)
        2) complete trips successfully
        3) prefer progress with mild congestion awareness.
        """
        elapsed = max(float(delta_t), 1.0)

        prev_distance = self.get_distance_to_destination(prev_edge, vehicle.destination)
        curr_distance = self.get_distance_to_destination(current_edge, vehicle.destination)
        prev_eta = self._estimate_remaining_eta(prev_edge, vehicle.destination)
        curr_eta = self._estimate_remaining_eta(current_edge, vehicle.destination)

        reward = 0.0
        done = False

        # Dense shaping: small living/time and congestion costs.
        time_cost_scale = self._get_route_difficulty_scale(vehicle, prev_edge)
        reward -= self.travel_time_penalty * time_cost_scale * elapsed
        congestion = self.connection_info.edge_vehicle_count.get(current_edge, 0)
        edge_len = max(self.connection_info.edge_length_dict.get(current_edge, 5.0), 5.0)
        edge_density = congestion / edge_len
        reward -= 0.015 * edge_density * elapsed
        reward -= float(np.clip(externality_penalty, 0.0, 1.5))

        mean_density = float(np.mean(self._density_vec)) if len(self._density_vec) > 0 else 0.0
        marginal_pressure = max(edge_density - mean_density, 0.0)
        reward -= self.system_congestion_scale * marginal_pressure * elapsed

        # Progress shaping using ETA and distance improvement.
        if math.isfinite(prev_eta) and math.isfinite(curr_eta):
            reward += self.eta_progress_scale * np.clip(prev_eta - curr_eta, -3.0, 3.0)

        # Tertiary tie-breaker: shortest-path distance progress.
        if math.isfinite(prev_distance) and math.isfinite(curr_distance):
            progress = (prev_distance - curr_distance) * (self.distance_tiebreak_scale * self.progress_reward_scale)
            reward += float(np.clip(progress, -1.0, 1.0))

        # Safety and control quality penalties.
        if repeated_recent_edges > 0:
            reward -= self.loop_repeat_penalty * min(repeated_recent_edges, 3)
        if uturn_repeat:
            reward -= 3.0
        if long_horizon_loop:
            reward -= 4.0
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
                speed_bonus = max(0.0, 1.0 - (float(step) / float(MAX_SIMULATION_STEPS)))
                reward += 3.0 * speed_bonus
            elif terminal_outcome == "non_global_arrival":
                reward += self.non_global_arrival_penalty
            else:
                reward -= 8.0
            done = True
            return self._clip_reward(reward), done

        outgoing = self.connection_info.outgoing_edges_dict.get(current_edge, {})
        if (not outgoing or len(outgoing) == 0) and current_edge != vehicle.destination:
            reward -= 12.0
            done = True

        return self._clip_reward(reward), done

    def _clip_reward(self, reward_value):
        return float(np.clip(reward_value, self.reward_clip_low, self.reward_clip_high))

    def compute_pending_step_reward(self, vehicle, edge_id, elapsed, externality_penalty=0.0, pending_age=0, lane_change_deferrals=0):
        """
        Dense reward used while a decision is pending and has not finalized yet.
        Keeps the training objective travel-time centric without waiting for an edge transition.
        """
        elapsed = max(float(elapsed), 0.0)
        if elapsed <= 0.0:
            return 0.0
        time_cost_scale = self._get_route_difficulty_scale(vehicle, edge_id)
        reward = -self.travel_time_penalty * time_cost_scale * elapsed

        congestion = self.connection_info.edge_vehicle_count.get(edge_id, 0)
        edge_len = max(self.connection_info.edge_length_dict.get(edge_id, 5.0), 5.0)
        edge_density = congestion / edge_len
        reward -= 0.015 * edge_density * elapsed
        reward -= float(np.clip(externality_penalty, 0.0, 1.5))

        mean_density = float(np.mean(self._density_vec)) if len(self._density_vec) > 0 else 0.0
        marginal_pressure = max(edge_density - mean_density, 0.0)
        reward -= self.system_congestion_scale * marginal_pressure * elapsed
        reward -= self.pending_latency_penalty_per_step * float(max(pending_age, 0))
        reward -= 0.02 * float(max(lane_change_deferrals, 0))
        return self._clip_reward(reward)

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
        rolling_avg_travel_time = deque(maxlen=self.rolling_window)
        rolling_mismatch = deque(maxlen=self.rolling_window)
        csv_fields = [
            "episode", "epsilon", "replay", "train_steps", "mean_loss", "episode_return",
            "completion_rate", "avg_travel_time", "p50_travel_time", "p90_travel_time", "teleports", "teleported_controlled",
            "forced_actions", "decisions_opened", "decisions_finalized", "decisions_skipped",
            "decisions_superseded", "route_mismatch", "loop_events", "uturn_events",
            "short_cycle_events", "aba_bounce_events", "dead_end_reentry_events",
            "long_horizon_loop_events", "revisit_without_progress_events",
            "safety_overrides", "loop_avoidance_overrides", "distance_worsening_overrides",
            "fragment_build_failures", "fallback_overrides", "fallback_to_lane_feasible_now",
            "pending_decision_timeouts", "deferred_lane_change_actions",
            "lane_change_observe_started", "lane_change_observe_success",
            "lane_change_observe_abort_no_progress", "lane_change_observe_abort_commit_window",
            "same_edge_pending_released_no_progress", "cooldown_replans_blocked",
            "loop_override_count", "dead_end_reentry_override_count",
            "batched_policy_calls", "snapshot_cache_hits", "shortest_path_cache_hits",
            "exploration_actions", "policy_actions", "override_ratio",
            "policy_masked_actions_removed", "override_learning_transitions",
            "loop_prefilter_overrides", "cooldown_fallback_overrides",
            "observe_abort_fallback_overrides", "route_apply_fail_overrides",
            "override_learning_negative", "override_learning_imitation",
            "timeout_unfinished_controlled", "exited_without_destination", "arrived_non_global_target",
            "alive_at_step_cap", "decision_pending_at_episode_end", "mean_pending_age", "mean_decision_latency_steps",
            "mean_reward_per_finalized_decision", "mean_route_difficulty_eta", "p50_route_difficulty_eta",
            "p90_route_difficulty_eta", "fail_teleport", "fail_timeout", "fail_removed_non_destination",
            "fail_unreachable_transition", "fail_dead_end_no_outgoing",
        ]
        # Rewrite metrics each new training session to avoid schema drift/appending old runs.
        with open(self.metrics_csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()
        self._ensure_decision_debug_csv_header()

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
            simulation_get_min_expected = traci.simulation.getMinExpectedNumber
            simulation_step = traci.simulationStep
            simulation_get_arrived_ids = traci.simulation.getArrivedIDList
            vehicle_get_ids = traci.vehicle.getIDList
            pending_decisions = {}
            lane_change_deferrals = defaultdict(int)
            lane_change_cooldown_until = {}
            recent_edge_history = defaultdict(lambda: deque(maxlen=self.loop_window))
            prev_edge_by_vehicle = {}
            decision_metrics = defaultdict(float)

            episode_return = 0.0
            episode_teleport_events = 0
            teleported_controlled_ids = set()
            arrived_ids = set()
            total_controlled = len(vehicles)
            last_seen_edge_by_vehicle = {}
            last_planned_terminal_edge_by_vehicle = {}
            last_snapshot_by_vehicle = {}
            completed_travel_times = []
            completed_travel_time_ids = set()
            terminal_recorded_ids = set()
            arrived_debug_records = []
            final_outcome_by_vehicle = {}
            decision_latency_steps = []
            pending_age_samples = []
            finalized_decision_rewards = []
            route_difficulty_etas = []
            last_step_executed = -1
            alive_at_step_cap_ids = set()
            pending_debug_logged_ids = set()
            decision_debug_rows = []
            arrived_with_prestep_edge_not_destination = 0

            try:
                for step in range(MAX_SIMULATION_STEPS):
                    if simulation_get_min_expected() <= 0:
                        break
                    last_step_executed = step

                    # Keep density features fresh for routing choices and reward.
                    self.update_edge_vehicle_counts(step, every=self.density_refresh_every)
                    vehicle_ids = list(vehicle_get_ids())
                    controlled_live_ids = [vid for vid in vehicle_ids if vid in vehicles]
                    step_snapshots = self.collect_vehicle_snapshots(controlled_live_ids, step)

                    for vehicle_id in controlled_live_ids:
                        snapshot = step_snapshots.get(vehicle_id)
                        if snapshot is None:
                            continue

                        current_edge = snapshot.edge_id

                        vehicle = vehicles[vehicle_id]
                        vehicle.current_edge = current_edge
                        vehicle.current_speed = snapshot.speed
                        if getattr(vehicle, "_route_difficulty_eta_logged", False) is False:
                            eta0 = self._estimate_remaining_eta(current_edge, vehicle.destination)
                            if math.isfinite(eta0):
                                route_difficulty_etas.append(float(eta0))
                            vehicle._route_difficulty_eta_logged = True
                        last_seen_edge_by_vehicle[vehicle_id] = current_edge
                        last_snapshot_by_vehicle[vehicle_id] = snapshot
                        recent_edge_history[vehicle_id].append(current_edge)

                        # Do not cleanup pre-step destination reaches yet; terminal transition
                        # must be written exactly once before any state is removed.
                        if current_edge == vehicle.destination:
                            continue

                        prev_edge = prev_edge_by_vehicle.get(vehicle_id)
                        if vehicle_id in pending_decisions and current_edge != pending_decisions[vehicle_id].decision_edge:
                            pending = pending_decisions.pop(vehicle_id)
                            repeated_recent_edges = sum(1 for e in recent_edge_history[vehicle_id] if e == current_edge)
                            mismatch = not self.decision_engine.route_matches_expected(pending, current_edge)
                            if mismatch:
                                decision_metrics["route_mismatch"] += 1
                            signal_edges = set(recent_edge_history[vehicle_id]) | {current_edge}
                            edge_distance_lookup = {
                                edge: self.get_distance_to_destination(edge, vehicle.destination)
                                for edge in signal_edges
                            }
                            loop_signals = transition_signal(
                                recent_edge_history[vehicle_id],
                                current_edge,
                                edge_out_degree=self._edge_out_degree_map(list(recent_edge_history[vehicle_id]) + [current_edge]),
                                edge_distance_lookup=edge_distance_lookup,
                                progress_slack=self.decision_engine.loop_distance_slack,
                            )
                            if loop_signals["aba_bounce"]:
                                decision_metrics["aba_bounce_events"] += 1
                                decision_metrics["uturn_events"] += 1
                            if loop_signals["short_cycle"]:
                                decision_metrics["short_cycle_events"] += 1
                            if loop_signals["dead_end_reentry"]:
                                decision_metrics["dead_end_reentry_events"] += 1
                            if loop_signals.get("long_horizon_loop"):
                                decision_metrics["long_horizon_loop_events"] += 1
                            if loop_signals.get("revisit_without_progress"):
                                decision_metrics["revisit_without_progress_events"] += 1
                            ext_pen = max(self.connection_info.edge_vehicle_count.get(current_edge, 0) / max(self.connection_info.edge_length_dict.get(current_edge, 10.0), 10.0), 0.0)
                            prev_distance = self.get_distance_to_destination(pending.decision_edge, vehicle.destination)
                            curr_distance = self.get_distance_to_destination(current_edge, vehicle.destination)
                            prev_eta = self._estimate_remaining_eta(pending.decision_edge, vehicle.destination)
                            curr_eta = self._estimate_remaining_eta(current_edge, vehicle.destination)
                            reward, done = self.compute_reward(
                                vehicle, pending.last_credit_edge, current_edge, step, arrived=False,
                                repeated_recent_edges=repeated_recent_edges,
                                delta_t=max(step - pending.last_credit_step, 1),
                                route_mismatch=mismatch,
                                uturn_repeat=loop_signals["aba_bounce"] or loop_signals["short_cycle"],
                                long_horizon_loop=loop_signals.get("long_horizon_loop") or loop_signals.get("revisit_without_progress"),
                                externality_penalty=ext_pen,
                            )
                            next_ctx = self.decision_engine.build_context(
                                vehicle_id,
                                current_edge,
                                vehicle.destination,
                                step,
                                snapshot=snapshot,
                            )
                            next_state = self.encode_state(
                                vehicle_id,
                                current_edge,
                                vehicle.destination,
                                context=next_ctx,
                                vehicle=vehicle,
                                step=step,
                                snapshot=snapshot,
                            )
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
                            decision_latency_steps.append(float(max(step - pending.decision_step, 0)))
                            finalized_decision_rewards.append(float(reward))
                            episode_return += reward
                            if repeated_recent_edges > 1:
                                decision_metrics["loop_events"] += 1
                            if math.isfinite(prev_distance) and (not math.isfinite(curr_distance)):
                                decision_metrics["fail_unreachable_transition"] += 1
                            outgoing = self.connection_info.outgoing_edges_dict.get(current_edge, {})
                            if (not outgoing or len(outgoing) == 0) and current_edge != vehicle.destination:
                                decision_metrics["fail_dead_end_no_outgoing"] += 1
                            edge_density = self.connection_info.edge_vehicle_count.get(current_edge, 0) / max(
                                self.connection_info.edge_length_dict.get(current_edge, 5.0), 5.0
                            )
                            mean_density = float(np.mean(self._density_vec)) if len(self._density_vec) > 0 else 0.0
                            marginal_pressure = max(edge_density - mean_density, 0.0)
                            decision_debug_rows.append(self._build_decision_debug_row(
                                episode=episode,
                                step=step,
                                pending=pending,
                                actual_next_edge=current_edge,
                                finalized=1,
                                finalize_delay_steps=max(step - pending.decision_step, 0),
                                reward=reward,
                                done=int(bool(done)),
                                route_mismatch=int(bool(mismatch)),
                                teleported=0,
                                reached_global_destination=0,
                                prev_eta=prev_eta if math.isfinite(prev_eta) else "",
                                curr_eta=curr_eta if math.isfinite(curr_eta) else "",
                                prev_distance=prev_distance if math.isfinite(prev_distance) else "",
                                curr_distance=curr_distance if math.isfinite(curr_distance) else "",
                                edge_density=edge_density,
                                mean_density=mean_density,
                                externality_penalty=ext_pen,
                                marginal_pressure=marginal_pressure,
                            ))
                        elif vehicle_id in pending_decisions:
                            pending = pending_decisions[vehicle_id]
                            pending_phase = pending.metadata.get("phase", "route_pending")
                            if pending_phase == "observe_lane_change":
                                obs_context = self.decision_engine.build_context(
                                    vehicle_id,
                                    current_edge,
                                    vehicle.destination,
                                    step,
                                    snapshot=snapshot,
                                )
                                self._cache_metrics["snapshot_cache_hits"] += 1
                                status, reason = self.decision_engine.evaluate_lane_change_observation(
                                    pending.metadata,
                                    obs_context,
                                    pending.intended_action,
                                )
                                if status == "continue":
                                    prev_edge_by_vehicle[vehicle_id] = current_edge
                                    continue
                                pending_decisions.pop(vehicle_id, None)
                                if status == "success":
                                    decision_metrics["lane_change_observe_success"] += 1
                                    full_route, committed_next_edge, apply_error = self.decision_engine.apply_route_decision(
                                        vehicle_id, current_edge, pending.intended_action, vehicle.destination
                                    )
                                    if apply_error:
                                        decision_metrics["route_apply_fail"] += 1
                                        decision_metrics["route_apply_fail_overrides"] += 1
                                        override_penalty = self._clip_reward(-6.0)
                                        obs_policy_actions = self._policy_action_candidates(
                                            context=obs_context,
                                            recent_history=list(recent_edge_history[vehicle_id]),
                                            cooldown_active=step < lane_change_cooldown_until.get((vehicle_id, current_edge), -1),
                                            destination=vehicle.destination,
                                        )
                                        self.trainer.remember(
                                            pending.state,
                                            pending.intended_action,
                                            override_penalty,
                                            pending.state,
                                            False,
                                            next_valid_actions=obs_policy_actions,
                                            metadata={
                                                "override_learning": True,
                                                "override_type": "route_apply_failure",
                                                "original_action": pending.intended_action,
                                                "fallback_action": None,
                                                "override_cause": "route_apply_failure",
                                                "route_apply_failed": True,
                                                "observe_phase": True,
                                            },
                                        )
                                        decision_metrics["override_learning_transitions"] += 1
                                        decision_metrics["override_learning_negative"] += 1
                                        episode_return += override_penalty
                                        prev_edge_by_vehicle[vehicle_id] = current_edge
                                        continue
                                    pending.metadata["phase"] = "route_pending"
                                    pending.intended_next_edge = committed_next_edge
                                    pending.route_fragment = list(full_route[1:]) if full_route else []
                                    pending.context = obs_context
                                    pending.decision_step = step
                                    pending.last_credit_step = step
                                    pending.last_credit_edge = current_edge
                                    pending_decisions[vehicle_id] = pending
                                    decision_metrics["decisions_opened"] += 1
                                    prev_edge_by_vehicle[vehicle_id] = current_edge
                                    continue
                                if reason == "commit_window":
                                    decision_metrics["lane_change_observe_abort_commit_window"] += 1
                                    pending_pen = self.observe_commit_window_miss_penalty
                                elif reason == "low_speed":
                                    decision_metrics["lane_change_observe_abort_no_progress"] += 1
                                    pending_pen = self.observe_low_speed_penalty
                                else:
                                    decision_metrics["lane_change_observe_abort_no_progress"] += 1
                                    pending_pen = self.observe_no_progress_penalty
                                pending_pen = self._clip_reward(pending_pen + self.same_edge_repeat_chase_penalty)
                                self.trainer.remember(
                                    pending.state,
                                    pending.intended_action,
                                    pending_pen,
                                    self.encode_state(
                                        vehicle_id, current_edge, vehicle.destination, context=obs_context, vehicle=vehicle, step=step, snapshot=snapshot
                                    ),
                                    False,
                                    next_valid_actions=obs_context.available_actions,
                                    metadata={
                                        "override_learning": True,
                                        "observe_abort": reason or "no_progress",
                                        "override_cause": "lane_change_observe_abort",
                                        "abort_reason": reason or "no_progress",
                                        "override_type": "observe_abort_fallback",
                                        "original_action": pending.intended_action,
                                    },
                                )
                                decision_metrics["override_learning_transitions"] += 1
                                decision_metrics["override_learning_negative"] += 1
                                episode_return += pending_pen
                                decision_metrics["same_edge_pending_released_no_progress"] += 1
                                cooldown_key = (vehicle_id, current_edge)
                                lane_change_cooldown_until[cooldown_key] = step + self.decision_engine.cooldown_steps
                                fallback_action = self._select_fallback_action(
                                    obs_context,
                                    blocked_action=pending.intended_action,
                                    destination=vehicle.destination,
                                    recent_history=list(recent_edge_history[vehicle_id]),
                                )
                                if fallback_action is None:
                                    prev_edge_by_vehicle[vehicle_id] = current_edge
                                    continue
                                full_route, committed_next_edge, apply_error = self.decision_engine.apply_route_decision(
                                    vehicle_id, current_edge, fallback_action, vehicle.destination
                                )
                                if apply_error:
                                    prev_edge_by_vehicle[vehicle_id] = current_edge
                                    continue
                                next_state = self.encode_state(
                                    vehicle_id, current_edge, vehicle.destination, context=obs_context, vehicle=vehicle, step=step, snapshot=snapshot
                                )
                                pending_decisions[vehicle_id] = PendingDecision(
                                    state=next_state,
                                    intended_action=fallback_action,
                                    intended_next_edge=committed_next_edge,
                                    decision_edge=current_edge,
                                    decision_step=step,
                                    last_credit_edge=current_edge,
                                    last_credit_step=step,
                                    destination=vehicle.destination,
                                    context=obs_context,
                                    lane_change_requested=False,
                                    route_fragment=list(full_route[1:]) if full_route else [],
                                    metadata={"phase": "route_pending", "action_source": "observe_fallback"},
                                )
                                decision_metrics["fallback_overrides"] += 1
                                decision_metrics["fallback_to_lane_feasible_now"] += 1
                                decision_metrics["observe_abort_fallback_overrides"] += 1
                                observe_policy_actions = self._policy_action_candidates(
                                    context=obs_context,
                                    recent_history=list(recent_edge_history[vehicle_id]),
                                    cooldown_active=True,
                                    destination=vehicle.destination,
                                )
                                imitation_reward = self._clip_reward(0.10)
                                self.trainer.remember(
                                    next_state,
                                    fallback_action,
                                    imitation_reward,
                                    next_state,
                                    False,
                                    next_valid_actions=observe_policy_actions,
                                    metadata={
                                        "override_learning": True,
                                        "override_type": "observe_abort_fallback",
                                        "original_action": pending.intended_action,
                                        "fallback_action": fallback_action,
                                        "imitation_credit": True,
                                    },
                                )
                                decision_metrics["override_learning_transitions"] += 1
                                decision_metrics["override_learning_imitation"] += 1
                                prev_edge_by_vehicle[vehicle_id] = current_edge
                                continue
                            pending_age = self.decision_engine.pending_age_steps(pending, step)
                            pending_age_samples.append(float(pending_age))
                            elapsed_pending = max(step - pending.last_credit_step, 0)
                            if elapsed_pending > 0:
                                ext_pen = max(
                                    self.connection_info.edge_vehicle_count.get(current_edge, 0)
                                    / max(self.connection_info.edge_length_dict.get(current_edge, 10.0), 10.0),
                                    0.0,
                                )
                                pending_reward = self.compute_pending_step_reward(
                                    vehicle,
                                    current_edge,
                                    elapsed=elapsed_pending,
                                    externality_penalty=ext_pen,
                                    pending_age=pending_age,
                                    lane_change_deferrals=pending.metadata.get("lane_change_deferrals", 0),
                                )
                                next_ctx = self.decision_engine.build_context(
                                    vehicle_id,
                                    current_edge,
                                    vehicle.destination,
                                    step,
                                    snapshot=snapshot,
                                )
                                next_state = self.encode_state(
                                    vehicle_id,
                                    current_edge,
                                    vehicle.destination,
                                    context=next_ctx,
                                    vehicle=vehicle,
                                    step=step,
                                    snapshot=snapshot,
                                )
                                self.trainer.remember(
                                    pending.state,
                                    pending.intended_action,
                                    pending_reward,
                                    next_state,
                                    False,
                                    next_valid_actions=next_ctx.available_actions,
                                    metadata={"interim_pending_credit": True},
                                )
                                episode_return += pending_reward
                                pending.state = next_state
                                pending.last_credit_edge = current_edge
                                pending.last_credit_step = step
                            if self.decision_engine.should_timeout_pending(pending, step):
                                timeout_ctx = self.decision_engine.build_context(
                                    vehicle_id,
                                    current_edge,
                                    vehicle.destination,
                                    step,
                                    snapshot=snapshot,
                                )
                                timeout_state = self.encode_state(
                                    vehicle_id,
                                    current_edge,
                                    vehicle.destination,
                                    context=timeout_ctx,
                                    vehicle=vehicle,
                                    step=step,
                                    snapshot=snapshot,
                                )
                                timeout_penalty = self._clip_reward(self.pending_replan_penalty)
                                self.trainer.remember(
                                    pending.state,
                                    pending.intended_action,
                                    timeout_penalty,
                                    timeout_state,
                                    False,
                                    next_valid_actions=timeout_ctx.available_actions,
                                    metadata={"pending_timeout_replan": True},
                                )
                                episode_return += timeout_penalty
                                decision_metrics["pending_decision_timeouts"] += 1
                                decision_metrics["same_edge_pending_released_no_progress"] += 1
                                pending_decisions.pop(vehicle_id, None)
                                lane_change_deferrals[vehicle_id] = 0
                                lane_change_cooldown_until[(vehicle_id, current_edge)] = step + self.decision_engine.cooldown_steps
                                continue
                            pending_ctx = self.decision_engine.build_context(
                                vehicle_id,
                                current_edge,
                                vehicle.destination,
                                step,
                                snapshot=snapshot,
                            )
                            wrong_lane_commit = (
                                pending_ctx.commit_window
                                and pending.intended_action not in pending_ctx.lane_feasible_now_actions
                            )
                            no_progress_same_edge = (
                                pending_age >= self.decision_engine.pending_progress_timeout_steps
                                and current_edge == pending.decision_edge
                            )
                            if wrong_lane_commit or no_progress_same_edge:
                                decision_metrics["same_edge_pending_released_no_progress"] += 1
                                pending_decisions.pop(vehicle_id, None)
                                lane_change_cooldown_until[(vehicle_id, current_edge)] = step + self.decision_engine.cooldown_steps

                        context = self.decision_engine.build_context(
                            vehicle_id,
                            current_edge,
                            vehicle.destination,
                            step,
                            snapshot=snapshot,
                        )
                        if vehicle_id in pending_decisions:
                            decision_metrics["decisions_skipped"] += 1
                            prev_edge_by_vehicle[vehicle_id] = current_edge
                            continue

                        state = self.encode_state(
                            vehicle_id,
                            current_edge,
                            vehicle.destination,
                            context=context,
                            vehicle=vehicle,
                            step=step,
                            snapshot=snapshot,
                        )
                        cooldown_until = lane_change_cooldown_until.get((vehicle_id, current_edge), -1)
                        action_source = "forced" if context.forced_action is not None else ""
                        if context.forced_action is not None:
                            action = context.forced_action
                            decision_metrics["forced_actions"] += 1
                        elif context.skip_reason:
                            decision_metrics["decisions_skipped"] += 1
                            prev_edge_by_vehicle[vehicle_id] = current_edge
                            continue
                        elif not self.decision_engine.is_decision_open(context):
                            prev_edge_by_vehicle[vehicle_id] = current_edge
                            continue
                        else:
                            # available_actions = executor/safety feasibility set (unchanged semantics).
                            # policy_actions = stricter learning-time subset to reduce harmful overrides.
                            # Fallback machinery below remains the final safety layer.
                            cooldown_active = step < cooldown_until
                            policy_actions = self._policy_action_candidates(
                                context=context,
                                recent_history=list(recent_edge_history[vehicle_id]),
                                cooldown_active=cooldown_active,
                                destination=vehicle.destination,
                            )
                            removed_actions = max(len(context.available_actions) - len(policy_actions), 0)
                            decision_metrics["policy_masked_actions_removed"] += removed_actions
                            action, action_source = self.trainer.select_action(
                                state, policy_actions, return_source=True
                            )
                            if action is None:
                                decision_metrics["decisions_skipped"] += 1
                                prev_edge_by_vehicle[vehicle_id] = current_edge
                                continue
                            if action_source == "explore":
                                decision_metrics["exploration_actions"] += 1
                            elif action_source == "policy":
                                decision_metrics["policy_actions"] += 1

                        next_edge = self.decision_engine.get_next_edge(current_edge, action)
                        if next_edge is None:
                            decision_metrics["safety_overrides"] += 1
                            decision_metrics["loop_avoidance_overrides"] += 1
                            prev_edge_by_vehicle[vehicle_id] = current_edge
                            continue

                        safe_ok, safety_details = self.decision_engine.prefilter_action_for_loops(
                            context=context,
                            action_idx=action,
                            destination=vehicle.destination,
                            recent_history=list(recent_edge_history[vehicle_id]),
                            distance_fn=self.get_distance_to_destination,
                        )
                        if not safe_ok:
                            original_action = action
                            decision_metrics["loop_override_count"] += 1
                            decision_metrics["loop_prefilter_overrides"] += 1
                            if safety_details.get("dead_end_reentry"):
                                decision_metrics["dead_end_reentry_override_count"] += 1
                            action = self._select_fallback_action(
                                context,
                                blocked_action=action,
                                destination=vehicle.destination,
                                recent_history=list(recent_edge_history[vehicle_id]),
                            )
                            if action is None:
                                prev_edge_by_vehicle[vehicle_id] = current_edge
                                continue
                            decision_metrics["fallback_overrides"] += 1
                            decision_metrics["safety_overrides"] += 1
                            action_source = "loop_prefilter_fallback"
                            override_penalty = self._clip_reward(self.loop_trap_override_penalty)
                            policy_actions_after_override = self._policy_action_candidates(
                                context=context,
                                recent_history=list(recent_edge_history[vehicle_id]),
                                cooldown_active=step < lane_change_cooldown_until.get((vehicle_id, current_edge), -1),
                                destination=vehicle.destination,
                            )
                            self.trainer.remember(
                                state,
                                original_action,
                                override_penalty,
                                state,
                                False,
                                next_valid_actions=policy_actions_after_override,
                                metadata={
                                    "override_learning": True,
                                    "override_type": "loop_prefilter_fallback",
                                    "original_action": original_action,
                                    "fallback_action": action,
                                    "override_cause": "loop_prefilter_fallback",
                                },
                            )
                            decision_metrics["override_learning_transitions"] += 1
                            decision_metrics["override_learning_negative"] += 1
                            imitation_reward = self._clip_reward(0.15)
                            self.trainer.remember(
                                state,
                                action,
                                imitation_reward,
                                state,
                                False,
                                next_valid_actions=policy_actions_after_override,
                                metadata={
                                    "override_learning": True,
                                    "override_type": "loop_prefilter_fallback",
                                    "original_action": original_action,
                                    "fallback_action": action,
                                    "imitation_credit": True,
                                },
                            )
                            decision_metrics["override_learning_transitions"] += 1
                            decision_metrics["override_learning_imitation"] += 1
                            episode_return += override_penalty
                            next_edge = self.decision_engine.get_next_edge(current_edge, action)
                            if next_edge is None:
                                prev_edge_by_vehicle[vehicle_id] = current_edge
                                continue

                        lane_change_requested = False
                        if action not in context.lane_feasible_now_actions:
                            if step < cooldown_until:
                                original_action = action
                                decision_metrics["cooldown_replans_blocked"] += 1
                                decision_metrics["cooldown_fallback_overrides"] += 1
                                fallback_action = self._select_fallback_action(
                                    context,
                                    blocked_action=action,
                                    destination=vehicle.destination,
                                    recent_history=list(recent_edge_history[vehicle_id]),
                                )
                                if fallback_action is None:
                                    prev_edge_by_vehicle[vehicle_id] = current_edge
                                    continue
                                action = fallback_action
                                action_source = "cooldown_fallback"
                                decision_metrics["fallback_overrides"] += 1
                                decision_metrics["fallback_to_lane_feasible_now"] += 1
                                override_penalty = self._clip_reward(self.same_edge_repeat_chase_penalty)
                                policy_actions_after_override = self._policy_action_candidates(
                                    context=context,
                                    recent_history=list(recent_edge_history[vehicle_id]),
                                    cooldown_active=True,
                                    destination=vehicle.destination,
                                )
                                self.trainer.remember(
                                    state,
                                    original_action,
                                    override_penalty,
                                    state,
                                    False,
                                    next_valid_actions=policy_actions_after_override,
                                    metadata={
                                        "override_learning": True,
                                        "override_type": "cooldown_fallback",
                                        "original_action": original_action,
                                        "fallback_action": action,
                                        "override_cause": "cooldown_fallback",
                                    },
                                )
                                decision_metrics["override_learning_transitions"] += 1
                                decision_metrics["override_learning_negative"] += 1
                                imitation_reward = self._clip_reward(0.10)
                                self.trainer.remember(
                                    state,
                                    action,
                                    imitation_reward,
                                    state,
                                    False,
                                    next_valid_actions=policy_actions_after_override,
                                    metadata={
                                        "override_learning": True,
                                        "override_type": "cooldown_fallback",
                                        "original_action": original_action,
                                        "fallback_action": action,
                                        "imitation_credit": True,
                                    },
                                )
                                decision_metrics["override_learning_transitions"] += 1
                                decision_metrics["override_learning_imitation"] += 1
                                episode_return += override_penalty
                            else:
                                lane_change_requested, lane_change_ok = self.decision_engine.try_request_lane_change(context, action)
                                decision_metrics["lane_change_attempts"] += 1
                                if lane_change_ok:
                                    decision_metrics["lane_change_success"] += 1
                                else:
                                    decision_metrics["lane_change_fail"] += 1
                                decision_metrics["lane_change_observe_started"] += 1
                                decision_metrics["deferred_lane_change_actions"] += 1
                                observe_metadata = self.decision_engine.start_lane_change_observe(
                                    context, action, step, lane_change_requested, lane_change_ok
                                )
                                pending_decisions[vehicle_id] = PendingDecision(
                                    state=state,
                                    intended_action=action,
                                    intended_next_edge=next_edge,
                                    decision_edge=current_edge,
                                    decision_step=step,
                                    last_credit_edge=current_edge,
                                    last_credit_step=step,
                                    destination=vehicle.destination,
                                    context=context,
                                    lane_change_requested=lane_change_requested,
                                    route_fragment=[],
                                    metadata={"action_source": action_source, **observe_metadata},
                                )
                                decision_metrics["decisions_opened"] += 1
                                prev_edge_by_vehicle[vehicle_id] = current_edge
                                continue

                        full_route, committed_next_edge, apply_error = self.decision_engine.apply_route_decision(
                            vehicle_id,
                            current_edge,
                            action,
                            vehicle.destination,
                        )
                        if apply_error:
                            decision_metrics["route_apply_fail"] += 1
                            decision_metrics["route_apply_fail_overrides"] += 1
                            decision_metrics["fragment_build_failures"] += 1
                            override_penalty = self._clip_reward(-6.0)
                            policy_actions_after_override = self._policy_action_candidates(
                                context=context,
                                recent_history=list(recent_edge_history[vehicle_id]),
                                cooldown_active=step < lane_change_cooldown_until.get((vehicle_id, current_edge), -1),
                                destination=vehicle.destination,
                            )
                            self.trainer.remember(
                                state,
                                action,
                                override_penalty,
                                state,
                                False,
                                next_valid_actions=policy_actions_after_override,
                                metadata={
                                    "override_learning": True,
                                    "override_type": "route_apply_failure",
                                    "original_action": action,
                                    "fallback_action": None,
                                    "override_cause": "route_apply_failure",
                                    "route_apply_failed": True,
                                },
                            )
                            decision_metrics["override_learning_transitions"] += 1
                            decision_metrics["override_learning_negative"] += 1
                            episode_return += override_penalty
                            prev_edge_by_vehicle[vehicle_id] = current_edge
                            continue
                        last_planned_terminal_edge_by_vehicle[vehicle_id] = full_route[-1] if full_route else vehicle.destination

                        if vehicle_id in pending_decisions:
                            decision_metrics["decisions_superseded"] += 1
                        pending_decisions[vehicle_id] = PendingDecision(
                            state=state,
                            intended_action=action,
                            intended_next_edge=committed_next_edge,
                            decision_edge=current_edge,
                            decision_step=step,
                            last_credit_edge=current_edge,
                            last_credit_step=step,
                            destination=vehicle.destination,
                            context=context,
                            lane_change_requested=lane_change_requested,
                            route_fragment=list(full_route[1:]) if full_route else [],
                            metadata={"action_source": action_source, "lane_change_deferrals": lane_change_deferrals.get(vehicle_id, 0)},
                        )
                        lane_change_deferrals[vehicle_id] = 0
                        decision_metrics["decisions_opened"] += 1
                        prev_edge_by_vehicle[vehicle_id] = current_edge

                    simulation_step()

                    arrived_this_step = set(simulation_get_arrived_ids())
                    teleported_ids = self.get_teleport_ids()
                    live_after_step = set(vehicle_get_ids())
                    removed_controlled_ids = {
                        vid for vid in controlled_live_ids
                        if vid not in live_after_step and vid in vehicles
                    }

                    if teleported_ids:
                        episode_teleport_events += len(teleported_ids)
                        decision_metrics["teleports"] += len(teleported_ids)
                        teleported_controlled_ids.update(tid for tid in teleported_ids if tid in vehicles)

                    for removed_id in sorted(removed_controlled_ids):
                        if removed_id in final_outcome_by_vehicle:
                            continue
                        vehicle = vehicles[removed_id]
                        last_seen_edge = last_seen_edge_by_vehicle.get(removed_id, "<unknown>")
                        outcome = self._classify_terminal_outcome(
                            removed_id,
                            arrived_this_step,
                            teleported_ids,
                            is_live=(removed_id in live_after_step),
                        )
                        final_outcome_by_vehicle[removed_id] = outcome
                        reached_global_destination = (outcome == "global_arrival")
                        if reached_global_destination:
                            arrived_ids.add(removed_id)
                            if removed_id not in completed_travel_time_ids:
                                completed_travel_time_ids.add(removed_id)
                                completed_travel_times.append(max(float(step) - float(vehicle.start_time), 0.0))
                            if last_seen_edge != vehicle.destination:
                                arrived_with_prestep_edge_not_destination += 1

                        arrived_debug_records.append({
                            "vehicle_id": removed_id,
                            "global_destination": vehicle.destination,
                            "last_seen_edge": last_seen_edge,
                            "observed_destination_edge": reached_global_destination,
                            "last_planned_terminal_edge": last_planned_terminal_edge_by_vehicle.get(removed_id, "<unset>"),
                            "reached_global_destination": reached_global_destination,
                            "terminal_outcome": outcome,
                        })
                        episode_return += self._finalize_terminal_transition(
                            vehicle_id=removed_id,
                            vehicle=vehicle,
                            outcome=outcome,
                            step=step,
                            pending_decisions=pending_decisions,
                            terminal_recorded_ids=terminal_recorded_ids,
                            last_snapshot_by_vehicle=last_snapshot_by_vehicle,
                            last_seen_edge_by_vehicle=last_seen_edge_by_vehicle,
                            decision_metrics=decision_metrics,
                            decision_latency_steps=decision_latency_steps,
                            finalized_decision_rewards=finalized_decision_rewards,
                            decision_debug_rows=decision_debug_rows,
                            episode=episode,
                            in_arrived_ids=(removed_id in arrived_this_step),
                            in_teleport_ids=(removed_id in teleported_ids),
                        )
                        self.cleanup_vehicle_state(
                            removed_id,
                            pending_decisions,
                            prev_edge_by_vehicle,
                            last_seen_edge_by_vehicle,
                            last_planned_terminal_edge_by_vehicle,
                            recent_edge_history,
                            last_snapshot_by_vehicle,
                        )

                    if step % self.train_every == 0:
                        for _ in range(self.grad_steps):
                            self.trainer.replay()

                    if step % self.step_log_every == 0:
                        self._print_step_progress(
                            episode=episode,
                            step=step,
                            total_controlled=total_controlled,
                            arrived_ids=arrived_ids,
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
                if last_step_executed >= (MAX_SIMULATION_STEPS - 1):
                    alive_at_step_cap_ids = {vid for vid in vehicle_get_ids() if vid in vehicles}
                    for vid in alive_at_step_cap_ids:
                        if vid not in final_outcome_by_vehicle:
                            final_outcome_by_vehicle[vid] = "alive_at_step_cap"
                        timeout_vehicle = vehicles[vid]
                        episode_return += self._finalize_terminal_transition(
                            vehicle_id=vid,
                            vehicle=timeout_vehicle,
                            outcome="timeout",
                            step=last_step_executed,
                            pending_decisions=pending_decisions,
                            terminal_recorded_ids=terminal_recorded_ids,
                            last_snapshot_by_vehicle=last_snapshot_by_vehicle,
                            last_seen_edge_by_vehicle=last_seen_edge_by_vehicle,
                            decision_metrics=decision_metrics,
                            decision_latency_steps=decision_latency_steps,
                            finalized_decision_rewards=finalized_decision_rewards,
                            decision_debug_rows=decision_debug_rows,
                            episode=episode,
                            in_arrived_ids=False,
                            in_teleport_ids=False,
                        )
                global_arrival_count = sum(1 for outcome in final_outcome_by_vehicle.values() if outcome == "global_arrival")
                terminal_teleport_count = sum(1 for outcome in final_outcome_by_vehicle.values() if outcome == "teleport")
                removed_nonarrival_count = sum(1 for outcome in final_outcome_by_vehicle.values() if outcome == "removed_nonarrival")
                alive_at_step_cap_count = sum(1 for outcome in final_outcome_by_vehicle.values() if outcome == "alive_at_step_cap")
                completion_rate = (
                    global_arrival_count / float(total_controlled)
                    if total_controlled > 0 else 0.0
                )
                avg_travel_time = float(np.mean(completed_travel_times)) if completed_travel_times else 0.0
                p50_travel_time = float(np.percentile(completed_travel_times, 50)) if completed_travel_times else 0.0
                p90_travel_time = float(np.percentile(completed_travel_times, 90)) if completed_travel_times else 0.0
                avg_return = episode_return / float(total_controlled) if total_controlled > 0 else 0.0
                mean_pending_age = (
                    float(np.mean(pending_age_samples)) if pending_age_samples else 0.0
                )
                mean_decision_latency_steps = (
                    float(np.mean(decision_latency_steps)) if decision_latency_steps else 0.0
                )
                mean_reward_per_finalized_decision = (
                    float(np.mean(finalized_decision_rewards)) if finalized_decision_rewards else 0.0
                )
                mean_route_difficulty_eta = (
                    float(np.mean(route_difficulty_etas)) if route_difficulty_etas else 0.0
                )
                p50_route_difficulty_eta = (
                    float(np.percentile(route_difficulty_etas, 50)) if route_difficulty_etas else 0.0
                )
                p90_route_difficulty_eta = (
                    float(np.percentile(route_difficulty_etas, 90)) if route_difficulty_etas else 0.0
                )

                rolling_teleport_events.append(float(episode_teleport_events))
                rolling_teleported_controlled.append(float(len(teleported_controlled_ids)))
                rolling_completion_rate.append(float(completion_rate))
                rolling_avg_return.append(float(avg_return))
                rolling_avg_travel_time.append(float(avg_travel_time))
                rolling_mismatch.append(float(decision_metrics["route_mismatch"]))

                roll_tele_events = sum(rolling_teleport_events) / len(rolling_teleport_events)
                roll_tele_ctrl = sum(rolling_teleported_controlled) / len(rolling_teleported_controlled)
                roll_completion = sum(rolling_completion_rate) / len(rolling_completion_rate)
                roll_return = sum(rolling_avg_return) / len(rolling_avg_return)
                roll_avg_travel_time = sum(rolling_avg_travel_time) / len(rolling_avg_travel_time)
                roll_mismatch = sum(rolling_mismatch) / len(rolling_mismatch)

                self.trainer.epsilon = max(
                    self.trainer.epsilon_min,
                    self.trainer.epsilon * self.trainer.epsilon_decay
                )
                print(
                    f"\n[EP {episode:03d} DONE] eps={self.trainer.epsilon:.4f} train={self.trainer.train_steps} "
                    f"replay={len(self.trainer.memory)} ret={avg_return:.3f} "
                    f"done={global_arrival_count}/{total_controlled} "
                    f"failed={max(total_controlled-global_arrival_count,0)} avg_tt={avg_travel_time:.2f} "
                    f"p50_tt={p50_travel_time:.2f} p90_tt={p90_travel_time:.2f}"
                )
                print(
                    f"  rolling({len(rolling_teleport_events)}): completion={roll_completion:.1%} "
                    f"avg_return={roll_return:.3f} avg_tt={roll_avg_travel_time:.2f} teleports/ep={roll_tele_events:.2f} "
                    f"teleported_ctrl/ep={roll_tele_ctrl:.2f} mismatch/ep={roll_mismatch:.2f}"
                )
                print(
                    "  decisions: opened={:.0f} finalized={:.0f} skipped={:.0f} forced={:.0f} superseded={:.0f} "
                    "lane_change(a/s/f)={:.0f}/{:.0f}/{:.0f} loops={:.0f} uturn={:.0f} "
                    "short_cycle={:.0f} aba={:.0f} dead_end_reentry={:.0f} "
                    "apply_fail={:.0f} overrides={:.0f} override_ratio={:.1%}".format(
                        decision_metrics["decisions_opened"],
                        decision_metrics["decisions_finalized"],
                        decision_metrics["decisions_skipped"],
                        decision_metrics["forced_actions"],
                        decision_metrics["decisions_superseded"],
                        decision_metrics["lane_change_attempts"],
                        decision_metrics["lane_change_success"],
                        decision_metrics["lane_change_fail"],
                        decision_metrics["loop_events"],
                        decision_metrics["uturn_events"],
                        decision_metrics["short_cycle_events"],
                        decision_metrics["aba_bounce_events"],
                        decision_metrics["dead_end_reentry_events"],
                        decision_metrics["route_apply_fail"],
                        (
                            decision_metrics["safety_overrides"]
                            + decision_metrics["fallback_overrides"]
                            + decision_metrics["route_apply_fail"]
                        ),
                        (
                            decision_metrics["safety_overrides"]
                            + decision_metrics["fallback_overrides"]
                            + decision_metrics["route_apply_fail"]
                        ) / max(decision_metrics["decisions_opened"], 1.0),
                    )
                )
                print(
                    "  policy mix: explore={:.0f} policy={:.0f} deferred_lane_change={:.0f} fallback={:.0f}".format(
                        decision_metrics["exploration_actions"],
                        decision_metrics["policy_actions"],
                        decision_metrics["deferred_lane_change_actions"],
                        decision_metrics["fallback_overrides"],
                    )
                )

                exited_without_destination = removed_nonarrival_count
                arrived_non_global_target = 0
                print(
                    f"Controlled exit diagnostics | "
                    f"arrived={global_arrival_count}/{total_controlled}, "
                    f"teleported_terminal={terminal_teleport_count}/{total_controlled}, "
                    f"removed_nonarrival={removed_nonarrival_count}/{total_controlled}, "
                    f"alive_at_step_cap={alive_at_step_cap_count}/{total_controlled}"
                )
                print(
                    "  terminal_outcomes: "
                    f"global_arrival={global_arrival_count} "
                    f"teleport={terminal_teleport_count} "
                    f"removed_nonarrival={removed_nonarrival_count} "
                    f"alive_at_step_cap={alive_at_step_cap_count} "
                    f"arrived_with_prestep_edge_not_destination={arrived_with_prestep_edge_not_destination}"
                )

                if self.debug_exit_diagnostics:
                    mismatched_arrivals = [
                        record for record in arrived_debug_records
                        if (
                            record["terminal_outcome"] == "global_arrival"
                            and record["last_seen_edge"] != record["global_destination"]
                        )
                    ]
                    print(
                        f"Arrival debug | total_arrived={len(arrived_debug_records)}, "
                        f"arrived_with_prestep_edge_not_destination={len(mismatched_arrivals)}"
                    )

                    for record in mismatched_arrivals[:self.debug_exit_diagnostics_limit]:
                        print(
                            "  ARRIVED_PRESTEP_EDGE_MISMATCH "
                            f"vehicle={record['vehicle_id']} "
                            f"last_seen_edge={record['last_seen_edge']} "
                            f"last_planned_terminal_edge={record['last_planned_terminal_edge']} "
                            f"global_destination={record['global_destination']}"
                        )

                    if len(mismatched_arrivals) > self.debug_exit_diagnostics_limit:
                        print(
                            "  ARRIVED_PRESTEP_EDGE_MISMATCH ... "
                            f"{len(mismatched_arrivals) - self.debug_exit_diagnostics_limit} more vehicles"
                        )

                    removed_nonarrival_ids = sorted(
                        [vid for vid, outcome in final_outcome_by_vehicle.items() if outcome == "removed_nonarrival"]
                    )
                    for vehicle_id in removed_nonarrival_ids[:self.debug_exit_diagnostics_limit]:
                        vehicle = vehicles[vehicle_id]
                        print(
                            "  EXITED_WITHOUT_DEST "
                            f"vehicle={vehicle_id} "
                            f"last_seen_edge={last_seen_edge_by_vehicle.get(vehicle_id, '<unknown>')} "
                            f"last_planned_terminal_edge={last_planned_terminal_edge_by_vehicle.get(vehicle_id, '<unset>')} "
                            f"global_destination={vehicle.destination}"
                        )

                    if len(removed_nonarrival_ids) > self.debug_exit_diagnostics_limit:
                        print(
                            "  EXITED_WITHOUT_DEST ... "
                            f"{len(removed_nonarrival_ids) - self.debug_exit_diagnostics_limit} more vehicles"
                        )
                for vid, pending in pending_decisions.items():
                    if vid in pending_debug_logged_ids:
                        continue
                    decision_debug_rows.append(self._build_decision_debug_row(
                        episode=episode,
                        step=last_step_executed,
                        pending=pending,
                        actual_next_edge="",
                        finalized=0,
                        finalize_delay_steps=max(last_step_executed - pending.decision_step, 0),
                    ))
                self._append_decision_debug_rows(decision_debug_rows)
                traci.close()
                with open(self.metrics_csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fields)
                    row = {
                        "episode": episode,
                        "epsilon": self.trainer.epsilon,
                        "replay": len(self.trainer.memory),
                        "train_steps": self.trainer.train_steps,
                        "mean_loss": self.trainer.last_loss if self.trainer.last_loss is not None else "",
                        "episode_return": avg_return,
                        "completion_rate": completion_rate,
                        "avg_travel_time": avg_travel_time,
                        "p50_travel_time": p50_travel_time,
                        "p90_travel_time": p90_travel_time,
                        "teleports": episode_teleport_events,
                        "teleported_controlled": len(teleported_controlled_ids),
                        "forced_actions": decision_metrics["forced_actions"],
                        "decisions_opened": decision_metrics["decisions_opened"],
                        "decisions_finalized": decision_metrics["decisions_finalized"],
                        "decisions_skipped": decision_metrics["decisions_skipped"],
                        "decisions_superseded": decision_metrics["decisions_superseded"],
                        "route_mismatch": decision_metrics["route_mismatch"],
                        "loop_events": decision_metrics["loop_events"],
                        "uturn_events": decision_metrics["uturn_events"],
                        "short_cycle_events": decision_metrics["short_cycle_events"],
                        "aba_bounce_events": decision_metrics["aba_bounce_events"],
                        "dead_end_reentry_events": decision_metrics["dead_end_reentry_events"],
                        "long_horizon_loop_events": decision_metrics["long_horizon_loop_events"],
                        "revisit_without_progress_events": decision_metrics["revisit_without_progress_events"],
                        "safety_overrides": decision_metrics["safety_overrides"],
                        "loop_avoidance_overrides": decision_metrics["loop_avoidance_overrides"],
                        "distance_worsening_overrides": decision_metrics["distance_worsening_overrides"],
                        "fragment_build_failures": decision_metrics["fragment_build_failures"],
                        "fallback_overrides": decision_metrics["fallback_overrides"],
                        "fallback_to_lane_feasible_now": decision_metrics["fallback_to_lane_feasible_now"],
                        "pending_decision_timeouts": decision_metrics["pending_decision_timeouts"],
                        "deferred_lane_change_actions": decision_metrics["deferred_lane_change_actions"],
                        "lane_change_observe_started": decision_metrics["lane_change_observe_started"],
                        "lane_change_observe_success": decision_metrics["lane_change_observe_success"],
                        "lane_change_observe_abort_no_progress": decision_metrics["lane_change_observe_abort_no_progress"],
                        "lane_change_observe_abort_commit_window": decision_metrics["lane_change_observe_abort_commit_window"],
                        "same_edge_pending_released_no_progress": decision_metrics["same_edge_pending_released_no_progress"],
                        "cooldown_replans_blocked": decision_metrics["cooldown_replans_blocked"],
                        "loop_override_count": decision_metrics["loop_override_count"],
                        "dead_end_reentry_override_count": decision_metrics["dead_end_reentry_override_count"],
                        "batched_policy_calls": decision_metrics["batched_policy_calls"],
                        "snapshot_cache_hits": self._cache_metrics["snapshot_cache_hits"],
                        "shortest_path_cache_hits": self._cache_metrics["shortest_path_cache_hits"],
                        "exploration_actions": decision_metrics["exploration_actions"],
                        "policy_actions": decision_metrics["policy_actions"],
                        "override_ratio": (
                            decision_metrics["safety_overrides"]
                            + decision_metrics["fallback_overrides"]
                            + decision_metrics["route_apply_fail"]
                        ) / max(decision_metrics["decisions_opened"], 1.0),
                        "policy_masked_actions_removed": decision_metrics["policy_masked_actions_removed"],
                        "override_learning_transitions": decision_metrics["override_learning_transitions"],
                        "loop_prefilter_overrides": decision_metrics["loop_prefilter_overrides"],
                        "cooldown_fallback_overrides": decision_metrics["cooldown_fallback_overrides"],
                        "observe_abort_fallback_overrides": decision_metrics["observe_abort_fallback_overrides"],
                        "route_apply_fail_overrides": decision_metrics["route_apply_fail_overrides"],
                        "override_learning_negative": decision_metrics["override_learning_negative"],
                        "override_learning_imitation": decision_metrics["override_learning_imitation"],
                        "timeout_unfinished_controlled": alive_at_step_cap_count,
                        "exited_without_destination": exited_without_destination,
                        "arrived_non_global_target": arrived_non_global_target,
                        "alive_at_step_cap": alive_at_step_cap_count,
                        "decision_pending_at_episode_end": len(pending_decisions),
                        "mean_pending_age": mean_pending_age,
                        "mean_decision_latency_steps": mean_decision_latency_steps,
                        "mean_reward_per_finalized_decision": mean_reward_per_finalized_decision,
                        "mean_route_difficulty_eta": mean_route_difficulty_eta,
                        "p50_route_difficulty_eta": p50_route_difficulty_eta,
                        "p90_route_difficulty_eta": p90_route_difficulty_eta,
                        "fail_teleport": terminal_teleport_count,
                        "fail_timeout": decision_metrics["fail_timeout"],
                        "fail_removed_non_destination": decision_metrics["fail_removed_non_destination"],
                        "fail_unreachable_transition": decision_metrics["fail_unreachable_transition"],
                        "fail_dead_end_no_outgoing": decision_metrics["fail_dead_end_no_outgoing"],
                    }
                    if set(row.keys()) != set(csv_fields):
                        raise ValueError("rl_episode_metrics.csv row schema does not match header")
                    writer.writerow(row)

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
            [counts[e] / max(lengths.get(e, 1e-6), 1e-6) for e in edge_list],
            dtype=np.float32
        )
        if len(self._density_vec) > 0:
            self._density_mean = float(np.mean(self._density_vec))
            self._density_std = float(np.std(self._density_vec))
        else:
            self._density_mean = 0.0
            self._density_std = 0.0
        self._last_density_step = step
