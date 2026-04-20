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
from core.strategic_options import ActiveStrategicDecision, ReplayManager, StrategicTransition
from core.tactical_option_executor import TacticalOptionExecutor

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



class StrategicOptionTrainer:
    """Option-scoring trainer with shared-state encoder and per-option MLP scorer."""

    def __init__(self, shared_state_size, option_feature_size, learning_rate=0.001, gamma=0.97, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01, batch_size=64, replay_capacity=100000, replay_warmup=2000):
        from keras.layers import Input, Dense, Concatenate
        from keras.models import Model

        self.shared_state_size = int(shared_state_size)
        self.option_feature_size = int(option_feature_size)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.batch_size = int(batch_size)
        self.replay_warmup = int(max(replay_warmup, batch_size))

        shared_in = Input(shape=(self.shared_state_size,), name="shared_state")
        option_in = Input(shape=(self.option_feature_size,), name="option_features")
        shared_emb = Dense(128, activation="relu", name="shared_encoder_1")(shared_in)
        shared_emb = Dense(64, activation="relu", name="shared_encoder_2")(shared_emb)
        fused = Concatenate(name="fused_shared_option")([shared_emb, option_in])
        h = Dense(64, activation="relu", name="option_scorer_1")(fused)
        h = Dense(32, activation="relu", name="option_scorer_2")(h)
        score_out = Dense(1, activation="linear", name="option_score")(h)
        self.model = Model(inputs=[shared_in, option_in], outputs=score_out)
        self.model.compile(loss=Huber(delta=1.0), optimizer=Adam(learning_rate=learning_rate))

        self.replay = ReplayManager(strategic_capacity=replay_capacity)
        self.last_sample_fractions = {"strategic_fraction": 1.0, "tactical_failure_fraction": 0.0, "terminal_only_fraction": 0.0}
        self.last_sample_counts = {"strategic": 0, "tactical_failure": 0, "terminal_only": 0, "total": 0}

    def score_options(self, shared_state, option_features):
        if not option_features:
            return np.array([], dtype=np.float32)
        n = len(option_features)
        shared_batch = np.repeat(np.asarray(shared_state, dtype=np.float32), n, axis=0)
        option_batch = np.asarray(option_features, dtype=np.float32)
        scores = self.model.predict([shared_batch, option_batch], verbose=0).reshape(-1)
        return scores

    def select_option(self, shared_state, option_features):
        if not option_features:
            return None, "none"
        if np.random.rand() <= self.epsilon:
            return int(np.random.randint(0, len(option_features))), "explore"
        scores = self.score_options(shared_state, option_features)
        return int(np.argmax(scores)), "policy"

    def _target_value(self, next_shared, next_option_features, done):
        if done or next_shared is None or not next_option_features:
            return 0.0
        scores = self.score_options(next_shared, next_option_features)
        return float(np.max(scores)) if len(scores) else 0.0

    def train_step(self):
        if len(self.replay.strategic_replay) < self.replay_warmup:
            return None
        batch, fractions = self.replay.sample_for_training(self.batch_size)
        self.last_sample_fractions = fractions
        self.last_sample_counts = {
            "strategic": int(fractions.get("strategic_count", 0)),
            "tactical_failure": int(fractions.get("tactical_failure_count", 0)),
            "terminal_only": int(fractions.get("terminal_only_count", 0)),
            "total": int(fractions.get("total_count", len(batch))),
        }
        losses = []
        for tr in batch:
            if tr.chosen_option_idx >= len(tr.state_option_features):
                continue
            state_shared = np.asarray(tr.state_shared, dtype=np.float32)
            option_feat = np.asarray(tr.state_option_features[tr.chosen_option_idx], dtype=np.float32)
            target = float(tr.aggregated_reward) + (0.0 if tr.done else self.gamma * self._target_value(tr.next_state_shared, tr.next_state_option_features, tr.done))
            loss = self.model.train_on_batch([state_shared, option_feat], np.array([target], dtype=np.float32))
            if loss is not None:
                losses.append(float(loss))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return float(np.mean(losses)) if losses else None

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
        epsilon_min=0.01,
        replay_capacity=100000,
        elite_replay_capacity=None,
        elite_fraction=0.25,
        batch_size=128,
        replay_warmup=10000,
        target_update_every=400,
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
        self.elite_fraction = float(np.clip(elite_fraction, 0.0, 0.5))
        elite_capacity = elite_replay_capacity if elite_replay_capacity is not None else max(replay_capacity // 4, batch_size * 4)
        self.elite_memory = ReplayBuffer(elite_capacity)
        self.elite_transitions_added = 0
        self.elite_samples_drawn = 0
        self.replay_main_kept_finalized = 0
        self.replay_main_kept_pending_timeout = 0
        self.replay_main_kept_terminal = 0
        self.replay_main_kept_other = 0
        self.replay_main_dropped = 0
        self.staged_episode_transitions = []
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
    
    def _is_elite_transition(self, reward, done, metadata):
        metadata = metadata or {}
        if metadata.get("override_learning", False):
            return False
        if metadata.get("imitation_credit", False):
            return False
        if metadata.get("synthetic_terminal_no_pending", False):
            return False
        terminal_outcome = metadata.get("terminal_outcome")
        if terminal_outcome in {"teleport", "timeout", "removed_nonarrival"}:
            return False
        if metadata.get("episode_bucket") != "good":
            return False
        if not metadata.get("decision_open", False):
            return False
        if int(metadata.get("available_count", 0)) < 2:
            return False
        if metadata.get("forced_action", False):
            return False
        if terminal_outcome == "global_arrival":
            return True
        return bool(
            metadata.get("decision_finalized", False)
            and (not metadata.get("mismatch", False))
            and float(reward) >= 0.0
        )

    def _should_store_main_transition(self, reward, done, metadata):
        metadata = metadata or {}
        if metadata.get("override_learning", False):
            return False
        if metadata.get("imitation_credit", False):
            return False
        if metadata.get("synthetic_terminal_no_pending", False):
            return False

        terminal_outcome = metadata.get("terminal_outcome")
        if terminal_outcome in {"global_arrival", "teleport", "timeout", "removed_nonarrival"}:
            return True

        if not metadata.get("decision_open", False):
            return False
        if int(metadata.get("available_count", 0)) < 2:
            return False
        if metadata.get("forced_action", False):
            return False

        is_finalized = bool(metadata.get("decision_finalized", False))
        is_timeout_or_observe = bool(
            metadata.get("pending_timeout_replan", False)
            or metadata.get("observe_no_progress", False)
            or metadata.get("observe_low_speed", False)
            or metadata.get("observe_commit_window_miss", False)
        )
        if not (is_finalized or is_timeout_or_observe):
            return False

        bucket = str(metadata.get("episode_bucket", "bad"))
        if bucket == "good":
            keep_prob = 1.0 if is_finalized else 0.75
        elif bucket == "okay":
            keep_prob = 0.70 if is_finalized else 0.40
        else:
            keep_prob = 0.35 if is_finalized else 0.15
        return random.random() < keep_prob

    def stage_transition(self, state, action, reward, next_state, done, next_valid_actions=None, metadata=None):
        self.staged_episode_transitions.append(
            (state, action, reward, next_state, done, next_valid_actions, metadata)
        )

    def _episode_bucket(self, avg_tt, avg_return, completion_rate, teleported_controlled):
        if avg_tt < 210 and avg_return >= 9 and completion_rate >= 0.98 and teleported_controlled <= 3:
            return "good"
        if avg_tt < 230 and avg_return >= 4 and completion_rate >= 0.95 and teleported_controlled <= 6:
            return "okay"
        return "bad"

    def flush_staged_episode(self, avg_tt, avg_return, completion_rate, teleported_controlled):
        bucket = self._episode_bucket(avg_tt, avg_return, completion_rate, teleported_controlled)
        for state, action, reward, next_state, done, next_valid_actions, metadata in self.staged_episode_transitions:
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            metadata["episode_bucket"] = bucket
            if self._should_store_main_transition(reward, done, metadata):
                self.memory.add(state, action, reward, next_state, done, next_valid_actions, metadata=metadata)
                terminal_outcome = metadata.get("terminal_outcome")
                if terminal_outcome in {"global_arrival", "teleport", "timeout", "removed_nonarrival"}:
                    self.replay_main_kept_terminal += 1
                elif metadata.get("decision_finalized", False):
                    self.replay_main_kept_finalized += 1
                elif metadata.get("pending_timeout_replan", False):
                    self.replay_main_kept_pending_timeout += 1
                else:
                    self.replay_main_kept_other += 1
            else:
                self.replay_main_dropped += 1
            if self._is_elite_transition(reward, done, metadata):
                self.elite_memory.add(state, action, reward, next_state, done, next_valid_actions, metadata=metadata)
                self.elite_transitions_added += 1
        self.staged_episode_transitions.clear()

    def remember(self, state, action, reward, next_state, done, next_valid_actions=None, metadata=None):
        """
        Store 1 transition for replay
        """
        self.stage_transition(state, action, reward, next_state, done, next_valid_actions, metadata)
    
    def replay(self):
        """
        Train the Q-network from replayed experiences. Update q-values of previous state based on the most recent one.
        """
        if len(self.memory) < self.replay_warmup:
            return
        elite_bs = 0
        if len(self.elite_memory) >= max(8, self.batch_size // 8):
            elite_bs = min(int(round(self.batch_size * self.elite_fraction)), len(self.elite_memory))
        base_bs = self.batch_size - elite_bs
        base_bs = min(base_bs, len(self.memory))
        if base_bs <= 0:
            return
        minibatch = self.memory.sample(base_bs)
        if elite_bs > 0:
            elite_batch = self.elite_memory.sample(elite_bs)
            minibatch += elite_batch
            self.elite_samples_drawn += elite_bs
        random.shuffle(minibatch)
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
        epsilon_min=0.01,
        gamma=0.97,
        replay_capacity=100000,
        batch_size=128,
        replay_warmup=10000,
        train_every=6,
        grad_steps=1,
        rolling_window=100,
        use_double_dqn=True,
        target_pattern=3,
        debug_exit_diagnostics=False,
        debug_exit_diagnostics_limit=20,
        step_log_every=100,
        density_refresh_every=4,
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
        self.system_congestion_scale = 0.04
        self.selfless_reward_scale = 0.35
        self.selfless_reward_clip = 3.0
        self.loop_window = 12
        self.loop_repeat_penalty = 1.5
        # Objective priority:
        # 1) minimize travel time (dominant)
        # 2) congestion externality (secondary)
        # 3) shortest-path distance as tie-breaker
        self.travel_time_penalty = 0.05
        self.eta_progress_scale = 0.65
        self.distance_tiebreak_scale = 0.06
        self.score_slack = 30.0
        self.reward_clip_low = -20.0
        self.reward_clip_high = 20.0
        self.pending_timeout_penalty = -18.0
        self.pending_latency_penalty_per_step = 0.008
        self.pending_replan_penalty = -0.4
        self.stale_disappeared_penalty = -14.0
        self.observe_no_progress_penalty = -0.5
        self.observe_low_speed_penalty = -0.4
        self.observe_commit_window_miss_penalty = -0.7
        self.same_edge_repeat_chase_penalty = -0.5
        self.fallback_missed_lane_penalty = -0.4
        self.loop_trap_override_penalty = -1.4
        self.tail_delay_threshold_eta_mult = 1.35
        self.tail_delay_threshold_min_steps = 180.0
        self.tail_delay_threshold_max_steps = 320.0
        self.tail_delay_linear_penalty = 0.06
        self.tail_delay_quadratic_penalty = 0.00012
        self.tail_arrival_penalty_per_25_steps = 0.75
        self.tail_arrival_penalty_cap = 8.0

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
        self.tactical_executor = TacticalOptionExecutor(self.decision_engine)

        # state = [edge_embedding, destination_embedding]
        #         + edge/lane/reachable/available feasibility masks (4*6)
        #         + commit flag + 3 lane features + 3 travel-time features
        #         + local congestion summary
        #         + per-action branch features (6 actions * 5 features)
        # NOTE: compact-state size changed; retraining is required.
        self.edge_embedding_dim = 8
        self.local_congestion_k = 6
        self._init_edge_embeddings(seed=1337)
        self.state_size = (2 * self.edge_embedding_dim) + 24 + 1 + 3 + 3 + self.local_congestion_k + 30
        self.option_feature_size = 18
        self.action_size = 6
        self.metrics_csv_path = os.path.join(self.sumocfg_dir, "rl_episode_metrics.csv")
        self._density_vec = np.zeros(len(self.connection_info.edge_list), dtype=np.float32)
        self._density_mean = 0.0
        self._density_std = 0.0
        self._density_p95 = 0.0
        # Density is now vehicles per 100m per lane; feature scale changed, retraining is required.
        self.density_scale_m = 100.0
        self._last_density_step = -10**9
        self._lane_length_cache = {}
        self._passenger_edge_set = set(self.connection_info.edge_list)
        self.congestion_density_threshold = 0.30
        self.congestion_low_speed_threshold = 2.0
        self.emergency_decel_threshold = 4.5
        self.teleport_jam_density_threshold = 0.55
        # Legacy DQN trainer remains for backward compatibility but is not used by the RL strategic path.
        self.trainer = DQNTrainer(
            self.state_size,
            self.action_size,
            gamma=gamma,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            replay_capacity=replay_capacity,
            elite_replay_capacity=max(replay_capacity // 4, batch_size * 4),
            elite_fraction=0.25,
            batch_size=batch_size,
            replay_warmup=replay_warmup,
            target_update_every=400,
            target_soft_tau=1.0,
            use_double_dqn=use_double_dqn,
        )
        self.strategic_trainer = StrategicOptionTrainer(
            shared_state_size=self.state_size,
            option_feature_size=self.option_feature_size,
            gamma=gamma,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            replay_warmup=replay_warmup,
        )
        self._strategic_main_allowed_tactical_outcomes = {
            "success",
            "commit_window_miss",
            "stalled_lane_change",
            "forced_by_lane_commit",
            "apply_route_failed",
            "became_impossible_after_selection",
        }
        print("[RL] Trainer path: option-based StrategicOptionTrainer (legacy DQN trainer disabled for run path)")
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
        mean_density_samples=None,
        congestion_high_pressure_steps=0,
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
        mean_density = (
            float(np.mean(mean_density_samples))
            if mean_density_samples else 0.0
        )
        skip_to_final = float(decision_metrics["decisions_skipped"]) / max(
            float(decision_metrics["decisions_finalized"]),
            1.0,
        )
        print(
            "  diagnostics: mean_density={:.4f} congested_steps={} pending_timeout={} "
            "lane_change(a/s/f)={:.0f}/{:.0f}/{:.0f} emergency_brake={} teleport(jam/yield)={:.0f}/{:.0f} "
            "skip_to_finalized={:.2f}".format(
                mean_density,
                int(congestion_high_pressure_steps),
                int(decision_metrics["pending_decision_timeouts"]),
                decision_metrics["lane_change_attempts"],
                decision_metrics["lane_change_success"],
                decision_metrics["lane_change_fail"],
                int(decision_metrics["emergency_brake_events"]),
                decision_metrics["teleport_inferred_jam"],
                decision_metrics["teleport_inferred_yield_or_deadlock"],
                skip_to_final,
            )
        )

    def _action_social_cost_proxy(self, current_edge, action_idx, destination):
        """
        Lower is better for "selfless" local decisions.
        Proxy combines downstream edge pressure + residual distance.
        """
        next_edge = self.decision_engine.get_next_edge(current_edge, action_idx)
        if next_edge is None:
            return float("inf")
        density = self._edge_density(next_edge)
        eta_proxy = self._estimate_remaining_eta(next_edge, destination)
        if not math.isfinite(eta_proxy):
            eta_proxy = float(MAX_SIMULATION_STEPS)
        return (1.25 * float(density)) + (0.01 * float(eta_proxy))

    def _edge_lane_count(self, edge_id):
        return max(len(self.connection_info.edge_lane_ids.get(edge_id, [])), 1)

    def _edge_lane_meters(self, edge_id):
        edge_len = max(float(self.connection_info.edge_length_dict.get(edge_id, 5.0)), 5.0)
        return edge_len * float(self._edge_lane_count(edge_id))

    def _edge_density(self, edge_id, count=None):
        if count is None:
            count = self.connection_info.edge_vehicle_count.get(edge_id, 0)
        return (float(count) * float(self.density_scale_m)) / max(self._edge_lane_meters(edge_id), 5.0)

    def _occupied_density_p95(self, density_vec):
        occupied = density_vec[density_vec > 0.0]
        if occupied.size == 0:
            return 0.0
        return float(np.percentile(occupied, 95))

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
        current_density = self._edge_density(edge_id)
        outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {})
        outgoing_densities = [
            self._edge_density(next_edge)
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

    def _per_action_branch_features(self, context, destination):
        features = np.zeros(30, dtype=np.float32)
        lane_now = set(context.lane_feasible_now_actions)
        for action_idx in range(6):
            base = action_idx * 5
            if action_idx not in context.edge_valid_actions:
                continue
            next_edge = self.decision_engine.get_next_edge(context.edge_id, action_idx)
            if next_edge is None:
                continue
            features[base + 0] = float(context.required_lane_shift.get(action_idx, 0)) / 3.0
            features[base + 1] = 1.0 if action_idx in lane_now else 0.0
            features[base + 2] = float(self._edge_density(next_edge))
            eta = self._estimate_remaining_eta(next_edge, destination)
            features[base + 3] = (
                min(float(eta) / float(MAX_SIMULATION_STEPS), 1.0)
                if math.isfinite(eta) else 1.0
            )
            social = self._action_social_cost_proxy(context.edge_id, action_idx, destination)
            features[base + 4] = min(float(social), 10.0) if math.isfinite(social) else 10.0
        return features

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
            density = self._edge_density(edge_id)

            state[objective_base + 0] = min(elapsed / float(MAX_SIMULATION_STEPS), 1.0)
            state[objective_base + 1] = (
                min(float(remaining_eta) / float(MAX_SIMULATION_STEPS), 1.0)
                if math.isfinite(remaining_eta) else 1.0
            )
            state[objective_base + 2] = min(float(density), 1.0)

        local_congestion = self._local_congestion_features(edge_id)
        congestion_end = objective_base + 3 + self.local_congestion_k
        state[objective_base + 3:congestion_end] = local_congestion
        state[congestion_end:congestion_end + 30] = self._per_action_branch_features(context, destination_edge)
        return state.reshape(1, -1)
    
    def valid_actions_for_vehicle(self, vehicle_id, edge_id, destination_edge, step):
        context = self.decision_engine.build_context(vehicle_id, edge_id, destination_edge, step)
        return context.available_actions

    def _policy_action_candidates(
        self,
        context,
        recent_history,
        cooldown_active,
        destination,
        decision_metrics=None,
    ):
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
        extra_buffer = max(6.0, 0.35 * float(self.decision_engine.lane_change_margin_m))
        comfortable_dist_threshold = commit_distance + extra_buffer

        safe_lane_now_actions = []
        proactive_actions = []
        filtered_available_actions = []

        for action in available_actions:
            safe_ok, _ = self.decision_engine.prefilter_action_for_loops(
                context=context,
                action_idx=action,
                destination=destination,
                recent_history=recent_history,
                distance_fn=self.get_distance_to_destination,
                distance_slack=self.score_slack,
            )
            if not safe_ok:
                continue
            filtered_available_actions.append(action)
            if action in lane_now:
                safe_lane_now_actions.append(action)
                continue

            if cooldown_active and len(safe_lane_now_actions) > 0:
                continue
            if float(context.speed) < 0.5:
                continue

            max_shift = 2 if float(context.dist_to_end) >= (
                comfortable_dist_threshold + float(self.decision_engine.lane_change_margin_m)
            ) else 1

            required_shift = int(context.required_lane_shift.get(action, 99))

            if context.commit_window and required_shift > 1:
                continue
            if required_shift > max_shift:
                continue
            if float(context.dist_to_end) <= comfortable_dist_threshold:
                continue

            proactive_actions.append(action)
            if decision_metrics is not None and required_shift == 2:
                decision_metrics["proactive_shift2_candidates_kept"] += 1
            if (
                decision_metrics is not None
                and context.commit_window
                and required_shift == 1
                and action in filtered_available_actions
            ):
                decision_metrics["soft_commit_window_admissions"] += 1

        policy_actions = sorted(set(safe_lane_now_actions) | set(proactive_actions))
        if not policy_actions:
            policy_actions = sorted(set(filtered_available_actions))
        if not policy_actions:
            return available_actions
        if decision_metrics is not None:
            broader_available_set = set(filtered_available_actions)
            lane_now_set = set(safe_lane_now_actions)
            policy_set = set(policy_actions)
            if len(broader_available_set) > len(lane_now_set):
                decision_metrics["policy_candidates_with_broader_available"] += 1
                if policy_set == lane_now_set and len(policy_set) < len(broader_available_set):
                    decision_metrics["policy_candidates_collapsed_to_lane_now_only"] += 1
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
            self.trainer.stage_transition(
                base_state,
                0,
                reward,
                next_state,
                done,
                next_valid_actions=[],
                metadata={
                    "terminal_outcome": outcome,
                    "synthetic_terminal_no_pending": True,
                    "decision_finalized": True,
                },
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
                step=step,
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

        final_metadata = dict(pending.metadata) if isinstance(pending.metadata, dict) else {}
        final_metadata["terminal_outcome"] = outcome
        final_metadata["decision_finalized"] = True
        self.trainer.stage_transition(
            pending.state,
            pending.intended_action,
            reward,
            next_state,
            done,
            next_valid_actions=[],
            metadata=final_metadata,
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

    def _tail_delay_threshold(self, vehicle, reference_edge):
        cached = getattr(vehicle, "tail_delay_threshold_steps", None)
        if cached is not None:
            return float(cached)
        initial_eta = self._estimate_remaining_eta(reference_edge, vehicle.destination)
        threshold = float(
            np.clip(
                self.tail_delay_threshold_eta_mult * float(initial_eta),
                self.tail_delay_threshold_min_steps,
                self.tail_delay_threshold_max_steps,
            )
        )
        setattr(vehicle, "tail_delay_threshold_steps", threshold)
        return threshold

    def _tail_delay_penalty_increment(self, vehicle, reference_edge, step, delta_steps):
        delta_steps = max(float(delta_steps), 0.0)
        if delta_steps <= 0.0:
            return 0.0
        elapsed_now = max(float(step) - float(vehicle.start_time), 0.0)
        elapsed_prev = max(elapsed_now - delta_steps, 0.0)
        threshold = self._tail_delay_threshold(vehicle, reference_edge)
        overflow_now = max(elapsed_now - threshold, 0.0)
        overflow_prev = max(elapsed_prev - threshold, 0.0)
        if overflow_now <= overflow_prev:
            return 0.0
        linear = self.tail_delay_linear_penalty * (overflow_now - overflow_prev)
        quadratic = self.tail_delay_quadratic_penalty * ((overflow_now ** 2) - (overflow_prev ** 2))
        return float(max(linear + quadratic, 0.0))

    def _tail_arrival_penalty(self, vehicle, reference_edge, step):
        threshold = self._tail_delay_threshold(vehicle, reference_edge)
        elapsed_now = max(float(step) - float(vehicle.start_time), 0.0)
        overflow = max(elapsed_now - threshold, 0.0)
        if overflow <= 0.0:
            return 0.0
        penalty = self.tail_arrival_penalty_per_25_steps * (overflow / 25.0)
        return float(min(penalty, self.tail_arrival_penalty_cap))
    
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
        selfless_delta=0.0,
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
        edge_density = self._edge_density(current_edge)
        reward -= 0.005 * edge_density * elapsed

        mean_density = float(self._density_mean)
        marginal_pressure = max(edge_density - mean_density, 0.0)
        reward -= self.system_congestion_scale * marginal_pressure * elapsed
        reward += self.selfless_reward_scale * float(
            np.clip(selfless_delta, -self.selfless_reward_clip, self.selfless_reward_clip)
        )
        reward -= self._tail_delay_penalty_increment(
            vehicle,
            current_edge,
            step=step,
            delta_steps=elapsed,
        )

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
                reward -= self._tail_arrival_penalty(vehicle, current_edge, step)
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

    def compute_pending_step_reward(self, vehicle, edge_id, elapsed, step, externality_penalty=0.0, pending_age=0, lane_change_deferrals=0):
        """
        Dense reward used while a decision is pending and has not finalized yet.
        Keeps the training objective travel-time centric without waiting for an edge transition.
        """
        elapsed = max(float(elapsed), 0.0)
        if elapsed <= 0.0:
            return 0.0
        time_cost_scale = self._get_route_difficulty_scale(vehicle, edge_id)
        reward = -self.travel_time_penalty * time_cost_scale * elapsed

        edge_density = self._edge_density(edge_id)
        reward -= 0.005 * edge_density * elapsed

        mean_density = float(self._density_mean)
        marginal_pressure = max(edge_density - mean_density, 0.0)
        reward -= self.system_congestion_scale * marginal_pressure * elapsed
        reward -= self.pending_latency_penalty_per_step * float(max(pending_age, 0))
        reward -= 0.02 * float(max(lane_change_deferrals, 0))
        reward -= self._tail_delay_penalty_increment(
            vehicle,
            edge_id,
            step=step,
            delta_steps=elapsed,
        )
        return self._clip_reward(reward)

    def generate_episode_vehicles(self, episode_seed=None):
        """
        Generate controlled and uncontrolled vehicles for one training episode.
        """
        generator = target_vehicles_generator(os.path.join(self.sumocfg_dir, self.net_file))
        route_path = os.path.join(self.sumocfg_dir, self.route_file)
        vehicle_list = generator.generate_vehicles(
            num_target_vehicles=100,
            num_random_vehicles=100,
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

    def _build_option_features(self, strategic_context):
        return [opt.option_features for opt in strategic_context.options if opt.option_features is not None]

    def _queue_len(self, edge_id):
        try:
            return traci.edge.getLastStepVehicleNumber(edge_id)
        except Exception:
            return 0.0

    def _get_destination_edge(self, vehicle_id, fallback_edge):
        try:
            route = traci.vehicle.getRoute(vehicle_id)
            if route:
                return route[-1]
        except Exception:
            pass
        return fallback_edge

    def _state_for_context(self, vehicle_id, context, destination):
        snapshot = VehicleSnapshot(
            vehicle_id=vehicle_id,
            step=int(context.step),
            edge_id=context.edge_id,
            lane_id=context.lane_id,
            lane_index=context.lane_index,
            lane_count=context.lane_count,
            lane_position=max(float(self.connection_info.edge_length_dict.get(context.edge_id, 5.0)) - float(context.dist_to_end), 0.0),
            lane_length=max(float(self.connection_info.edge_length_dict.get(context.edge_id, 5.0)), 5.0),
            dist_to_end=context.dist_to_end,
            speed=context.speed,
        )
        return self.encode_state(vehicle_id, context.edge_id, destination, step=int(context.step), snapshot=snapshot)

    def _init_episode_metrics(self):
        return {
            "decision_open_count": 0,
            "learned_decision_count": 0,
            "forced_nonlearned_decision_count": 0,
            "option_count_total": 0,
            "executable_option_count_total": 0,
            "option_success_count": 0,
            "option_failure_counts": defaultdict(int),
            "decision_horizon_steps": [],
            "lane_change_attempts": 0,
            "lane_change_successes": 0,
            "lane_change_failure_counts": defaultdict(int),
            "commit_window_miss_count": 0,
            "stalled_lane_change_count": 0,
            "forced_by_lane_commit_count": 0,
            "teleport_during_execution_count": 0,
            "collision_during_execution_count": 0,
            "timeout_during_execution_count": 0,
        }

    def run(self):
        """Run option-based strategic RL with deterministic tactical execution."""
        sumo_binary = checkBinary("sumo")
        assert hasattr(self, "strategic_trainer"), "Strategic trainer must be initialized."
        if isinstance(self.strategic_trainer.model.input_shape, list):
            assert len(self.strategic_trainer.model.input_shape) == 2, "Strategic model must be two-input shared+option scorer."
        assert self.trainer is not None, "Legacy trainer object missing unexpectedly."

        for episode in range(self.episodes):
            traci.start([sumo_binary, "-c", self.sumocfg_path, "--start"])
            step = 0
            active_decisions = {}
            metrics = self._init_episode_metrics()

            while step < MAX_SIMULATION_STEPS and (traci.simulation.getMinExpectedNumber() > 0):
                traci.simulationStep()
                step += 1
                live_ids = set(traci.vehicle.getIDList())

                for vid in list(live_ids):
                    try:
                        edge_id = traci.vehicle.getRoadID(vid)
                        if not edge_id or edge_id.startswith(":"):
                            continue
                        dest = self._get_destination_edge(vid, edge_id)
                        context = self.decision_engine.build_context(vid, edge_id, dest, step)
                    except Exception:
                        continue

                    # Open decision if none active.
                    if vid not in active_decisions:
                        shared_state = self._state_for_context(vid, context, dest)
                        strategic_context = self.tactical_executor.build_options(
                            context,
                            shared_state,
                            queue_length_fn=self._queue_len,
                        )
                        executable_options = self.tactical_executor.filter_executable_options(strategic_context)
                        metrics["decision_open_count"] += 1
                        metrics["option_count_total"] += len(strategic_context.options)
                        metrics["executable_option_count_total"] += len(executable_options)

                        if len(executable_options) < 2:
                            metrics["forced_nonlearned_decision_count"] += 1
                            continue

                        option_features = [opt.option_features for opt in executable_options]
                        selected_idx, source = self.strategic_trainer.select_option(shared_state, option_features)
                        if selected_idx is None:
                            continue
                        selected_option = executable_options[selected_idx]
                        tactical_state = self.tactical_executor.begin_execution(vid, selected_option, step)
                        active_decisions[vid] = ActiveStrategicDecision(
                            context=strategic_context,
                            chosen_option_idx=selected_idx,
                            chosen_option=selected_option,
                            opened_step=step,
                            tactical_state=tactical_state,
                        )
                        metrics["learned_decision_count"] += 1

                    # Step active decision.
                    active = active_decisions.get(vid)
                    if not active:
                        continue
                    active.reward_accumulator += -float(self.travel_time_penalty)
                    active.horizon_steps += 1
                    self.tactical_executor.step_execution(active.tactical_state, context, active.chosen_option, step)
                    outcome = self.tactical_executor.resolve_execution(active.tactical_state, context, active.chosen_option)

                    if outcome == "in_progress":
                        continue

                    done = bool(outcome != "success")
                    if outcome == "success":
                        metrics["option_success_count"] += 1
                    else:
                        metrics["option_failure_counts"][outcome] += 1
                        if outcome == "commit_window_miss":
                            metrics["commit_window_miss_count"] += 1
                        elif outcome == "stalled_lane_change":
                            metrics["stalled_lane_change_count"] += 1
                        elif outcome == "forced_by_lane_commit":
                            metrics["forced_by_lane_commit_count"] += 1
                        elif outcome == "teleport":
                            metrics["teleport_during_execution_count"] += 1
                        elif outcome == "collision":
                            metrics["collision_during_execution_count"] += 1
                        elif outcome == "timeout":
                            metrics["timeout_during_execution_count"] += 1

                    next_shared = None
                    next_option_features = None
                    if not done:
                        next_shared = self._state_for_context(vid, context, dest)
                        next_ctx = self.tactical_executor.build_options(context, next_shared, queue_length_fn=self._queue_len)
                        next_option_features = self._build_option_features(next_ctx)

                    transition = StrategicTransition(
                        state_shared=active.context.shared_state,
                        state_option_features=self._build_option_features(active.context),
                        chosen_option_idx=active.chosen_option_idx,
                        aggregated_reward=float(active.reward_accumulator),
                        next_state_shared=next_shared,
                        next_state_option_features=next_option_features,
                        done=bool(done),
                        outcome=("resolved" if outcome == "success" else "failed"),
                        tactical_outcome=None if outcome == "success" else outcome,
                        horizon_steps=max(active.horizon_steps, 1),
                    )
                    if outcome in self._strategic_main_allowed_tactical_outcomes:
                        self.strategic_trainer.replay.add_strategic(transition)
                    if outcome != "success":
                        self.strategic_trainer.replay.add_tactical_failure(transition)
                    metrics["decision_horizon_steps"].append(float(active.horizon_steps))
                    active_decisions.pop(vid, None)

                self.strategic_trainer.train_step()

            # Resolve active decisions at episode terminal once.
            for vid, active in list(active_decisions.items()):
                transition = StrategicTransition(
                    state_shared=active.context.shared_state,
                    state_option_features=self._build_option_features(active.context),
                    chosen_option_idx=active.chosen_option_idx,
                    aggregated_reward=float(active.reward_accumulator),
                    next_state_shared=None,
                    next_state_option_features=None,
                    done=True,
                    outcome="failed",
                    tactical_outcome="timeout",
                    horizon_steps=max(active.horizon_steps, 1),
                )
                self.strategic_trainer.replay.add_terminal_only(transition)
                metrics["timeout_during_execution_count"] += 1
                active_decisions.pop(vid, None)

            traci.close()
            assert len(self.trainer.memory) == 0, "Legacy DQN replay should remain unused in redesigned run path."

            avg_option_count = metrics["option_count_total"] / max(metrics["decision_open_count"], 1)
            avg_exec_option_count = metrics["executable_option_count_total"] / max(metrics["decision_open_count"], 1)
            avg_horizon = float(np.mean(metrics["decision_horizon_steps"])) if metrics["decision_horizon_steps"] else 0.0
            success_rate = metrics["option_success_count"] / max(metrics["learned_decision_count"], 1)
            print(
                f"[EP {episode+1:03d}] open={metrics['decision_open_count']} learned={metrics['learned_decision_count']} "
                f"forced_nonlearned={metrics['forced_nonlearned_decision_count']} avg_options={avg_option_count:.2f} "
                f"avg_exec_options={avg_exec_option_count:.2f} strategic_success_rate={success_rate:.2%} "
                f"avg_horizon={avg_horizon:.2f} replay(s/t/term)=({len(self.strategic_trainer.replay.strategic_replay)}/"
                f"{len(self.strategic_trainer.replay.tactical_failure_replay)}/{len(self.strategic_trainer.replay.terminal_only_replay)}) "
                f"sample_fracs={self.strategic_trainer.last_sample_fractions} "
                f"sample_counts={self.strategic_trainer.last_sample_counts} "
                f"legacy_replay_size={len(self.trainer.memory)}"
            )

        self.strategic_trainer.model.save(self.model_output_path)
        print(f"Option-based strategic model saved to {self.model_output_path}")
