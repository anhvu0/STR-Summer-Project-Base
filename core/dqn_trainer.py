"""Dueling Double-DQN trainer with prioritized n-step replay."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import Add, Dense, Input, Lambda
from keras.losses import Huber
from keras.optimizers import Adam


@dataclass
class ReplayItem:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_valid_actions: List[int]
    metadata: Dict[str, object]


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.items: List[ReplayItem] = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.next_idx = 0

    def __len__(self) -> int:
        return len(self.items)

    def add(self, item: ReplayItem, priority: float = 1.0) -> None:
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            self.items[self.next_idx] = item

        self.priorities[self.next_idx] = max(float(priority), 1e-6)
        self.next_idx = (self.next_idx + 1) % self.capacity

    def sample(self, batch_size: int, frame_idx: int) -> Tuple[List[ReplayItem], np.ndarray, np.ndarray]:
        size = len(self.items)
        if size == 0:
            raise ValueError("Replay buffer is empty.")

        probs = self.priorities[:size] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(size, batch_size, p=probs)

        beta = min(1.0, self.beta_start + frame_idx * 1e-5)
        weights = (size * probs[indices]) ** (-beta)
        weights /= weights.max()
        sampled = [self.items[i] for i in indices]
        return sampled, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, err in zip(indices, td_errors):
            self.priorities[int(idx)] = max(abs(float(err)), 1e-6)


class NStepCollector:
    def __init__(self, n_step: int, gamma: float):
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.buffer: Deque[ReplayItem] = deque(maxlen=n_step)

    def add(self, item: ReplayItem) -> Optional[ReplayItem]:
        self.buffer.append(item)
        if len(self.buffer) < self.n_step and not item.done:
            return None
        return self._pop_n_step()

    def flush(self) -> List[ReplayItem]:
        out: List[ReplayItem] = []
        while self.buffer:
            out.append(self._pop_n_step())
        return out

    def _pop_n_step(self) -> ReplayItem:
        total_reward = 0.0
        done = False
        next_state = self.buffer[-1].next_state
        next_valid_actions = self.buffer[-1].next_valid_actions
        metadata = dict(self.buffer[0].metadata)

        for i, entry in enumerate(self.buffer):
            total_reward += (self.gamma**i) * entry.reward
            if entry.done:
                done = True
                next_state = entry.next_state
                next_valid_actions = entry.next_valid_actions
                break

        first = self.buffer.popleft()
        return ReplayItem(
            state=first.state,
            action=first.action,
            reward=float(total_reward),
            next_state=next_state,
            done=done,
            next_valid_actions=next_valid_actions,
            metadata=metadata,
        )


class DQNTrainer:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay_decisions: int = 120000,
        batch_size: int = 128,
        replay_capacity: int = 50000,
        replay_warmup: int = 4000,
        target_update_every: int = 500,
        gradient_clip_norm: float = 5.0,
        n_step: int = 3,
    ):
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.replay_warmup = int(replay_warmup)
        self.target_update_every = int(target_update_every)
        self.gradient_clip_norm = float(gradient_clip_norm)

        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay_decisions = max(int(epsilon_decay_decisions), 1)
        self.decision_count = 0

        self.replay = PrioritizedReplayBuffer(capacity=replay_capacity)
        self.n_step_collector = NStepCollector(n_step=n_step, gamma=gamma)

        self.model = self._build_dueling_model(learning_rate)
        self.target_model = self._build_dueling_model(learning_rate)
        self.target_model.set_weights(self.model.get_weights())
        self.train_steps = 0
        self.last_loss: Optional[float] = None

    def _build_dueling_model(self, learning_rate: float) -> Model:
        inp = Input(shape=(self.state_size,))
        x = Dense(256, activation="relu")(inp)
        x = Dense(256, activation="relu")(x)

        value = Dense(128, activation="relu")(x)
        value = Dense(1, activation="linear")(value)

        advantage = Dense(128, activation="relu")(x)
        advantage = Dense(self.action_size, activation="linear")(advantage)
        advantage = Lambda(lambda a: a - tf.reduce_mean(a, axis=1, keepdims=True))(advantage)

        q_values = Add()([value, advantage])
        model = Model(inputs=inp, outputs=q_values)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate, clipnorm=self.gradient_clip_norm),
            loss=Huber(),
        )
        return model

    @property
    def epsilon(self) -> float:
        frac = min(self.decision_count / float(self.epsilon_decay_decisions), 1.0)
        return self.epsilon_start + frac * (self.epsilon_min - self.epsilon_start)

    def select_action(self, state: np.ndarray, valid_actions: Sequence[int]) -> Tuple[Optional[int], str]:
        if not valid_actions:
            return None, "none"

        self.decision_count += 1
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(valid_actions)), "explore"

        q_values = self.model.predict(state, verbose=0)[0]
        masked = np.full_like(q_values, -1e9)
        masked[list(valid_actions)] = q_values[list(valid_actions)]
        return int(np.argmax(masked)), "policy"

    def remember(self, item: ReplayItem) -> None:
        n_item = self.n_step_collector.add(item)
        if n_item is not None:
            self.replay.add(n_item)

    def flush_episode(self) -> None:
        for item in self.n_step_collector.flush():
            self.replay.add(item)

    def train_step(self) -> Optional[float]:
        if len(self.replay) < max(self.replay_warmup, self.batch_size):
            return None

        batch, indices, weights = self.replay.sample(self.batch_size, self.train_steps)
        states = np.vstack([x.state for x in batch])
        actions = np.array([x.action for x in batch], dtype=np.int32)
        rewards = np.array([x.reward for x in batch], dtype=np.float32)
        next_states = np.vstack([x.next_state for x in batch])
        dones = np.array([x.done for x in batch], dtype=np.bool_)

        q = self.model.predict(states, verbose=0)
        q_next_online = self.model.predict(next_states, verbose=0)
        q_next_target = self.target_model.predict(next_states, verbose=0)

        target = q.copy()
        td_errors = np.zeros(self.batch_size, dtype=np.float32)

        for i, item in enumerate(batch):
            if dones[i] or not item.next_valid_actions:
                bootstrap = 0.0
            else:
                masked_online = np.full(self.action_size, -1e9, dtype=np.float32)
                masked_online[item.next_valid_actions] = q_next_online[i][item.next_valid_actions]
                next_action = int(np.argmax(masked_online))
                bootstrap = q_next_target[i][next_action]

            y = rewards[i] + (0.0 if dones[i] else self.gamma * bootstrap)
            td_errors[i] = y - q[i][actions[i]]
            target[i][actions[i]] = y

        sample_weights = weights
        loss = self.model.train_on_batch(states, target, sample_weight=sample_weights)
        self.last_loss = float(loss) if loss is not None else None
        self.replay.update_priorities(indices, td_errors)

        self.train_steps += 1
        if self.train_steps % self.target_update_every == 0:
            self.target_model.set_weights(self.model.get_weights())
        return self.last_loss
