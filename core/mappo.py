from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from core.routing_graph import (
    RoutingGraphObservation,
    RoutingGraphSpec,
    routing_graph_batch_to_tensors,
    stack_routing_graph_observations,
)


def _build_mlp(input_size: int, hidden_sizes: Sequence[int], output_size: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    previous = int(input_size)
    for hidden in hidden_sizes:
        hidden = int(hidden)
        layers.append(nn.Linear(previous, hidden))
        layers.append(nn.ReLU())
        previous = hidden
    layers.append(nn.Linear(previous, int(output_size)))
    return nn.Sequential(*layers)


def action_mask_from_valid_actions(action_size: int, valid_actions: Optional[Sequence[int]]) -> np.ndarray:
    mask = np.zeros(int(action_size), dtype=np.float32)
    if valid_actions is None:
        mask[:] = 1.0
        return mask
    for action in valid_actions:
        action_idx = int(action)
        if 0 <= action_idx < int(action_size):
            mask[action_idx] = 1.0
    return mask


def _masked_logits(logits: torch.Tensor, action_masks: torch.Tensor) -> torch.Tensor:
    action_masks = action_masks.to(dtype=torch.bool)
    safe_logits = logits.masked_fill(~action_masks, -1.0e9)
    no_valid = ~action_masks.any(dim=-1, keepdim=True)
    if no_valid.any():
        safe_logits = torch.where(no_valid, logits, safe_logits)
    return safe_logits


def _action_distribution_from_logits(
    logits: torch.Tensor,
    action_masks: torch.Tensor,
    config: "MAPPOConfig",
    *,
    exploratory: bool,
) -> Categorical:
    masked_logits = _masked_logits(logits, action_masks)
    temperature = max(float(config.action_sampling_temperature), 1.0e-6)
    if not exploratory or float(config.valid_action_exploration_mix) <= 0.0:
        return Categorical(logits=masked_logits / temperature)

    valid_mask = action_masks.to(dtype=torch.float32)
    valid_counts = valid_mask.sum(dim=-1, keepdim=True)
    no_valid = valid_counts <= 0.0
    uniform_valid = valid_mask / valid_counts.clamp_min(1.0)
    full_uniform = torch.full_like(uniform_valid, 1.0 / max(int(logits.shape[-1]), 1))
    uniform_valid = torch.where(no_valid, full_uniform, uniform_valid)

    base_probs = torch.softmax(masked_logits / temperature, dim=-1)
    mix = float(np.clip(config.valid_action_exploration_mix, 0.0, 0.50))
    mixed_probs = ((1.0 - mix) * base_probs) + (mix * uniform_valid)
    mixed_probs = mixed_probs / mixed_probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    return Categorical(probs=mixed_probs)


def _copy_graph_observation(observation: RoutingGraphObservation) -> RoutingGraphObservation:
    return RoutingGraphObservation(
        node_dynamic_features=np.asarray(observation.node_dynamic_features, dtype=np.float32).copy(),
        scalar_features=np.asarray(observation.scalar_features, dtype=np.float32).reshape(-1).copy(),
        action_features=np.asarray(observation.action_features, dtype=np.float32).copy(),
        action_node_indices=np.asarray(observation.action_node_indices, dtype=np.int64).reshape(-1).copy(),
        current_node_index=int(observation.current_node_index),
        destination_node_index=int(observation.destination_node_index),
    )


def _slice_graph_tensor_batch(
    batch: Dict[str, torch.Tensor],
    batch_indices: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    return {
        key: value.index_select(0, batch_indices)
        for key, value in batch.items()
    }


def _gather_node_embeddings(node_embeddings: torch.Tensor, node_indices: torch.Tensor) -> torch.Tensor:
    hidden_size = int(node_embeddings.shape[-1])
    safe_indices = torch.clamp(node_indices, min=0)
    gathered = node_embeddings.gather(
        1,
        safe_indices.unsqueeze(-1).expand(-1, -1, hidden_size),
    )
    valid_mask = (node_indices >= 0).unsqueeze(-1)
    return gathered * valid_mask


def _gather_single_node_embedding(node_embeddings: torch.Tensor, node_indices: torch.Tensor) -> torch.Tensor:
    gathered = _gather_node_embeddings(node_embeddings, node_indices.unsqueeze(-1))
    return gathered.squeeze(1)


def _mean_neighbor_aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    node_count = int(x.shape[1])
    if edge_index.numel() == 0:
        return torch.zeros_like(x)

    src = edge_index[0]
    dst = edge_index[1]
    messages = x.index_select(1, src)
    aggregated = torch.zeros_like(x)
    aggregated.index_add_(1, dst, messages)

    counts = torch.zeros(node_count, dtype=x.dtype, device=x.device)
    counts.index_add_(0, dst, torch.ones(dst.shape[0], dtype=x.dtype, device=x.device))
    return aggregated / counts.clamp_min(1.0).view(1, node_count, 1)


class GraphMessagePassingLayer(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.update_linear_1 = nn.Linear(2 * int(hidden_size), int(hidden_size))
        self.update_linear_2 = nn.Linear(int(hidden_size), int(hidden_size))
        self.norm = nn.LayerNorm(int(hidden_size))
        self.dropout = float(max(dropout, 0.0))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        aggregated = _mean_neighbor_aggregate(x, edge_index)
        update_input = torch.cat((x, aggregated), dim=-1)
        updates = F.relu(self.update_linear_1(update_input))
        updates = self.update_linear_2(updates)
        if self.dropout > 0.0:
            updates = F.dropout(updates, p=self.dropout, training=self.training)
        return self.norm(x + updates)


class RoutingGraphEncoder(nn.Module):
    def __init__(
        self,
        graph_spec: RoutingGraphSpec,
        *,
        hidden_size: int,
        layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.node_count = int(graph_spec.node_count)
        self.feature_layout = graph_spec.feature_layout

        edge_index = torch.as_tensor(graph_spec.edge_index, dtype=torch.long)
        node_static_features = torch.as_tensor(graph_spec.node_static_features, dtype=torch.float32)
        self.register_buffer("edge_index", edge_index, persistent=False)
        self.register_buffer("node_static_features", node_static_features, persistent=False)

        node_input_dim = (
            int(self.feature_layout.node_static_dim)
            + int(self.feature_layout.node_dynamic_dim)
        )
        self.input_projection = nn.Linear(node_input_dim, self.hidden_size)
        self.layers = nn.ModuleList(
            GraphMessagePassingLayer(self.hidden_size, dropout=dropout)
            for _ in range(max(int(layers), 1))
        )
        self.output_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, observation_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        dynamic_features = observation_batch["node_dynamic_features"]
        if int(dynamic_features.shape[1]) != self.node_count:
            raise ValueError(
                "Observation node count {} does not match graph encoder node count {}.".format(
                    int(dynamic_features.shape[1]),
                    self.node_count,
                )
            )

        static_features = self.node_static_features.unsqueeze(0).expand(dynamic_features.shape[0], -1, -1)
        node_inputs = torch.cat((static_features, dynamic_features), dim=-1)
        node_embeddings = F.relu(self.input_projection(node_inputs))
        for layer in self.layers:
            node_embeddings = layer(node_embeddings, self.edge_index)
        return self.output_norm(node_embeddings)


@dataclass(frozen=True)
class MAPPOConfig:
    actor_learning_rate: float = 3.0e-4
    critic_learning_rate: float = 1.0e-3
    gamma: float = 0.99
    clip_epsilon: float = 0.20
    entropy_coef: float = 0.02
    value_coef: float = 0.50
    max_grad_norm: float = 10.0
    update_epochs: int = 6
    minibatch_size: int = 256
    normalize_advantages: bool = True
    min_transitions_per_update: int = 64
    actor_hidden_sizes: Tuple[int, ...] = (256, 128)
    critic_hidden_sizes: Tuple[int, ...] = (256, 128)
    graph_hidden_size: int = 128
    graph_layers: int = 3
    graph_dropout: float = 0.0
    action_sampling_temperature: float = 1.0
    valid_action_exploration_mix: float = 0.04


@dataclass
class MAPPOTransition:
    observation: RoutingGraphObservation
    central_observation: np.ndarray
    action: int
    action_mask: np.ndarray
    log_prob: float
    value: float
    reward: float
    next_observation: RoutingGraphObservation
    next_central_observation: np.ndarray
    done: bool
    discount_steps: int = 1
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionSelection:
    action: int
    log_prob: float
    value: float
    entropy: float
    action_mask: np.ndarray


class MAPPOActor(nn.Module):
    def __init__(self, graph_spec: RoutingGraphSpec, action_size: int, config: MAPPOConfig):
        super().__init__()
        self.feature_layout = graph_spec.feature_layout
        self.action_size = int(action_size)
        self.graph_encoder = RoutingGraphEncoder(
            graph_spec,
            hidden_size=int(config.graph_hidden_size),
            layers=int(config.graph_layers),
            dropout=float(config.graph_dropout),
        )
        hidden_size = int(config.graph_hidden_size)
        self.scalar_encoder = _build_mlp(
            int(self.feature_layout.scalar_dim),
            (hidden_size,),
            hidden_size,
        )
        self.context_network = _build_mlp(
            4 * hidden_size,
            config.actor_hidden_sizes,
            hidden_size,
        )
        self.action_network = _build_mlp(
            (2 * hidden_size) + int(self.feature_layout.action_feature_dim),
            config.actor_hidden_sizes,
            1,
        )

    def forward(self, observation_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_embeddings = self.graph_encoder(observation_batch)
        current_embeddings = _gather_single_node_embedding(
            node_embeddings,
            observation_batch["current_node_index"],
        )
        destination_embeddings = _gather_single_node_embedding(
            node_embeddings,
            observation_batch["destination_node_index"],
        )
        graph_summary = node_embeddings.mean(dim=1)
        scalar_embeddings = self.scalar_encoder(observation_batch["scalar_features"])

        context_features = torch.cat(
            (
                current_embeddings,
                destination_embeddings,
                graph_summary,
                scalar_embeddings,
            ),
            dim=-1,
        )
        shared_context = self.context_network(context_features)

        action_node_embeddings = _gather_node_embeddings(
            node_embeddings,
            observation_batch["action_node_indices"],
        )
        expanded_context = shared_context.unsqueeze(1).expand(-1, self.action_size, -1)
        action_inputs = torch.cat(
            (
                expanded_context,
                action_node_embeddings,
                observation_batch["action_features"],
            ),
            dim=-1,
        )
        logits = self.action_network(action_inputs).squeeze(-1)
        return logits


class MAPPOCritic(nn.Module):
    def __init__(
        self,
        graph_spec: RoutingGraphSpec,
        central_observation_size: int,
        config: MAPPOConfig,
    ):
        super().__init__()
        self.feature_layout = graph_spec.feature_layout
        self.central_observation_size = int(central_observation_size)
        self.graph_encoder = RoutingGraphEncoder(
            graph_spec,
            hidden_size=int(config.graph_hidden_size),
            layers=int(config.graph_layers),
            dropout=float(config.graph_dropout),
        )
        hidden_size = int(config.graph_hidden_size)
        self.scalar_encoder = _build_mlp(
            int(self.feature_layout.scalar_dim),
            (hidden_size,),
            hidden_size,
        )
        self.central_encoder = _build_mlp(
            self.central_observation_size,
            (hidden_size,),
            hidden_size,
        )
        self.value_network = _build_mlp(
            5 * hidden_size,
            config.critic_hidden_sizes,
            1,
        )

    def forward(
        self,
        observation_batch: Dict[str, torch.Tensor],
        central_observations: torch.Tensor,
    ) -> torch.Tensor:
        node_embeddings = self.graph_encoder(observation_batch)
        current_embeddings = _gather_single_node_embedding(
            node_embeddings,
            observation_batch["current_node_index"],
        )
        destination_embeddings = _gather_single_node_embedding(
            node_embeddings,
            observation_batch["destination_node_index"],
        )
        graph_summary = node_embeddings.mean(dim=1)
        scalar_embeddings = self.scalar_encoder(observation_batch["scalar_features"])
        central_embeddings = self.central_encoder(central_observations)

        critic_inputs = torch.cat(
            (
                current_embeddings,
                destination_embeddings,
                graph_summary,
                scalar_embeddings,
                central_embeddings,
            ),
            dim=-1,
        )
        return self.value_network(critic_inputs).squeeze(-1)


class MAPPOTrainer:
    def __init__(
        self,
        graph_spec: RoutingGraphSpec,
        central_observation_size: int,
        action_size: int,
        config: Optional[MAPPOConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.graph_spec = graph_spec
        self.feature_layout = graph_spec.feature_layout
        self.central_observation_size = int(central_observation_size)
        self.action_size = int(action_size)
        if self.action_size != int(self.feature_layout.action_count):
            raise ValueError(
                "action_size {} does not match graph feature layout action_count {}.".format(
                    self.action_size,
                    int(self.feature_layout.action_count),
                )
            )

        self.config = config or MAPPOConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = MAPPOActor(
            self.graph_spec,
            self.action_size,
            self.config,
        ).to(self.device)
        self.critic = MAPPOCritic(
            self.graph_spec,
            self.central_observation_size,
            self.config,
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=float(self.config.actor_learning_rate),
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(self.config.critic_learning_rate),
        )

        self.buffer: List[MAPPOTransition] = []
        self.update_steps = 0
        self.transitions_collected = 0
        self.epsilon = 0.0
        self.epsilon_min = 0.0
        self.epsilon_decay = 1.0
        self.last_policy_loss: Optional[float] = None
        self.last_value_loss: Optional[float] = None
        self.last_entropy: Optional[float] = None
        self.last_approx_kl: Optional[float] = None
        self.last_clip_fraction: Optional[float] = None

    @property
    def train_steps(self) -> int:
        return int(self.update_steps)

    @property
    def memory(self):
        return self.buffer

    @property
    def last_loss(self) -> Optional[float]:
        return self.last_policy_loss

    @classmethod
    def episode_metric_fields(cls) -> Tuple[str, ...]:
        return (
            "updates_cumulative",
            "updates_episode",
            "transitions_buffered",
            "transitions_collected_episode",
            "transitions_collected_cumulative",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
        )

    def reset_episode_tracking(self) -> None:
        self.last_policy_loss = None
        self.last_value_loss = None
        self.last_entropy = None
        self.last_approx_kl = None
        self.last_clip_fraction = None

    def capture_episode_metric_snapshot(self) -> Dict[str, int]:
        return {
            "update_steps": int(self.update_steps),
            "transitions_collected": int(self.transitions_collected),
        }

    def episode_metric_row(self, counter_start: Dict[str, int]) -> Dict[str, object]:
        return {
            "updates_cumulative": int(self.update_steps),
            "updates_episode": int(self.update_steps - int(counter_start.get("update_steps", 0))),
            "transitions_buffered": int(len(self.buffer)),
            "transitions_collected_episode": int(
                self.transitions_collected - int(counter_start.get("transitions_collected", 0))
            ),
            "transitions_collected_cumulative": int(self.transitions_collected),
            "policy_loss": self.last_policy_loss if self.last_policy_loss is not None else "",
            "value_loss": self.last_value_loss if self.last_value_loss is not None else "",
            "entropy": self.last_entropy if self.last_entropy is not None else "",
            "approx_kl": self.last_approx_kl if self.last_approx_kl is not None else "",
            "clip_fraction": self.last_clip_fraction if self.last_clip_fraction is not None else "",
        }

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _observation_to_tensor_batch(
        self,
        observation: RoutingGraphObservation,
    ) -> Dict[str, torch.Tensor]:
        return routing_graph_batch_to_tensors(
            stack_routing_graph_observations([observation]),
            self.device,
        )

    def select_action(
        self,
        observation: RoutingGraphObservation,
        valid_actions: Optional[Sequence[int]],
        central_observation: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> ActionSelection:
        central_array = np.asarray(central_observation, dtype=np.float32).reshape(1, -1)
        action_mask = action_mask_from_valid_actions(self.action_size, valid_actions).reshape(1, -1)

        with torch.no_grad():
            observation_tensor_batch = self._observation_to_tensor_batch(observation)
            central_tensor = self._to_tensor(central_array)
            action_mask_tensor = self._to_tensor(action_mask)
            logits = self.actor(observation_tensor_batch)
            masked_logits = _masked_logits(logits, action_mask_tensor)
            distribution = _action_distribution_from_logits(
                logits,
                action_mask_tensor,
                self.config,
                exploratory=True,
            )
            if deterministic:
                action_tensor = torch.argmax(masked_logits, dim=-1)
            else:
                action_tensor = distribution.sample()
            log_prob_tensor = distribution.log_prob(action_tensor)
            entropy_tensor = distribution.entropy()
            value_tensor = self.critic(observation_tensor_batch, central_tensor)

        return ActionSelection(
            action=int(action_tensor.item()),
            log_prob=float(log_prob_tensor.item()),
            value=float(value_tensor.item()),
            entropy=float(entropy_tensor.item()),
            action_mask=action_mask.reshape(-1).copy(),
        )

    def record_transition(
        self,
        *,
        observation: RoutingGraphObservation,
        central_observation: np.ndarray,
        action: int,
        action_mask: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        next_observation: RoutingGraphObservation,
        next_central_observation: np.ndarray,
        done: bool,
        discount_steps: int = 1,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        transition = MAPPOTransition(
            observation=_copy_graph_observation(observation),
            central_observation=np.asarray(central_observation, dtype=np.float32).reshape(-1).copy(),
            action=int(action),
            action_mask=np.asarray(action_mask, dtype=np.float32).reshape(-1).copy(),
            log_prob=float(log_prob),
            value=float(value),
            reward=float(reward),
            next_observation=_copy_graph_observation(next_observation),
            next_central_observation=np.asarray(next_central_observation, dtype=np.float32).reshape(-1).copy(),
            done=bool(done),
            discount_steps=max(int(discount_steps), 1),
            metadata=dict(metadata or {}),
        )
        self.buffer.append(transition)
        self.transitions_collected += 1

    def stage_transition(self, *args, **kwargs) -> None:
        return None

    def replay(self) -> int:
        return 0

    def flush_staged_episode(self, *args, **kwargs) -> None:
        return None

    def _evaluate_actions(
        self,
        observations: Dict[str, torch.Tensor],
        central_observations: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(observations)
        distribution = _action_distribution_from_logits(
            logits,
            action_masks,
            self.config,
            exploratory=True,
        )
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.critic(observations, central_observations)
        return log_probs, entropy, values

    def update(self) -> int:
        transition_count = len(self.buffer)
        if transition_count < int(self.config.min_transitions_per_update):
            return 0

        observations = stack_routing_graph_observations(
            [transition.observation for transition in self.buffer]
        )
        central_observations = np.stack(
            [transition.central_observation for transition in self.buffer],
            axis=0,
        )
        actions = np.asarray([transition.action for transition in self.buffer], dtype=np.int64)
        action_masks = np.stack([transition.action_mask for transition in self.buffer], axis=0)
        old_log_probs = np.asarray([transition.log_prob for transition in self.buffer], dtype=np.float32)
        old_values = np.asarray([transition.value for transition in self.buffer], dtype=np.float32)
        rewards = np.asarray([transition.reward for transition in self.buffer], dtype=np.float32)
        next_observations = stack_routing_graph_observations(
            [transition.next_observation for transition in self.buffer]
        )
        next_central_observations = np.stack(
            [transition.next_central_observation for transition in self.buffer],
            axis=0,
        )
        dones = np.asarray([transition.done for transition in self.buffer], dtype=np.float32)
        discount_steps = np.asarray([transition.discount_steps for transition in self.buffer], dtype=np.float32)

        observation_tensor_batch = routing_graph_batch_to_tensors(observations, self.device)
        next_observation_tensor_batch = routing_graph_batch_to_tensors(next_observations, self.device)
        central_tensor = self._to_tensor(central_observations)
        next_central_tensor = self._to_tensor(next_central_observations)

        with torch.no_grad():
            next_values = self.critic(next_observation_tensor_batch, next_central_tensor).cpu().numpy()

        discount_factors = np.power(float(self.config.gamma), discount_steps)
        returns = rewards + (1.0 - dones) * discount_factors * next_values
        advantages = returns - old_values
        if self.config.normalize_advantages and transition_count > 1:
            advantages = (advantages - advantages.mean()) / max(advantages.std(), 1.0e-8)

        actions_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        action_mask_tensor = self._to_tensor(action_masks)
        old_log_prob_tensor = self._to_tensor(old_log_probs)
        returns_tensor = self._to_tensor(returns)
        advantage_tensor = self._to_tensor(advantages)

        minibatch_size = max(1, min(int(self.config.minibatch_size), transition_count))
        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        approx_kls: List[float] = []
        clip_fractions: List[float] = []

        index_array = np.arange(transition_count)
        for _ in range(int(self.config.update_epochs)):
            np.random.shuffle(index_array)
            for start in range(0, transition_count, minibatch_size):
                batch_idx_array = index_array[start:start + minibatch_size]
                batch_idx = torch.as_tensor(batch_idx_array, dtype=torch.long, device=self.device)
                batch_obs = _slice_graph_tensor_batch(observation_tensor_batch, batch_idx)
                batch_central = central_tensor.index_select(0, batch_idx)
                batch_actions = actions_tensor.index_select(0, batch_idx)
                batch_masks = action_mask_tensor.index_select(0, batch_idx)
                batch_old_log_probs = old_log_prob_tensor.index_select(0, batch_idx)
                batch_returns = returns_tensor.index_select(0, batch_idx)
                batch_advantages = advantage_tensor.index_select(0, batch_idx)

                new_log_probs, entropy, predicted_values = self._evaluate_actions(
                    batch_obs,
                    batch_central,
                    batch_actions,
                    batch_masks,
                )
                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - float(self.config.clip_epsilon),
                    1.0 + float(self.config.clip_epsilon),
                )
                policy_loss = -torch.mean(torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages))
                value_loss = torch.mean((predicted_values - batch_returns) ** 2)
                entropy_bonus = torch.mean(entropy)
                total_loss = (
                    policy_loss
                    + float(self.config.value_coef) * value_loss
                    - float(self.config.entropy_coef) * entropy_bonus
                )

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.config.max_grad_norm))
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float(self.config.max_grad_norm))
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                policy_losses.append(float(policy_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                entropies.append(float(entropy_bonus.detach().cpu().item()))
                approx_kls.append(float((batch_old_log_probs - new_log_probs).mean().detach().cpu().item()))
                clip_fractions.append(
                    float((torch.abs(ratio - 1.0) > float(self.config.clip_epsilon)).float().mean().detach().cpu().item())
                )

        self.last_policy_loss = float(np.mean(policy_losses)) if policy_losses else None
        self.last_value_loss = float(np.mean(value_losses)) if value_losses else None
        self.last_entropy = float(np.mean(entropies)) if entropies else None
        self.last_approx_kl = float(np.mean(approx_kls)) if approx_kls else None
        self.last_clip_fraction = float(np.mean(clip_fractions)) if clip_fractions else None
        self.update_steps += 1
        self.buffer.clear()
        return transition_count

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {
            "format": "str-mappo-gnn-v1",
            "central_observation_size": int(self.central_observation_size),
            "action_size": int(self.action_size),
            "feature_layout": asdict(self.feature_layout),
            "config": asdict(self.config),
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "update_steps": int(self.update_steps),
            "transitions_collected": int(self.transitions_collected),
        }
        torch.save(checkpoint, path)


def _validate_checkpoint_feature_layout(
    checkpoint: Dict[str, object],
    graph_spec: RoutingGraphSpec,
) -> None:
    checkpoint_layout = dict(checkpoint.get("feature_layout") or {})
    runtime_layout = asdict(graph_spec.feature_layout)
    if checkpoint_layout != runtime_layout:
        raise ValueError(
            "Checkpoint graph feature layout {} is incompatible with runtime layout {}.".format(
                checkpoint_layout,
                runtime_layout,
            )
        )
    checkpoint_action_size = int(checkpoint.get("action_size", -1))
    if checkpoint_action_size != int(graph_spec.feature_layout.action_count):
        raise ValueError(
            "Checkpoint action_size {} is incompatible with runtime action_count {}.".format(
                checkpoint_action_size,
                int(graph_spec.feature_layout.action_count),
            )
        )


def load_mappo_checkpoint(
    path: str,
    graph_spec: RoutingGraphSpec,
    device: Optional[torch.device] = None,
):
    map_location = device if device is not None else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} is not a valid MAPPO checkpoint.")
    if checkpoint.get("format") != "str-mappo-gnn-v1":
        raise ValueError(
            f"{path} uses checkpoint format {checkpoint.get('format')!r}; "
            "expected 'str-mappo-gnn-v1'. Legacy vector checkpoints are intentionally unsupported."
        )

    _validate_checkpoint_feature_layout(checkpoint, graph_spec)

    config = MAPPOConfig(**dict(checkpoint.get("config") or {}))
    runtime_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = MAPPOActor(
        graph_spec,
        checkpoint["action_size"],
        config,
    ).to(runtime_device)
    critic = MAPPOCritic(
        graph_spec,
        checkpoint["central_observation_size"],
        config,
    ).to(runtime_device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    actor.eval()
    critic.eval()
    return actor, critic, checkpoint
