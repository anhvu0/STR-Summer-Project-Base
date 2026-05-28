from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import os

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


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


@dataclass(frozen=True)
class MAPPOConfig:
    actor_learning_rate: float = 3.0e-4
    critic_learning_rate: float = 1.0e-3
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.20
    entropy_coef: float = 0.030
    # If set, entropy_coef is linearly annealed toward entropy_coef_end over training.
    # Set higher than entropy_coef initially: e.g. entropy_coef=0.15, entropy_coef_end=0.01
    entropy_coef_end: Optional[float] = None
    value_coef: float = 0.50
    max_grad_norm: float = 10.0
    update_epochs: int = 6
    minibatch_size: int = 512
    normalize_advantages: bool = True
    min_transitions_per_update: int = 64
    target_kl: Optional[float] = 0.015
    actor_hidden_sizes: Tuple[int, ...] = (256, 128)
    critic_hidden_sizes: Tuple[int, ...] = (256, 128)


@dataclass
class MAPPOTransition:
    observation: np.ndarray
    central_observation: np.ndarray
    action: int
    action_mask: np.ndarray
    log_prob: float
    value: float
    reward: float
    next_observation: np.ndarray
    next_central_observation: np.ndarray
    done: bool
    discount_steps: int = 1
    metadata: Dict[str, object] = field(default_factory=dict)
    critic_only: bool = False


@dataclass(frozen=True)
class ActionSelection:
    action: int
    log_prob: float
    value: float
    entropy: float
    action_mask: np.ndarray


class MAPPOActor(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.observation_size = int(observation_size)
        self.action_size = int(action_size)
        self.network = _build_mlp(self.observation_size, hidden_sizes, self.action_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class MAPPOCritic(nn.Module):
    def __init__(self, observation_size: int, central_observation_size: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.observation_size = int(observation_size)
        self.central_observation_size = int(central_observation_size)
        critic_input_size = self.observation_size + self.central_observation_size
        self.network = _build_mlp(critic_input_size, hidden_sizes, 1)

    def forward(self, observations: torch.Tensor, central_observations: torch.Tensor) -> torch.Tensor:
        critic_input = torch.cat((observations, central_observations), dim=-1)
        return self.network(critic_input).squeeze(-1)


class MAPPOTrainer:
    def __init__(
        self,
        observation_size: int,
        central_observation_size: int,
        action_size: int,
        config: Optional[MAPPOConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.observation_size = int(observation_size)
        self.central_observation_size = int(central_observation_size)
        self.action_size = int(action_size)
        self.config = config or MAPPOConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = MAPPOActor(
            self.observation_size,
            self.action_size,
            self.config.actor_hidden_sizes,
        ).to(self.device)
        self.critic = MAPPOCritic(
            self.observation_size,
            self.central_observation_size,
            self.config.critic_hidden_sizes,
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
        self.buffer_generation: int = 0
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
        self.last_entropy_coef: float = float(self.config.entropy_coef)
        self._effective_entropy_coef: float = float(self.config.entropy_coef)

    @property
    def train_steps(self) -> int:
        return int(self.update_steps)

    def set_entropy_progress(self, progress: float) -> float:
        """Linearly anneal entropy_coef toward entropy_coef_end.

        Args:
            progress: Training progress in [0, 1] (episode / total_episodes).
        Returns:
            The effective entropy coefficient for this update.
        """
        end = self.config.entropy_coef_end
        if end is None:
            self._effective_entropy_coef = float(self.config.entropy_coef)
        else:
            p = float(np.clip(progress, 0.0, 1.0))
            self._effective_entropy_coef = float(
                self.config.entropy_coef + (end - self.config.entropy_coef) * p
            )
        self.last_entropy_coef = self._effective_entropy_coef
        return self._effective_entropy_coef

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
            "entropy_coef_effective",
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
            "entropy_coef_effective": self.last_entropy_coef,
            "approx_kl": self.last_approx_kl if self.last_approx_kl is not None else "",
            "clip_fraction": self.last_clip_fraction if self.last_clip_fraction is not None else "",
        }

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def select_action(
        self,
        observation: np.ndarray,
        valid_actions: Optional[Sequence[int]],
        central_observation: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> ActionSelection:
        observation_array = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        central_array = np.asarray(central_observation, dtype=np.float32).reshape(1, -1)
        action_mask = action_mask_from_valid_actions(self.action_size, valid_actions).reshape(1, -1)

        with torch.no_grad():
            observation_tensor = self._to_tensor(observation_array)
            central_tensor = self._to_tensor(central_array)
            action_mask_tensor = self._to_tensor(action_mask)
            logits = self.actor(observation_tensor)
            masked_logits = _masked_logits(logits, action_mask_tensor)
            distribution = Categorical(logits=masked_logits)
            if deterministic:
                action_tensor = torch.argmax(masked_logits, dim=-1)
            else:
                action_tensor = distribution.sample()
            log_prob_tensor = distribution.log_prob(action_tensor)
            entropy_tensor = distribution.entropy()
            value_tensor = self.critic(observation_tensor, central_tensor)

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
        observation: np.ndarray,
        central_observation: np.ndarray,
        action: int,
        action_mask: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        next_observation: np.ndarray,
        next_central_observation: np.ndarray,
        done: bool,
        discount_steps: int = 1,
        metadata: Optional[Dict[str, object]] = None,
        critic_only: bool = False,
    ) -> None:
        transition = MAPPOTransition(
            observation=np.asarray(observation, dtype=np.float32).reshape(-1),
            central_observation=np.asarray(central_observation, dtype=np.float32).reshape(-1),
            action=int(action),
            action_mask=np.asarray(action_mask, dtype=np.float32).reshape(-1),
            log_prob=float(log_prob),
            value=float(value),
            reward=float(reward),
            next_observation=np.asarray(next_observation, dtype=np.float32).reshape(-1),
            next_central_observation=np.asarray(next_central_observation, dtype=np.float32).reshape(-1),
            done=bool(done),
            discount_steps=max(int(discount_steps), 1),
            metadata=dict(metadata or {}),
            critic_only=bool(critic_only),
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
        observations: torch.Tensor,
        central_observations: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(observations)
        masked_logits = _masked_logits(logits, action_masks)
        distribution = Categorical(logits=masked_logits)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.critic(observations, central_observations)
        return log_probs, entropy, values

    def _compute_gae(
        self,
        rewards: np.ndarray,
        old_values: np.ndarray,
        next_values: np.ndarray,
        dones: np.ndarray,
        discount_steps: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        transition_count = len(rewards)
        advantages = np.zeros(transition_count, dtype=np.float32)

        # Group policy transitions by vehicle_id for per-trajectory GAE.
        # Critic-only samples are value targets for forced junction mechanics; keeping
        # them out of the route trajectory prevents macro-route advantages from being
        # chained through overlapping micro decisions.
        from collections import defaultdict
        traj_indices: Dict[str, List[int]] = defaultdict(list)
        solo_indices: List[int] = []
        for i, t in enumerate(self.buffer):
            vid = t.metadata.get("vehicle_id") if isinstance(t.metadata, dict) else None
            if bool(getattr(t, "critic_only", False)):
                solo_indices.append(i)
            elif vid:
                traj_indices[vid].append(i)
            else:
                solo_indices.append(i)

        gamma = float(self.config.gamma)
        lam = float(self.config.gae_lambda)

        for indices in list(traj_indices.values()) + [[i] for i in solo_indices]:
            gae = 0.0
            for i in reversed(indices):
                df = float(np.power(gamma, discount_steps[i]))
                delta = rewards[i] + df * next_values[i] * (1.0 - dones[i]) - old_values[i]
                gae = delta + df * lam * (1.0 - dones[i]) * gae
                advantages[i] = gae

        returns = advantages + old_values
        return advantages, returns

    def update(self) -> int:
        transition_count = len(self.buffer)
        if transition_count < int(self.config.min_transitions_per_update):
            return 0

        observations = np.stack([transition.observation for transition in self.buffer], axis=0)
        central_observations = np.stack([transition.central_observation for transition in self.buffer], axis=0)
        actions = np.asarray([transition.action for transition in self.buffer], dtype=np.int64)
        action_masks = np.stack([transition.action_mask for transition in self.buffer], axis=0)
        old_log_probs = np.asarray([transition.log_prob for transition in self.buffer], dtype=np.float32)
        old_values = np.asarray([transition.value for transition in self.buffer], dtype=np.float32)
        rewards = np.asarray([transition.reward for transition in self.buffer], dtype=np.float32)
        next_observations = np.stack([transition.next_observation for transition in self.buffer], axis=0)
        next_central_observations = np.stack(
            [transition.next_central_observation for transition in self.buffer],
            axis=0,
        )
        dones = np.asarray([transition.done for transition in self.buffer], dtype=np.float32)
        discount_steps = np.asarray([transition.discount_steps for transition in self.buffer], dtype=np.float32)
        critic_only_flags = np.asarray([transition.critic_only for transition in self.buffer], dtype=bool)
        critic_only_tensor = torch.as_tensor(critic_only_flags, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            next_obs_tensor = self._to_tensor(next_observations)
            next_central_tensor = self._to_tensor(next_central_observations)
            next_values = self.critic(next_obs_tensor, next_central_tensor).cpu().numpy()

        advantages, returns = self._compute_gae(rewards, old_values, next_values, dones, discount_steps)
        if self.config.normalize_advantages and transition_count > 1:
            policy_mask = ~critic_only_flags
            if policy_mask.sum() > 1:
                policy_advantages = advantages[policy_mask]
                advantages[policy_mask] = (
                    (policy_advantages - policy_advantages.mean())
                    / max(policy_advantages.std(), 1.0e-8)
                )

        obs_tensor = self._to_tensor(observations)
        central_tensor = self._to_tensor(central_observations)
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
        target_kl = self.config.target_kl
        kl_exceeded = False
        for _ in range(int(self.config.update_epochs)):
            if kl_exceeded:
                break
            np.random.shuffle(index_array)
            for start in range(0, transition_count, minibatch_size):
                batch_idx = index_array[start:start + minibatch_size]
                batch_obs = obs_tensor[batch_idx]
                batch_central = central_tensor[batch_idx]
                batch_actions = actions_tensor[batch_idx]
                batch_masks = action_mask_tensor[batch_idx]
                batch_old_log_probs = old_log_prob_tensor[batch_idx]
                batch_returns = returns_tensor[batch_idx]
                batch_advantages = advantage_tensor[batch_idx]

                batch_critic_only = critic_only_tensor[batch_idx]
                is_policy = ~batch_critic_only   # True for genuine policy decisions

                # critic_only transitions carry direction-level action indices (0-5) from the
                # forced branch, which are out of range for the route actor (action_size=4).
                # Clamp them to 0 — their log_probs are multiplied by a zero mask anyway.
                batch_actions_safe = batch_actions.clone()
                if batch_critic_only.any():
                    batch_actions_safe[batch_critic_only] = batch_actions_safe[batch_critic_only].clamp(0, self.action_size - 1)

                new_log_probs, entropy, predicted_values = self._evaluate_actions(
                    batch_obs,
                    batch_central,
                    batch_actions_safe,
                    batch_masks,
                )
                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - float(self.config.clip_epsilon),
                    1.0 + float(self.config.clip_epsilon),
                )
                # Policy loss and entropy only from genuine policy decisions (not critic_only forced transitions).
                surrogate = torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages)
                n_policy = is_policy.float().sum().clamp(min=1.0)
                policy_loss = -(surrogate * is_policy.float()).sum() / n_policy
                value_loss = torch.mean((predicted_values - batch_returns) ** 2)
                entropy_bonus = (entropy * is_policy.float()).sum() / n_policy
                total_loss = (
                    policy_loss
                    + float(self.config.value_coef) * value_loss
                    - float(self._effective_entropy_coef) * entropy_bonus
                )

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.config.max_grad_norm))
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float(self.config.max_grad_norm))
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # KL and clip metrics computed only on policy decisions.
                if is_policy.any():
                    batch_approx_kl = float(
                        (batch_old_log_probs[is_policy] - new_log_probs[is_policy]).mean().detach().cpu().item()
                    )
                    clip_frac = float(
                        (torch.abs(ratio[is_policy] - 1.0) > float(self.config.clip_epsilon))
                        .float().mean().detach().cpu().item()
                    )
                else:
                    batch_approx_kl = 0.0
                    clip_frac = 0.0
                policy_losses.append(float(policy_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                entropies.append(float(entropy_bonus.detach().cpu().item()))
                approx_kls.append(batch_approx_kl)
                clip_fractions.append(clip_frac)

                if target_kl is not None and batch_approx_kl > float(target_kl):
                    kl_exceeded = True
                    break

        self.last_policy_loss = float(np.mean(policy_losses)) if policy_losses else None
        self.last_value_loss = float(np.mean(value_losses)) if value_losses else None
        self.last_entropy = float(np.mean(entropies)) if entropies else None
        self.last_approx_kl = float(np.mean(approx_kls)) if approx_kls else None
        self.last_clip_fraction = float(np.mean(clip_fractions)) if clip_fractions else None
        self.update_steps += 1
        self.buffer.clear()
        self.buffer_generation += 1
        return transition_count

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {
            "format": "str-mappo-v1",
            "state_size": int(self.observation_size),
            "central_observation_size": int(self.central_observation_size),
            "action_size": int(self.action_size),
            "config": asdict(self.config),
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "update_steps": int(self.update_steps),
            "transitions_collected": int(self.transitions_collected),
        }
        torch.save(checkpoint, path)


def load_mappo_checkpoint(path: str, device: Optional[torch.device] = None):
    map_location = device if device is not None else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} is not a valid MAPPO checkpoint.")
    if checkpoint.get("format") != "str-mappo-v1":
        raise ValueError(
            f"{path} uses checkpoint format {checkpoint.get('format')!r}; expected 'str-mappo-v1'."
        )

    config = MAPPOConfig(**dict(checkpoint.get("config") or {}))
    runtime_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = MAPPOActor(
        checkpoint["state_size"],
        checkpoint["action_size"],
        config.actor_hidden_sizes,
    ).to(runtime_device)
    critic = MAPPOCritic(
        checkpoint["state_size"],
        checkpoint["central_observation_size"],
        config.critic_hidden_sizes,
    ).to(runtime_device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    actor.eval()
    critic.eval()
    return actor, critic, checkpoint
