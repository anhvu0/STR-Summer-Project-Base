from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
import random
import numpy as np


@dataclass
class BranchOption:
    option_id: int
    outgoing_edge: str
    movement_label: str
    target_lane_set: tuple[int, ...]
    required_lane_shift: int
    min_execution_horizon_steps: int
    commit_start_distance_m: float
    success_condition: str
    failure_conditions: tuple[str, ...]
    currently_executable: bool
    tactical_risk_flags: dict
    option_features: np.ndarray | None = None


@dataclass
class StrategicDecisionContext:
    vehicle_id: str
    step: int
    current_edge: str
    lane_index: int
    lane_count: int
    dist_to_end: float
    speed: float
    shared_state: np.ndarray
    options: list[BranchOption]


@dataclass
class StrategicTransition:
    state_shared: np.ndarray
    state_option_features: list[np.ndarray]
    chosen_option_idx: int
    aggregated_reward: float
    next_state_shared: np.ndarray | None
    next_state_option_features: list[np.ndarray] | None
    done: bool
    outcome: str
    tactical_outcome: str | None
    horizon_steps: int


@dataclass
class ActiveStrategicDecision:
    context: StrategicDecisionContext
    chosen_option_idx: int
    chosen_option: BranchOption
    opened_step: int
    reward_accumulator: float = 0.0
    horizon_steps: int = 0
    tactical_state: dict = field(default_factory=dict)


class StrategicReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = max(int(capacity), 1)
        self._buffer: Deque[StrategicTransition] = deque(maxlen=self.capacity)

    def add(self, transition: StrategicTransition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> List[StrategicTransition]:
        return random.sample(self._buffer, min(batch_size, len(self._buffer)))

    def __len__(self) -> int:
        return len(self._buffer)


class ReplayManager:
    def __init__(
        self,
        strategic_capacity: int = 100_000,
        tactical_failure_capacity: int = 40_000,
        terminal_only_capacity: int = 20_000,
        terminal_sampling_cap: float = 0.15,
        tactical_sampling_ratio: float = 0.10,
    ):
        self.strategic_replay = StrategicReplayBuffer(strategic_capacity)
        self.tactical_failure_replay = StrategicReplayBuffer(tactical_failure_capacity)
        self.terminal_only_replay = StrategicReplayBuffer(terminal_only_capacity)
        self.terminal_sampling_cap = float(np.clip(terminal_sampling_cap, 0.0, 0.8))
        self.tactical_sampling_ratio = float(np.clip(tactical_sampling_ratio, 0.0, 0.8))

    def add_strategic(self, transition: StrategicTransition) -> None:
        self.strategic_replay.add(transition)

    def add_tactical_failure(self, transition: StrategicTransition) -> None:
        self.tactical_failure_replay.add(transition)

    def add_terminal_only(self, transition: StrategicTransition) -> None:
        self.terminal_only_replay.add(transition)

    def sample_for_training(self, batch_size: int) -> Tuple[List[StrategicTransition], Dict[str, float]]:
        batch_size = max(int(batch_size), 1)
        n_tactical = int(round(batch_size * self.tactical_sampling_ratio))
        n_terminal = int(round(batch_size * self.terminal_sampling_cap))
        n_main = max(batch_size - n_tactical - n_terminal, 1)

        main = self.strategic_replay.sample(n_main)
        tactical = self.tactical_failure_replay.sample(n_tactical) if n_tactical > 0 else []
        terminal = self.terminal_only_replay.sample(n_terminal) if n_terminal > 0 else []
        merged = main + tactical + terminal
        random.shuffle(merged)

        total = float(max(len(merged), 1))
        fractions = {
            "strategic_fraction": len(main) / total,
            "tactical_failure_fraction": len(tactical) / total,
            "terminal_only_fraction": len(terminal) / total,
            "strategic_count": len(main),
            "tactical_failure_count": len(tactical),
            "terminal_only_count": len(terminal),
            "total_count": len(merged),
        }
        return merged, fractions
