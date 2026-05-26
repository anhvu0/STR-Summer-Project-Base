from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class RoutingGraphFeatureLayout:
    node_static_dim: int
    node_dynamic_dim: int
    scalar_dim: int
    action_feature_dim: int
    action_count: int


@dataclass(frozen=True)
class RoutingGraphSpec:
    edge_ids: Tuple[str, ...]
    edge_id_to_index: Dict[str, int]
    edge_index: np.ndarray
    node_static_features: np.ndarray
    feature_layout: RoutingGraphFeatureLayout

    @property
    def node_count(self) -> int:
        return int(len(self.edge_ids))

    def zero_observation(self) -> "RoutingGraphObservation":
        return RoutingGraphObservation(
            node_dynamic_features=np.zeros(
                (self.node_count, self.feature_layout.node_dynamic_dim),
                dtype=np.float32,
            ),
            scalar_features=np.zeros(self.feature_layout.scalar_dim, dtype=np.float32),
            action_features=np.zeros(
                (self.feature_layout.action_count, self.feature_layout.action_feature_dim),
                dtype=np.float32,
            ),
            action_node_indices=np.full(
                self.feature_layout.action_count,
                -1,
                dtype=np.int64,
            ),
            current_node_index=0,
            destination_node_index=0,
        )


@dataclass(frozen=True)
class RoutingGraphObservation:
    node_dynamic_features: np.ndarray
    scalar_features: np.ndarray
    action_features: np.ndarray
    action_node_indices: np.ndarray
    current_node_index: int
    destination_node_index: int


@dataclass(frozen=True)
class RoutingGraphObservationBatch:
    node_dynamic_features: np.ndarray
    scalar_features: np.ndarray
    action_features: np.ndarray
    action_node_indices: np.ndarray
    current_node_index: np.ndarray
    destination_node_index: np.ndarray


def routing_graph_batch_to_tensors(
    batch: RoutingGraphObservationBatch,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        "node_dynamic_features": torch.as_tensor(
            batch.node_dynamic_features,
            dtype=torch.float32,
            device=device,
        ),
        "scalar_features": torch.as_tensor(
            batch.scalar_features,
            dtype=torch.float32,
            device=device,
        ),
        "action_features": torch.as_tensor(
            batch.action_features,
            dtype=torch.float32,
            device=device,
        ),
        "action_node_indices": torch.as_tensor(
            batch.action_node_indices,
            dtype=torch.long,
            device=device,
        ),
        "current_node_index": torch.as_tensor(
            batch.current_node_index,
            dtype=torch.long,
            device=device,
        ),
        "destination_node_index": torch.as_tensor(
            batch.destination_node_index,
            dtype=torch.long,
            device=device,
        ),
    }


def stack_routing_graph_observations(
    observations: Sequence[RoutingGraphObservation],
) -> RoutingGraphObservationBatch:
    if not observations:
        raise ValueError("observations must not be empty")

    return RoutingGraphObservationBatch(
        node_dynamic_features=np.stack(
            [np.asarray(obs.node_dynamic_features, dtype=np.float32) for obs in observations],
            axis=0,
        ),
        scalar_features=np.stack(
            [np.asarray(obs.scalar_features, dtype=np.float32).reshape(-1) for obs in observations],
            axis=0,
        ),
        action_features=np.stack(
            [np.asarray(obs.action_features, dtype=np.float32) for obs in observations],
            axis=0,
        ),
        action_node_indices=np.stack(
            [np.asarray(obs.action_node_indices, dtype=np.int64).reshape(-1) for obs in observations],
            axis=0,
        ),
        current_node_index=np.asarray(
            [int(obs.current_node_index) for obs in observations],
            dtype=np.int64,
        ),
        destination_node_index=np.asarray(
            [int(obs.destination_node_index) for obs in observations],
            dtype=np.int64,
        ),
    )
