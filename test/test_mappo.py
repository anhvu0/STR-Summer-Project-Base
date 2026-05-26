import numpy as np
import torch

from core.mappo import MAPPOConfig, MAPPOTrainer, action_mask_from_valid_actions, load_mappo_checkpoint
from core.routing_graph import (
    RoutingGraphFeatureLayout,
    RoutingGraphObservation,
    RoutingGraphSpec,
    stack_routing_graph_observations,
)


def test_action_mask_from_valid_actions_marks_only_requested_actions():
    mask = action_mask_from_valid_actions(6, [0, 2, 5])
    assert mask.tolist() == [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]


def _build_test_graph_spec():
    feature_layout = RoutingGraphFeatureLayout(
        node_static_dim=3,
        node_dynamic_dim=4,
        scalar_dim=5,
        action_feature_dim=2,
        action_count=4,
    )
    return RoutingGraphSpec(
        edge_ids=("e0", "e1", "e2"),
        edge_id_to_index={"e0": 0, "e1": 1, "e2": 2},
        edge_index=np.asarray(
            [
                [0, 0, 1, 1, 2, 2],
                [0, 1, 0, 2, 1, 2],
            ],
            dtype=np.int64,
        ),
        node_static_features=np.asarray(
            [
                [0.5, 0.2, 0.0],
                [0.7, 0.4, 0.2],
                [0.9, 0.6, 0.4],
            ],
            dtype=np.float32,
        ),
        feature_layout=feature_layout,
    )


def _random_observation(rng, graph_spec):
    return RoutingGraphObservation(
        node_dynamic_features=rng.normal(
            size=(graph_spec.node_count, graph_spec.feature_layout.node_dynamic_dim)
        ).astype(np.float32),
        scalar_features=rng.normal(
            size=(graph_spec.feature_layout.scalar_dim,)
        ).astype(np.float32),
        action_features=rng.normal(
            size=(graph_spec.feature_layout.action_count, graph_spec.feature_layout.action_feature_dim)
        ).astype(np.float32),
        action_node_indices=np.asarray([0, 1, 2, -1], dtype=np.int64),
        current_node_index=int(rng.integers(0, graph_spec.node_count)),
        destination_node_index=int(rng.integers(0, graph_spec.node_count)),
    )


def test_mappo_trainer_update_and_checkpoint_round_trip(tmp_path):
    graph_spec = _build_test_graph_spec()
    config = MAPPOConfig(
        actor_hidden_sizes=(32, 16),
        critic_hidden_sizes=(32, 16),
        update_epochs=2,
        minibatch_size=16,
        min_transitions_per_update=8,
        graph_hidden_size=24,
        graph_layers=2,
    )
    trainer = MAPPOTrainer(
        graph_spec=graph_spec,
        central_observation_size=6,
        action_size=4,
        config=config,
        device=torch.device("cpu"),
    )

    rng = np.random.default_rng(7)
    for idx in range(16):
        observation = _random_observation(rng, graph_spec)
        central_observation = rng.normal(size=(1, 6)).astype(np.float32)
        next_observation = _random_observation(rng, graph_spec)
        next_central_observation = rng.normal(size=(1, 6)).astype(np.float32)
        selection = trainer.select_action(
            observation,
            [0, 1, 3],
            central_observation,
            deterministic=(idx % 2 == 0),
        )
        trainer.record_transition(
            observation=observation,
            central_observation=central_observation,
            action=selection.action,
            action_mask=selection.action_mask,
            log_prob=selection.log_prob,
            value=selection.value,
            reward=float(rng.normal()),
            next_observation=next_observation,
            next_central_observation=next_central_observation,
            done=bool(idx % 5 == 0),
            discount_steps=1 + (idx % 3),
            metadata={"idx": idx},
        )

    updated_count = trainer.update()
    assert updated_count == 16
    assert trainer.train_steps == 1
    assert len(trainer.memory) == 0
    assert trainer.last_policy_loss is not None
    assert trainer.last_value_loss is not None

    checkpoint_path = tmp_path / "mappo.pt"
    trainer.save_checkpoint(str(checkpoint_path))
    actor, critic, checkpoint = load_mappo_checkpoint(
        str(checkpoint_path),
        graph_spec=graph_spec,
        device=torch.device("cpu"),
    )

    assert checkpoint["format"] == "str-mappo-gnn-v1"
    assert checkpoint["central_observation_size"] == 6
    assert checkpoint["action_size"] == 4

    sample_observations = [_random_observation(rng, graph_spec) for _ in range(3)]
    sample_batch = stack_routing_graph_observations(sample_observations)
    sample_observation_tensors = {
        "node_dynamic_features": torch.as_tensor(sample_batch.node_dynamic_features, dtype=torch.float32),
        "scalar_features": torch.as_tensor(sample_batch.scalar_features, dtype=torch.float32),
        "action_features": torch.as_tensor(sample_batch.action_features, dtype=torch.float32),
        "action_node_indices": torch.as_tensor(sample_batch.action_node_indices, dtype=torch.long),
        "current_node_index": torch.as_tensor(sample_batch.current_node_index, dtype=torch.long),
        "destination_node_index": torch.as_tensor(sample_batch.destination_node_index, dtype=torch.long),
    }
    sample_central = torch.as_tensor(rng.normal(size=(3, 6)).astype(np.float32))
    with torch.no_grad():
        original_logits = trainer.actor(sample_observation_tensors)
        loaded_logits = actor(sample_observation_tensors)
        original_values = trainer.critic(sample_observation_tensors, sample_central)
        loaded_values = critic(sample_observation_tensors, sample_central)

    assert torch.allclose(original_logits, loaded_logits, atol=1e-6)
    assert torch.allclose(original_values, loaded_values, atol=1e-6)
