import numpy as np
import torch

from core.mappo import MAPPOConfig, MAPPOTrainer, action_mask_from_valid_actions, load_mappo_checkpoint


def test_action_mask_from_valid_actions_marks_only_requested_actions():
    mask = action_mask_from_valid_actions(6, [0, 2, 5])
    assert mask.tolist() == [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]


def test_mappo_trainer_update_and_checkpoint_round_trip(tmp_path):
    config = MAPPOConfig(
        actor_hidden_sizes=(32, 16),
        critic_hidden_sizes=(32, 16),
        update_epochs=2,
        minibatch_size=16,
        min_transitions_per_update=8,
    )
    trainer = MAPPOTrainer(
        observation_size=10,
        central_observation_size=6,
        action_size=4,
        config=config,
        device=torch.device("cpu"),
    )

    rng = np.random.default_rng(7)
    for idx in range(16):
        observation = rng.normal(size=(1, 10)).astype(np.float32)
        central_observation = rng.normal(size=(1, 6)).astype(np.float32)
        next_observation = rng.normal(size=(1, 10)).astype(np.float32)
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
    actor, critic, checkpoint = load_mappo_checkpoint(str(checkpoint_path), device=torch.device("cpu"))

    assert checkpoint["format"] == "str-mappo-v1"
    assert checkpoint["state_size"] == 10
    assert checkpoint["central_observation_size"] == 6
    assert checkpoint["action_size"] == 4

    sample_observation = torch.as_tensor(rng.normal(size=(3, 10)).astype(np.float32))
    sample_central = torch.as_tensor(rng.normal(size=(3, 6)).astype(np.float32))
    with torch.no_grad():
        original_logits = trainer.actor(sample_observation)
        loaded_logits = actor(sample_observation)
        original_values = trainer.critic(sample_observation, sample_central)
        loaded_values = critic(sample_observation, sample_central)

    assert torch.allclose(original_logits, loaded_logits, atol=1e-6)
    assert torch.allclose(original_values, loaded_values, atol=1e-6)
