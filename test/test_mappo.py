import os
import numpy as np
import torch
from types import SimpleNamespace

from core.mappo import MAPPOConfig, MAPPOTrainer, action_mask_from_valid_actions, load_mappo_checkpoint

os.environ.setdefault("SUMO_HOME", ".")

from core.rl_training_pipeline import RLTrainingPipeline


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


def test_route_candidate_scorer_masks_and_round_trips(tmp_path):
    config = MAPPOConfig(
        actor_hidden_sizes=(24,),
        critic_hidden_sizes=(24,),
        route_candidate_feature_dim=3,
        update_epochs=1,
        minibatch_size=8,
        min_transitions_per_update=4,
    )
    trainer = MAPPOTrainer(
        observation_size=8 + 4 * 3,
        central_observation_size=5,
        action_size=4,
        config=config,
        device=torch.device("cpu"),
    )
    assert trainer.actor.route_candidate_feature_dim == 3
    assert trainer.actor.candidate_scorer is not None

    rng = np.random.default_rng(11)
    observation = rng.normal(size=(1, 20)).astype(np.float32)
    central_observation = rng.normal(size=(1, 5)).astype(np.float32)
    selection = trainer.select_action(
        observation,
        [1, 3],
        central_observation,
        deterministic=True,
    )

    assert selection.action in {1, 3}
    assert selection.masked_logits.shape == (4,)
    assert selection.masked_logits[0] < -1.0e8
    assert selection.masked_logits[2] < -1.0e8

    checkpoint_path = tmp_path / "route_mappo.pt"
    trainer.save_checkpoint(str(checkpoint_path))
    actor, _, checkpoint = load_mappo_checkpoint(str(checkpoint_path), device=torch.device("cpu"))

    assert checkpoint["config"]["route_candidate_feature_dim"] == 3
    sample_observation = torch.as_tensor(rng.normal(size=(3, 20)).astype(np.float32))
    with torch.no_grad():
        original_logits = trainer.actor(sample_observation)
        loaded_logits = actor(sample_observation)

    assert torch.allclose(original_logits, loaded_logits, atol=1e-6)


def test_route_epoch_step_reward_accumulates_elapsed_time():
    pipeline = RLTrainingPipeline.__new__(RLTrainingPipeline)
    pipeline.compute_pending_step_reward = (
        lambda vehicle, edge_id, elapsed, step, **kwargs: -float(elapsed)
    )

    trace = {
        "mappo_training": True,
        "mappo_reward_accumulator": 0.0,
        "route_last_credit_step": 10,
        "route_elapsed_steps": 0,
    }
    reward = RLTrainingPipeline._accumulate_route_epoch_step_reward(
        pipeline,
        {"veh": trace},
        "veh",
        SimpleNamespace(),
        "edgeA",
        13,
    )

    assert reward == -3.0
    assert trace["mappo_reward_accumulator"] == -3.0
    assert trace["route_elapsed_steps"] == 3
    assert trace["route_last_credit_step"] == 13

    reward = RLTrainingPipeline._accumulate_route_epoch_step_reward(
        pipeline,
        {"veh": trace},
        "veh",
        SimpleNamespace(),
        "edgeA",
        13,
    )
    assert reward == 0.0
    assert trace["mappo_reward_accumulator"] == -3.0


def test_episode_density_state_reset_allows_step_zero_refresh():
    pipeline = RLTrainingPipeline.__new__(RLTrainingPipeline)
    pipeline._edge_list = ("edgeA", "edgeB")
    pipeline.connection_info = SimpleNamespace(edge_vehicle_count={"edgeA": 4, "edgeB": 7})
    pipeline._density_vec = np.ones(2, dtype=np.float32)
    pipeline._density_mean = 1.5
    pipeline._density_std = 0.5
    pipeline._density_p95 = 2.0
    pipeline._last_density_step = 999

    RLTrainingPipeline._reset_episode_density_state(pipeline)

    assert pipeline.connection_info.edge_vehicle_count == {"edgeA": 0, "edgeB": 0}
    assert pipeline._density_vec.tolist() == [0.0, 0.0]
    assert pipeline._density_mean == 0.0
    assert pipeline._density_std == 0.0
    assert pipeline._density_p95 == 0.0
    assert pipeline._last_density_step < 0
