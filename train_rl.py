"""Entry point for training the MAPPO routing policy."""
import argparse

from core.mappo import MAPPOConfig
from core.rl_training_pipeline import RLTrainingPipeline


def parse_eval_seeds(raw_value):
    raw_value = (raw_value or "").strip()
    if not raw_value:
        return []
    return [int(token.strip()) for token in raw_value.split(",") if token.strip()]


def build_parser():
    """
    Build the CLI argument parser.
    """
    parser = argparse.ArgumentParser(description="Train a routing policy with MAPPO.")
    parser.add_argument(
        "--sumocfg",
        default="./configurations/myconfig.sumocfg",
        help="Path to SUMO .sumocfg file.",
    )       #If you change the sumocfg file, you need to retrain the model so it will reflect new files in there. At least for now until we can generalize routes and net

    parser.add_argument(
        "--model-output",
        default="./configurations/model/mappo_policy_nyc.pt",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--best-model-output",
        default="./configurations/model/mappo_policy_nyc.best.pt",
        help="Optional path for the best held-out frozen-eval checkpoint. Defaults to <model-output>.best.pt.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--spawn-interval",
        type=float,
        default=2.0,
        help="Interval between vehicle spawns.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=100,
        help="Run frozen held-out inference evaluation every N episodes. 0 disables frozen evaluation.",
    )
    parser.add_argument(
        "--eval-seeds",
        default="6000,6001,6002,6003,6004,6005,6006,6007,6008,6009,6010",
        help="Comma-separated held-out seeds for frozen inference evaluation.",
    )
    parser.add_argument(
        "--eval-spawn-interval",
        type=float,
        default=2.0,
        help="Optional spawn interval override for held-out frozen inference evaluation.",
    )
    parser.add_argument(
        "--fast-mode",
        dest="fast_mode",
        action="store_true",
        help="Enable the stripped runtime SUMO config and skip heavy output files for faster iteration.",
    )
    parser.add_argument(
        "--no-fast-mode",
        dest="fast_mode",
        action="store_false",
        help="Disable fast mode and keep the heavier debug outputs.",
    )
    parser.add_argument("--actor-lr", type=float, default=3.0e-4, help="MAPPO actor learning rate.")
    parser.add_argument("--critic-lr", type=float, default=1.0e-3, help="MAPPO critic learning rate.")
    parser.add_argument("--gamma", type=float, default=0.97, help="Discount factor for decision-level returns.")
    parser.add_argument("--clip-epsilon", type=float, default=0.20, help="PPO clipping coefficient.")
    parser.add_argument("--entropy-coef", type=float, default=0.02, help="Entropy bonus coefficient.")
    parser.add_argument("--value-coef", type=float, default=0.50, help="Value-loss coefficient.")
    parser.add_argument("--update-epochs", type=int, default=6, help="MAPPO epochs per episode rollout.")
    parser.add_argument("--minibatch-size", type=int, default=512, help="MAPPO minibatch size.")
    parser.add_argument("--graph-hidden-size", type=int, default=128, help="Hidden size for the GNN encoder.")
    parser.add_argument("--graph-layers", type=int, default=3, help="Number of message-passing layers.")
    parser.add_argument("--graph-dropout", type=float, default=0.0, help="Dropout applied inside the GNN encoder.")
    parser.add_argument(
        "--action-sampling-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature used for non-deterministic MAPPO training action sampling.",
    )
    parser.add_argument(
        "--valid-action-exploration-mix",
        type=float,
        default=0.04,
        help="Uniform valid-action probability mixed into MAPPO training samples; greedy inference is unchanged.",
    )
    parser.add_argument(
        "--min-transitions-per-update",
        type=int,
        default=64,
        help="Skip policy updates until at least this many decision transitions are collected.",
    )
    parser.set_defaults(fast_mode=True)
    return parser


def main():
    """
    Run the RL training pipeline.

    Better run main.py without arguments to avoid conflicts with files declared in main.py
    """
    parser = build_parser()
    args = parser.parse_args()
    mappo_config = MAPPOConfig(
        actor_learning_rate=args.actor_lr,
        critic_learning_rate=args.critic_lr,
        gamma=args.gamma,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        min_transitions_per_update=args.min_transitions_per_update,
        graph_hidden_size=args.graph_hidden_size,
        graph_layers=args.graph_layers,
        graph_dropout=args.graph_dropout,
        action_sampling_temperature=args.action_sampling_temperature,
        valid_action_exploration_mix=args.valid_action_exploration_mix,
    )
    pipeline = RLTrainingPipeline(
        sumocfg_path=args.sumocfg,
        model_output_path=args.model_output,
        best_model_output_path=args.best_model_output,
        episodes=args.episodes,
        spawn_interval=args.spawn_interval,
        mappo_config=mappo_config,
        eval_every=args.eval_every,
        frozen_eval_seeds=parse_eval_seeds(args.eval_seeds),
        eval_spawn_interval=args.eval_spawn_interval,
        fast_training_profile=args.fast_mode,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
