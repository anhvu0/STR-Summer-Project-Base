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
        default=1000,
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
        default=25,
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
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda for advantage estimation.")
    parser.add_argument("--clip-epsilon", type=float, default=0.15, help="PPO clipping coefficient.")
    parser.add_argument("--entropy-coef", type=float, default=0.15,
                        help="Initial entropy bonus coefficient. Raised from 0.03 to combat entropy collapse (see diagnosis).")
    parser.add_argument("--entropy-coef-end", type=float, default=0.01,
                        help="Final entropy coefficient after linear annealing over all training episodes. "
                             "Set equal to --entropy-coef to disable annealing.")
    parser.add_argument("--value-coef", type=float, default=0.50, help="Value-loss coefficient.")
    parser.add_argument("--update-epochs", type=int, default=6, help="MAPPO epochs per episode rollout.")
    parser.add_argument("--minibatch-size", type=int, default=512, help="MAPPO minibatch size.")
    parser.add_argument("--target-kl", type=float, default=0.015, help="KL divergence threshold for early stopping per update epoch. 0 disables.")
    parser.add_argument(
        "--min-transitions-per-update",
        type=int,
        default=2048,
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
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        entropy_coef_end=args.entropy_coef_end,
        value_coef=args.value_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        min_transitions_per_update=args.min_transitions_per_update,
        target_kl=args.target_kl if args.target_kl > 0 else None,
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
