"""
Entry point for training a reinforcement learning routing policy.

RIGHT HERE, DEFAULT SETTINGS USE FILES IN configurations folder. They affect the location of sumocfg file and xml file.
Files used here and files used in main.py must match. Otherwise -> Wrong dimensions

"""
import argparse

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
    parser = argparse.ArgumentParser(description="Train a routing policy with DQN.")
    parser.add_argument(
        "--sumocfg",
        default="./configurations/myconfig.sumocfg",
        help="Path to SUMO .sumocfg file.",
    )       #If you change the sumocfg file, you need to retrain the model so it will reflect new files in there. At least for now until we can generalize routes and net

    parser.add_argument(
        "--model-output",
        default="./configurations/model/rl_model_nyc.pt",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--best-model-output",
        default="./configurations/model/rl_model_nyc.best.pt",
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
        default=50,
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
    parser.add_argument(
        "--double-dqn",
        dest="use_double_dqn",
        action="store_true",
        help="Use online argmax plus target-network evaluation for replay bootstrapping.",
    )
    parser.add_argument(
        "--no-double-dqn",
        dest="use_double_dqn",
        action="store_false",
        help="Use the target network for both replay action selection and evaluation.",
    )
    parser.set_defaults(fast_mode=True, use_double_dqn=True)
    return parser


def main():
    """
    Run the RL training pipeline.

    Better run main.py without arguments to avoid conflicts with files declared in main.py
    """
    parser = build_parser()
    args = parser.parse_args()
    pipeline = RLTrainingPipeline(
        sumocfg_path=args.sumocfg,
        model_output_path=args.model_output,
        best_model_output_path=args.best_model_output,
        episodes=args.episodes,
        spawn_interval=args.spawn_interval,
        eval_every=args.eval_every,
        frozen_eval_seeds=parse_eval_seeds(args.eval_seeds),
        eval_spawn_interval=args.eval_spawn_interval,
        fast_training_profile=args.fast_mode,
        use_double_dqn=args.use_double_dqn,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
