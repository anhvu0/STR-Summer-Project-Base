"""
Entry point for training a reinforcement learning routing policy.

RIGHT HERE, DEFAULT SETTINGS USE FILES IN configurations folder. They affect the location of sumocfg file and xml file.
Files used here and files used in main.py must match. Otherwise -> Wrong dimensions

"""
import argparse

from core.rl_training_pipeline import RLTrainingPipeline


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
        default="./configurations/rl_model_test.h5",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--spawn-interval",
        type=float,
        default=2.0,
        help="Interval between vehicle spawns.",
    )
    parser.add_argument(
        "--debug-vehicle-id",
        action="append",
        default=[],
        help=(
            "Vehicle ID to trace in detail. Repeat this flag to trace multiple vehicles "
            "(for example: --debug-vehicle-id 60 --debug-vehicle-id 12)."
        ),
    )
    parser.add_argument(
        "--debug-log-path",
        default=None,
        help="Optional path to a JSONL debug log file for tracked vehicles.",
    )
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
        episodes=args.episodes,
        spawn_interval=args.spawn_interval,
        debug_vehicle_ids=args.debug_vehicle_id,
        debug_log_path=args.debug_log_path,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
