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
        default="./configurations/rl_model_map.h5",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=330,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--spawn-interval",
        type=float,
        default=5.5,
        help="Interval between vehicle spawns.",
    )
    parser.add_argument(
        "--train-every",
        type=int,
        default=20,
        help="Run replay updates every N simulation steps.",
    )
    parser.add_argument(
        "--max-sim-steps",
        type=int,
        default=2200,
        help="Upper bound on per-episode simulation steps.",
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
        train_every=args.train_every,
        max_simulation_steps=args.max_sim_steps,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
