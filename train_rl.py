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
        "--target-vehicles",
        type=int,
        default=10,
        help="Number of controlled vehicles per episode.",
    )
    parser.add_argument(
        "--random-vehicles",
        type=int,
        default=30,
        help="Number of uncontrolled background vehicles per episode.",
    )
    parser.add_argument(
        "--random-vehicle-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for background traffic demand during route generation.",
    )
    parser.add_argument(
        "--teleport-time",
        type=int,
        default=300,
        help="SUMO teleport timeout in seconds (-1 disables teleporting).",
    )
    parser.add_argument(
        "--stuck-wait-time-limit",
        type=float,
        default=120.0,
        help="Waiting-time threshold (s) to mark a controlled vehicle as stuck.",
    )
    parser.add_argument(
        "--stuck-terminal-penalty",
        type=float,
        default=-120.0,
        help="Terminal reward applied when a controlled vehicle is removed for being stuck.",
    )
    parser.add_argument(
        "--wait-time-penalty-scale",
        type=float,
        default=0.05,
        help="Per-second waiting-time penalty scale used in reward shaping.",
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
        num_target_vehicles=args.target_vehicles,
        num_random_vehicles=args.random_vehicles,
        random_vehicle_multiplier=args.random_vehicle_multiplier,
        teleport_time=args.teleport_time,
        stuck_wait_time_limit=args.stuck_wait_time_limit,
        stuck_terminal_penalty=args.stuck_terminal_penalty,
        wait_time_penalty_scale=args.wait_time_penalty_scale,
    )
    pipeline.run()


if __name__ == "__main__":
    main()