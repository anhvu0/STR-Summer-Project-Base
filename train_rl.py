"""Entry point for training the MAPPO routing policy."""
import argparse
import sys

# --- Use libsumo as a drop-in TraCI backend for training only ----------------
# libsumo runs SUMO in-process and skips the per-call TraCI socket round-trips
# that dominate the step loop. It is API- and value-compatible with traci (same
# functions, same constants, same returned values), so this changes no logic or
# metrics -- only the transport. We register it in sys.modules *before* importing
# the pipeline/controller so their top-level `import traci` and
# `from traci import constants` transparently resolve to libsumo in THIS process
# only. main.py is a separate process that never runs this file, so its GUI traci
# is untouched. If libsumo is unavailable we silently fall back to real traci.
try:
    import libsumo as _libsumo
    sys.modules["traci"] = _libsumo
    sys.modules["traci.constants"] = _libsumo.constants
except ImportError:
    pass
# -----------------------------------------------------------------------------

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
        default="./configurations/bottleneck.sumocfg",
        help="Path to SUMO .sumocfg file. Default is the high-price-of-anarchy bottleneck "
             "map built for the selfless-routing study (docs/bottleneck_map_design.md); "
             "use ./configurations/myconfig.sumocfg for the legacy NYC grid.",
    )       #If you change the sumocfg file, you need to retrain the model so it will reflect new files in there. At least for now until we can generalize routes and net

    parser.add_argument(
        "--model-output",
        default="./configurations/model/mappo_policy_bottleneck.pt",
        help="Path to save the trained model. NYC-grid checkpoints live at "
             "./configurations/model/mappo_policy_nyc.pt; keep map and checkpoint paired.",
    )
    parser.add_argument(
        "--best-model-output",
        default="./configurations/model/mappo_policy_bottleneck.best.pt",
        help="Optional path for the best held-out frozen-eval checkpoint. Defaults to <model-output>.best.pt.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--spawn-interval",
        type=float,
        default=1.0,
        help="Interval between vehicle spawns. 1.0 on the bottleneck map ~= 1 veh/s "
             "controlled demand: ~2x the bottleneck's capacity (selfish herding jams it) "
             "while total network capacity has slack (detours can absorb the excess).",
    )
    parser.add_argument(
        "--num-target-vehicles",
        type=int,
        default=300,
        help="Number of controlled (RL) vehicles per episode. Calibrated for the "
             "bottleneck map (see docs/bottleneck_map_design.md); the NYC-grid study "
             "used 350-450.",
    )
    parser.add_argument(
        "--num-random-vehicles",
        type=int,
        default=100,
        help="Number of uncontrolled background vehicles per episode.",
    )
    parser.add_argument(
        "--target-pattern",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Demand pattern: 1=one O/D, 2=ranged origins -> one shared destination "
             "(corridor congestion on grid maps), 3=ranged origins -> ranged destinations "
             "(dispersed, ~no congestion), 4=corridor: all source edges -> the sink edge "
             "(bottleneck maps; every vehicle faces the bottleneck-vs-detour dilemma).",
    )
    parser.add_argument(
        "--reroute-epoch-edges",
        type=int,
        default=2,
        help="Re-query the route policy every N completed edges. The first decision "
             "fires on the Nth edge of a trip, so on the bottleneck map this must be "
             "<=2 for the policy to decide at the fork (2 = decide exactly on the "
             "staging edge). The NYC-grid study used 5.",
    )
    parser.add_argument(
        "--team-reward-alpha",
        type=float,
        default=1.0,
        help="Weight in [0,1] on the shared fleet-congestion cost internalized by each "
             "agent. 0 = purely individual (selfish) objective; >0 rewards relieving "
             "congestion for the whole fleet (selfless routing). Recommended: 1.0.",
    )
    parser.add_argument(
        "--team-reward-scale",
        type=float,
        default=0.30,
        help="Per-step magnitude (travel-time units) of the shared fleet-congestion cost "
             "before scaling by --team-reward-alpha.",
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
        default=1.0,
        help="Optional spawn interval override for held-out frozen inference evaluation. "
             "Keep equal to --spawn-interval so eval demand matches training demand.",
    )
    parser.add_argument(
        "--eval-policy",
        choices=["greedy", "stochastic"],
        default="greedy",
        help="Deployment / frozen-eval route selection mode. greedy=argmax (reproducible); "
             "stochastic=sample from the policy so the fleet spreads across alternative routes "
             "(option (b) in docs/selfless_routing_analysis.md).",
    )
    parser.add_argument(
        "--disable-tail-delay-penalty",
        dest="disable_tail_delay_penalty",
        action="store_true",
        help="Zero the tail-delay penalty, which otherwise escalates exactly when a vehicle "
             "detours (structurally anti-selfless). DEFAULT ON (arm-C 'full authority'); "
             "use --enable-tail-delay-penalty to restore the legacy term.",
    )
    parser.add_argument(
        "--enable-tail-delay-penalty",
        dest="disable_tail_delay_penalty",
        action="store_false",
        help="Restore the legacy tail-delay penalty (anti-selfless; for A/B against the "
             "legacy objective only).",
    )
    parser.add_argument(
        "--disable-route-balance",
        dest="disable_route_balance",
        action="store_true",
        help="Zero the legacy hand-crafted route_balance reward so the principled team-reward "
             "term is the sole selfless driver. DEFAULT ON (arm-C 'full authority'); "
             "use --enable-route-balance to restore the legacy term.",
    )
    parser.add_argument(
        "--enable-route-balance",
        dest="disable_route_balance",
        action="store_false",
        help="Restore the legacy route_balance proxy reward (for A/B against the legacy "
             "objective only).",
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
    parser.add_argument("--gamma", type=float, default=0.995,
                        help="Discount factor for simulation-step returns. Higher default preserves delayed selfless-routing effects.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda for advantage estimation.")
    parser.add_argument("--clip-epsilon", type=float, default=0.05, help="PPO clipping coefficient.")
    parser.add_argument("--entropy-coef", type=float, default=0.15,
                        help="Initial entropy bonus coefficient. Raised from 0.03 to combat entropy collapse (see diagnosis).")
    parser.add_argument("--entropy-coef-end", type=float, default=0.05,
                        help="Final entropy coefficient after linear annealing over all training episodes. "
                             "Set equal to --entropy-coef to disable annealing.")
    parser.add_argument("--value-coef", type=float, default=0.50, help="Value-loss coefficient.")
    parser.add_argument("--update-epochs", type=int, default=4, help="MAPPO epochs per episode rollout.")
    parser.add_argument("--minibatch-size", type=int, default=512, help="MAPPO minibatch size.")
    parser.add_argument("--target-kl", type=float, default=0.015, help="KL divergence threshold for early stopping per update epoch. 0 disables.")
    parser.add_argument(
        "--min-transitions-per-update",
        type=int,
        default=768,
        help="Skip policy updates until at least this many decision transitions are collected. "
             "The bottleneck map yields ~2-3 decisions/vehicle (~900/episode at defaults), so "
             "768 keeps roughly one PPO update per episode; the NYC grid collected ~2500/episode.",
    )
    parser.add_argument(
        "--no-value-normalization",
        dest="normalize_value_targets",
        action="store_false",
        help="Disable running-statistics normalization of the critic's value target. "
             "Normalization is ON by default and keeps the value loss well-conditioned "
             "despite the large episode-to-episode swing in raw return scale.",
    )
    parser.add_argument(
        "--team-reward-mode",
        choices=["difference", "shared"],
        default="difference",
        help="How the fleet-congestion term is credited to each agent. "
             "difference=leave-one-out (each agent internalizes its slowness relative to "
             "the live fleet; the exogenous demand-driven common level cancels -> far higher "
             "learning signal-to-noise). shared=legacy global fleet-delay level paid identically "
             "by every agent. Recommended: difference."
             "Needs team_reward_alpha > 0 to have any effect.",
    )
    parser.add_argument(
        "--disable-route-reservations",
        action="store_true",
        help="Disable the Layer B anticipatory reservation field. When ON (default), a "
             "committed route books its leading edges in a decaying field so the route-"
             "candidate generator scores against effective (live + reserved) density, "
             "damping the simultaneous detour pile-on. See docs/coordination_throttle.md.",
    )
    # Arm-C "full authority" objective is the default: the two legacy reward terms
    # that structurally oppose selfless detours are zeroed unless explicitly re-enabled
    # (docs/selfless_routing_analysis.md §4.0/§4.3 - the config that made the greedy
    # policy genuinely selfless: 14.1s ETA sacrifice, 27% non-baseline, CIs clear).
    parser.set_defaults(
        fast_mode=True,
        normalize_value_targets=True,
        disable_tail_delay_penalty=True,
        disable_route_balance=True,
    )
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
        normalize_value_targets=args.normalize_value_targets,
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
        target_pattern=args.target_pattern,
        num_target_vehicles=args.num_target_vehicles,
        num_random_vehicles=args.num_random_vehicles,
        team_reward_alpha=args.team_reward_alpha,
        team_reward_scale=args.team_reward_scale,
        team_reward_mode=args.team_reward_mode,
        eval_deterministic=(args.eval_policy == "greedy"),
        route_reservations=not args.disable_route_reservations,
        reroute_epoch_edges=args.reroute_epoch_edges,
    )
    if args.disable_tail_delay_penalty:
        pipeline.tail_delay_linear_penalty = 0.0
        pipeline.tail_delay_quadratic_penalty = 0.0
    if args.disable_route_balance:
        pipeline.route_balance_reward_scale = 0.0
    pipeline.run()


if __name__ == "__main__":
    main()
