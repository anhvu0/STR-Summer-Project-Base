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
        default=100,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--spawn-interval",
        type=float,
        default=0.5,
        help="Interval between vehicle spawns.",
    )
    parser.add_argument(
        "--num-target-vehicles",
        type=int,
        default=450,
        help="Number of controlled (RL) vehicles per episode.",
    )
    parser.add_argument(
        "--num-random-vehicles",
        type=int,
        default=150,
        help="Number of uncontrolled background vehicles per episode.",
    )
    parser.add_argument(
        "--target-pattern",
        type=int,
        default=2,
        choices=[1, 2, 3, 4],
        help="Demand pattern: 1=one O/D, 2=ranged origins -> one shared destination "
             "(creates corridor congestion; use this for the NYC selfless-routing study), "
             "3=ranged origins -> ranged destinations (dispersed, ~no congestion), "
             "4=all topological sources -> single sink (directed bottleneck/braess funnel).",
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
        default=0.5,
        help="Optional spawn interval override for held-out frozen inference evaluation.",
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
        "--eval-stochastic-samples",
        type=int,
        default=3,
        help="Extra sampled-policy rollouts per seed in frozen eval. A greedy-argmax eval on "
             "fixed seeds is bit-identical across checkpoints until the argmax flips, so it "
             "cannot detect sub-argmax learning; the stochastic pass exposes it. 0 disables.",
    )
    parser.add_argument(
        "--disable-tail-delay-penalty",
        action="store_true",
        help="Zero the tail-delay penalty, which otherwise escalates exactly when a vehicle "
             "detours (structurally anti-selfless). Part of the arm-C 'full authority' config."
             "Include it in the cli to turn it to True. True is better because False actually add selfish noises.",
    )
    parser.add_argument(
        "--disable-route-balance",
        action="store_true",
        help="Zero the legacy hand-crafted route_balance reward so the principled team-reward "
             "term is the sole selfless driver. Part of the arm-C 'full authority' config."
             "Include it in the cli to turn it to True. True is better because False actually add selfish noises.",
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
        default=2048,
        help="Skip policy updates until at least this many decision transitions are collected.",
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
        choices=["difference", "shared", "marginal"],
        default="difference",
        help="How the fleet-congestion term is credited to each agent. "
             "difference=leave-one-out (each agent internalizes its slowness relative to "
             "the live fleet; the exogenous demand-driven common level cancels -> far higher "
             "learning signal-to-noise). shared=legacy global fleet-delay level paid identically "
             "by every agent. marginal=Pigovian marginal-cost/externality pricing: while on a "
             "congestible (single-lane VAR) edge an agent pays per step for the vehicles queued "
             "BEHIND it (scaled by its own slowness, 0 at free flow) -- the delay it imposes on "
             "followers. Unlike difference (which credits a selfless deviator negatively), the "
             "marginal charge is non-negative, per-agent attributable, and largest exactly when "
             "the fleet herds onto the Braess route, so it prices the coordination externality "
             "directly (EXPERIMENT_PLAN E5). Needs team_reward_alpha > 0 to have any effect.",
    )
    parser.add_argument(
        "--marginal-cost-scale",
        type=float,
        default=0.015,
        help="Per-step charge (travel-time units) per queued-follower for --team-reward-mode "
             "marginal, before scaling by --team-reward-alpha. The step charge is "
             "alpha * scale * min(vehicles_behind, 30) * own_slowness, capped at 0.6/step.",
    )
    parser.add_argument(
        "--skip-forced-route-epochs",
        action="store_true",
        help="Skip route-actor epochs that reach a decision point with <=1 distinct immediate "
             "next edge (a forced continuation, no real fork). Such epochs otherwise record "
             "near-deterministic single-fork policy transitions that dilute mean approx_kl and "
             "pad the batch; skipping concentrates the gradient on the genuine forks. Off by "
             "default (legacy NYC training unchanged); recommended ON for the Braess funnel with "
             "--reroute-epoch-edges 1.",
    )
    parser.add_argument(
        "--reroute-epoch-edges",
        type=int,
        default=5,
        help="Re-query the route policy every N completed edges (counter starts at N so "
             "the first decision fires on the 1st edge). NYC grid uses 5; the chained-"
             "Braess funnel needs 1 so the policy re-decides at every edge and hits both "
             "forks (on `stage` before diamond 1, on `link1` before diamond 2).",
    )
    parser.add_argument(
        "--disable-route-reservations",
        action="store_true",
        help="Disable the Layer B anticipatory reservation field. When ON (default), a "
             "committed route books its leading edges in a decaying field so the route-"
             "candidate generator scores against effective (live + reserved) density, "
             "damping the simultaneous detour pile-on. See docs/coordination_throttle.md.",
    )
    parser.set_defaults(fast_mode=True, normalize_value_targets=True)
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
        marginal_cost_scale=args.marginal_cost_scale,
        skip_forced_route_epochs=args.skip_forced_route_epochs,
        eval_deterministic=(args.eval_policy == "greedy"),
        route_reservations=not args.disable_route_reservations,
        eval_stochastic_samples=args.eval_stochastic_samples,
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
