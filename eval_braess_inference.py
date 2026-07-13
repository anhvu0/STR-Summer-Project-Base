"""Inference eval for the Braess map: does the trained policy capture the
coordination gap (approach T_SO), or does it herd like the selfish equilibrium?

For each held-out seed it generates the SAME 240-vehicle pattern-4 demand at the
TRAINING spawn interval (1.5s, NOT the pipeline's dense 0.5s frozen-eval default,
which gridlocks this map) and runs every controller through StrSumo, exactly like
RLTrainingPipeline._run_frozen_inference_eval.run_eval_controller. Because ALL
arms (MAPPO and both Dijkstra baselines) run through the identical harness, the
per-seed avg travel time is measured the same way (from detected release to
arrival), so the paired Wilcoxon tests are apples-to-apples.

Arms (all per-seed, same demand):
    MAPPO-greedy     : trained policy, argmax route (deployment mode)
    MAPPO-stoch      : trained policy, sampled route (fleet spreads)
    Dijkstra-static  : free-flow shortest path  -> the naive selfish baseline (T_static)
    Dijkstra-dynamic : shortest path on LIVE edge travel times, replanned each
                       decision -> the congestion-aware selfish equilibrium, i.e.
                       the paper's `dijkstra_dynamic` arm (information parity with
                       MAPPO). This is the baseline the coordination gap says a
                       replanner CANNOT beat and selfless routing can.

Fixed references at this demand (diag_braess_due_so.py 240 1.5, duration+departDelay):
    T_static ~ 409s (all-braess) ,  T_DUE ~ 406s (selfish eq) ,  T_SO ~ 303s (optimum)
A policy that "shows selfless routing" must push avg TT below the selfish
equilibrium (Dijkstra-dynamic / T_DUE) toward T_SO, with per-seed significance.

Usage:
  SUMO_HOME=... PYTHONPATH=. .venv/bin/python eval_braess_inference.py \
      [model.pt] [n_seeds]
"""
import sys
# libsumo shim (same as train_rl) so traci resolves in-process for speed.
try:
    import libsumo as _libsumo
    sys.modules["traci"] = _libsumo
    sys.modules["traci.constants"] = _libsumo.constants
except ImportError:
    pass

import copy
import os
import traci
from sumolib import checkBinary

from core.mappo import MAPPOConfig
from core.rl_training_pipeline import RLTrainingPipeline
from core.STR_SUMO import StrSumo
from controller.MAPPOController import MAPPOPolicy
from controller.DijkstraController import DijkstraPolicy

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

MODEL = sys.argv[1] if len(sys.argv) > 1 else "configurations/model/mappo_policy_braess.best.pt"
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SEEDS = [7000 + i for i in range(N_SEEDS)]   # held out from training/frozen-eval seeds (6000s)
SPAWN = 1.5
T_DUE, T_SO, T_STATIC = 406.0, 303.0, 409.0

pipe = RLTrainingPipeline(
    sumocfg_path="./configurations/braess.sumocfg",
    model_output_path="scratch_braess/_evaldummy.pt",
    episodes=1, spawn_interval=SPAWN,
    mappo_config=MAPPOConfig(),
    target_pattern=4, num_target_vehicles=240, num_random_vehicles=5,
    reroute_epoch_edges=1, eval_every=0, fast_training_profile=True,
    team_reward_alpha=1.0, team_reward_mode="marginal",
)
SUMO = checkBinary("sumo")
NET = os.path.join(pipe.sumocfg_dir, pipe.net_file)


def run_ctrl(controller, vehicles, sumo_seed):
    # Pin SUMO's internal RNG per run. libsumo keeps global RNG state across the many
    # traci.start/close cycles in one process, so an unseeded session inherits whatever the
    # previous arm left -> the same (model, seed) pair drifts run-to-run. An explicit --seed
    # (derived from the held-out episode seed) resets it, making every arm reproducible and
    # the paired comparison exact.
    sim = StrSumo(controller, pipe.connection_info, vehicles)
    traci.start([SUMO, "-c", pipe.runtime_sumocfg_path, "--quit-on-end",
                 "--no-step-log", "--no-warnings", "--seed", str(int(sumo_seed))])
    try:
        *_, stats = sim.run(verbose=False, return_stats=True, print_runtime_summary=False)
    finally:
        try:
            traci.close()
        except Exception:
            pass
    return stats


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def paired_test(name, a, b):
    """Paired one-sided Wilcoxon: is arm `a` (MAPPO) LESS than arm `b` (baseline)?"""
    diffs = [x - y for x, y in zip(a, b)]
    wins = sum(1 for d in diffs if d < 0)
    md = mean(diffs)
    line = (f"  {name:32s}: mean d={md:+7.1f}s  MAPPO faster on {wins}/{len(diffs)} seeds")
    if wilcoxon is not None and len(diffs) >= 6 and any(d != 0 for d in diffs):
        try:
            stat, p = wilcoxon(a, b, alternative="less")
            line += f"  | Wilcoxon p={p:.4g}"
        except Exception as exc:  # pragma: no cover
            line += f"  | Wilcoxon n/a ({exc})"
    return line


arms = {"MAPPO-greedy": [], "MAPPO-stoch": [], "Dijkstra-static": [], "Dijkstra-dynamic": []}
print(f"model={MODEL}  seeds={SEEDS[0]}..{SEEDS[-1]} (n={N_SEEDS})  spawn={SPAWN}  (240 veh, pattern 4)\n")
print(f"{'seed':>6} {'MAPPO-grd':>10} {'MAPPO-sto':>10} {'Dij-static':>11} {'Dij-dynamic':>12} {'nonzero%':>9}")
for seed in SEEDS:
    vehicles = pipe.generate_episode_vehicles(episode_seed=int(seed), spawn_interval_override=SPAWN)

    mg = MAPPOPolicy(copy.deepcopy(vehicles), pipe.connection_info, MODEL,
                     net_xml_file=NET, deterministic=True, reroute_epoch_edges=1)
    s_mg = run_ctrl(mg, copy.deepcopy(vehicles), seed)
    nz = (s_mg.get("controller_runtime_metrics") or {}).get("route_choice_nonzero_rate", 0.0)

    ms = MAPPOPolicy(copy.deepcopy(vehicles), pipe.connection_info, MODEL,
                     net_xml_file=NET, deterministic=False, reroute_epoch_edges=1)
    s_ms = run_ctrl(ms, copy.deepcopy(vehicles), seed)

    dj = DijkstraPolicy(pipe.connection_info, weight_mode="distance")
    s_dj = run_ctrl(dj, copy.deepcopy(vehicles), seed)

    dd = DijkstraPolicy(pipe.connection_info, weight_mode="traveltime")
    s_dd = run_ctrl(dd, copy.deepcopy(vehicles), seed)

    arms["MAPPO-greedy"].append(s_mg["avg_travel_time"])
    arms["MAPPO-stoch"].append(s_ms["avg_travel_time"])
    arms["Dijkstra-static"].append(s_dj["avg_travel_time"])
    arms["Dijkstra-dynamic"].append(s_dd["avg_travel_time"])
    print(f"{seed:>6} {s_mg['avg_travel_time']:>10.1f} {s_ms['avg_travel_time']:>10.1f} "
          f"{s_dj['avg_travel_time']:>11.1f} {s_dd['avg_travel_time']:>12.1f} {float(nz)*100:>8.1f}%")

print("\n=== means over held-out seeds ===")
for k, v in arms.items():
    print(f"  {k:16s}: {mean(v):7.1f}s")
print(f"\nReferences (measured, duration+departDelay): T_static {T_STATIC:.0f} | "
      f"T_DUE {T_DUE:.0f} (selfish eq) | T_SO {T_SO:.0f} (optimum)")

print("\n=== paired significance (per-seed, same demand, same harness metric) ===")
print(paired_test("MAPPO-greedy vs Dijkstra-static", arms["MAPPO-greedy"], arms["Dijkstra-static"]))
print(paired_test("MAPPO-greedy vs Dijkstra-dynamic", arms["MAPPO-greedy"], arms["Dijkstra-dynamic"]))
print("  (Dijkstra-dynamic == the congestion-aware selfish equilibrium / paper's dijkstra_dynamic "
      "arm. Beating it is the coordination-gap claim; beating only static is the weaker info-gap claim.)")

mg_mean = mean(arms["MAPPO-greedy"])
dd_mean = mean(arms["Dijkstra-dynamic"])
# Gap capture vs the WITHIN-HARNESS selfish equilibrium (Dijkstra-dynamic) toward the
# fixed T_SO optimum reference. Uses the harness floor so numerator/denominator share
# the deployment metric as far as possible (T_SO has no in-harness coordinated controller).
frac_fixed = (T_DUE - mg_mean) / (T_DUE - T_SO) if (T_DUE - T_SO) else float("nan")
frac_harness = (dd_mean - mg_mean) / (dd_mean - T_SO) if (dd_mean - T_SO) else float("nan")
print(f"\nMAPPO-greedy {mg_mean:.0f}s vs harness selfish-eq (Dijkstra-dynamic) {dd_mean:.0f}s "
      f"and optimum T_SO {T_SO:.0f}s.")
print(f"  Coordination gap captured (fixed T_DUE ref) : {frac_fixed*100:.0f}%")
print(f"  Coordination gap captured (harness DUE floor): {frac_harness*100:.0f}%")
print("  ~0% => herds like the selfish equilibrium; ~100% => reaches the optimum (fully selfless).")
