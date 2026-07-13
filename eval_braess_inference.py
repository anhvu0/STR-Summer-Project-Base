"""Inference eval for the Braess map: does the trained policy capture the
coordination gap (approach T_SO), or does it herd like the selfish equilibrium?

For each held-out seed it generates the SAME 240-vehicle pattern-4 demand at the
TRAINING spawn interval (1.5s, NOT the pipeline's dense 0.5s frozen-eval default,
which gridlocks this map) and runs every controller through StrSumo, exactly like
RLTrainingPipeline._run_frozen_inference_eval.run_eval_controller. All arms run
through the identical harness. PRIMARY metric (R1): SUMO tripinfo
duration + departDelay over ALL road users, unfinished vehicles included, i.e.
the SAME definition diag_braess_due_so.py uses for T_DUE/T_SO, so gap-captured
fractions no longer mix metric definitions. The harness release->arrival time
over controlled vehicles is kept as a secondary column. Per-seed rows are written
to Selfless_routing/reproduce/artifacts/braess_eval_<model>.csv.

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
  PYTHONHASHSEED=0 SUMO_HOME=... PYTHONPATH=. .venv/bin/python eval_braess_inference.py \
      [model.pt] [n_seeds]
"""
import os as _os
import sys

if _os.environ.get("PYTHONHASHSEED") != "0":
    # Route-commit order depends on the hash seed; unpinned runs are not comparable
    # to the paper protocol (near-gridlock seeds flip chaotically). Hard-stop.
    sys.exit("ERROR: run with PYTHONHASHSEED=0 (reproducibility protocol, see docstring)")
# libsumo shim (same as train_rl) so traci resolves in-process for speed.
try:
    import libsumo as _libsumo
    sys.modules["traci"] = _libsumo
    sys.modules["traci.constants"] = _libsumo.constants
except ImportError:
    pass

import copy
import csv
import os
import random
import xml.etree.ElementTree as ET

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


# Per-model tripinfo dir: R3's per-vehicle pairing reads these, so evals of two
# models must not clobber each other's files.
_MODEL_TAG = os.path.splitext(os.path.basename(
    sys.argv[1] if len(sys.argv) > 1 else "mappo_policy_braess.best.pt"))[0]
TRIPINFO_DIR = f"scratch_braess/tripinfo/{_MODEL_TAG}"
os.makedirs(TRIPINFO_DIR, exist_ok=True)


def parse_tripinfo(path):
    """R1 unified welfare metric: tripinfo duration + departDelay over ALL road users
    (controlled + background), unfinished vehicles included with their partial duration.
    This is the same definition diag_braess_due_so.py uses for T_DUE/T_SO, so the
    gap-captured fraction no longer mixes two metrics."""
    tri = ET.parse(path).getroot()
    tts, n_unfinished = [], 0
    for t in tri.iter("tripinfo"):
        dur = float(t.get("duration", 0.0))
        dd = max(float(t.get("departDelay", 0.0)), 0.0)  # -1 marks never-inserted
        if t.get("arrival", "-1") in ("-1", "-1.00"):
            n_unfinished += 1
        tts.append(dur + dd)
    return {
        "tripinfo_mean_tt": mean(tts),
        "tripinfo_n": len(tts),
        "tripinfo_unfinished": n_unfinished,
    }


def run_ctrl(controller, vehicles, sumo_seed, tripinfo_path):
    # Pin SUMO's internal RNG per run. libsumo keeps global RNG state across the many
    # traci.start/close cycles in one process, so an unseeded session inherits whatever the
    # previous arm left -> the same (model, seed) pair drifts run-to-run. An explicit --seed
    # (derived from the held-out episode seed) resets it, making every arm reproducible and
    # the paired comparison exact.
    # Also pin the Python-side RNGs so the STOCHASTIC arm reproduces run-to-run
    # (verified 2026-07-13: without this, artifact CSVs differ only on MAPPO-stoch rows).
    import random as _random

    import numpy as _np
    import torch as _torch
    _random.seed(int(sumo_seed))
    _np.random.seed(int(sumo_seed))
    _torch.manual_seed(int(sumo_seed))
    sim = StrSumo(controller, pipe.connection_info, vehicles)
    traci.start([SUMO, "-c", pipe.runtime_sumocfg_path, "--quit-on-end",
                 "--no-step-log", "--no-warnings", "--seed", str(int(sumo_seed)),
                 "--tripinfo-output", tripinfo_path,
                 "--tripinfo-output.write-unfinished"])
    try:
        *_, stats = sim.run(verbose=False, return_stats=True, print_runtime_summary=False)
    finally:
        try:
            traci.close()
        except Exception:
            pass
    stats.update(parse_tripinfo(tripinfo_path))
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


ARM_ORDER = ["MAPPO-greedy", "MAPPO-stoch", "Dijkstra-static", "Dijkstra-dynamic"]
arms = {k: [] for k in ARM_ORDER}          # R1 primary: tripinfo duration+departDelay, all users
arms_harness = {k: [] for k in ARM_ORDER}  # secondary: harness release->arrival, controlled only
rows = []
print(f"model={MODEL}  seeds={SEEDS[0]}..{SEEDS[-1]} (n={N_SEEDS})  spawn={SPAWN}  (240 veh, pattern 4)")
print("primary metric: tripinfo duration+departDelay over ALL road users (matches DUE/SO refs)\n")
print(f"{'seed':>6} {'MAPPO-grd':>10} {'MAPPO-sto':>10} {'Dij-static':>11} {'Dij-dynamic':>12} {'nonzero%':>9}")
for seed in SEEDS:
    vehicles = pipe.generate_episode_vehicles(episode_seed=int(seed), spawn_interval_override=SPAWN)
    stats_by_arm = {}

    mg = MAPPOPolicy(copy.deepcopy(vehicles), pipe.connection_info, MODEL,
                     net_xml_file=NET, deterministic=True, reroute_epoch_edges=1)
    stats_by_arm["MAPPO-greedy"] = s_mg = run_ctrl(
        mg, copy.deepcopy(vehicles), seed, f"{TRIPINFO_DIR}/mg_{seed}.xml")
    nz = (s_mg.get("controller_runtime_metrics") or {}).get("route_choice_nonzero_rate", 0.0)

    ms = MAPPOPolicy(copy.deepcopy(vehicles), pipe.connection_info, MODEL,
                     net_xml_file=NET, deterministic=False, reroute_epoch_edges=1)
    stats_by_arm["MAPPO-stoch"] = run_ctrl(
        ms, copy.deepcopy(vehicles), seed, f"{TRIPINFO_DIR}/ms_{seed}.xml")

    dj = DijkstraPolicy(pipe.connection_info, weight_mode="distance")
    stats_by_arm["Dijkstra-static"] = run_ctrl(
        dj, copy.deepcopy(vehicles), seed, f"{TRIPINFO_DIR}/dj_{seed}.xml")

    dd = DijkstraPolicy(pipe.connection_info, weight_mode="traveltime")
    stats_by_arm["Dijkstra-dynamic"] = run_ctrl(
        dd, copy.deepcopy(vehicles), seed, f"{TRIPINFO_DIR}/dd_{seed}.xml")

    for k in ARM_ORDER:
        s = stats_by_arm[k]
        arms[k].append(s["tripinfo_mean_tt"])
        arms_harness[k].append(s["avg_travel_time"])
        rows.append({
            "seed": seed, "arm": k,
            "tripinfo_mean_tt": round(s["tripinfo_mean_tt"], 3),
            "harness_avg_tt": round(s["avg_travel_time"], 3),
            "tripinfo_n": s["tripinfo_n"],
            "tripinfo_unfinished": s["tripinfo_unfinished"],
            "completion_rate": round(s["completion_rate"], 4),
            "timeout_rate": round(s["timeout_rate"], 4),
            "step_limit_reached": int(s["step_limit_reached"]),
            "max_step": s["max_step"],
            "p90_travel_time": round(s["p90_travel_time"], 3),
        })
    print(f"{seed:>6} {arms['MAPPO-greedy'][-1]:>10.1f} {arms['MAPPO-stoch'][-1]:>10.1f} "
          f"{arms['Dijkstra-static'][-1]:>11.1f} {arms['Dijkstra-dynamic'][-1]:>12.1f} {float(nz)*100:>8.1f}%")

# Raw per-seed artifact: every reported number regenerates from this file (R1).
ART_DIR = "Selfless_routing/reproduce/artifacts"
os.makedirs(ART_DIR, exist_ok=True)
model_tag = os.path.splitext(os.path.basename(MODEL))[0]
art_path = f"{ART_DIR}/braess_eval_{model_tag}.csv"
with open(art_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote per-seed artifact: {art_path}")

def median(xs):
    s = sorted(xs)
    n = len(s)
    return float("nan") if not n else (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))


def p90(xs):
    s = sorted(xs)
    if not s:
        return float("nan")
    idx = 0.9 * (len(s) - 1)
    lo = int(idx)
    return s[lo] + (idx - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])


def bootstrap_ci(xs, n_boot=10000, seed=0):
    rng = random.Random(seed)
    boots = sorted(mean([rng.choice(xs) for _ in xs]) for _ in range(n_boot))
    return boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def gridlocks(k):
    return sum(1 for r in rows if r["arm"] == k and (r["step_limit_reached"] or r["timeout_rate"] > 0))


print("\n=== per-arm summary, PRIMARY metric (tripinfo duration+departDelay, all users) ===")
print(f"  {'arm':16s} {'mean':>7s} {'95% CI':>17s} {'median':>7s} {'p90':>7s} {'gridlock':>8s}")
for k in ARM_ORDER:
    v = arms[k]
    lo, hi = bootstrap_ci(v)
    print(f"  {k:16s} {mean(v):7.1f} [{lo:7.1f},{hi:7.1f}] {median(v):7.1f} {p90(v):7.1f} {gridlocks(k):5d}/{len(v)}")
print("  secondary (harness release->arrival, controlled only):")
for k in ARM_ORDER:
    v = arms_harness[k]
    print(f"  {k:16s} {mean(v):7.1f}  median {median(v):7.1f}")

print(f"\nReferences (measured, duration+departDelay): T_static {T_STATIC:.0f} | "
      f"T_DUE {T_DUE:.0f} (selfish eq) | T_SO {T_SO:.0f} (optimum)")

print("\n=== paired significance, PRIMARY metric (per-seed, same demand) ===")
print(paired_test("MAPPO-greedy vs Dijkstra-static", arms["MAPPO-greedy"], arms["Dijkstra-static"]))
print(paired_test("MAPPO-greedy vs Dijkstra-dynamic", arms["MAPPO-greedy"], arms["Dijkstra-dynamic"]))
print("  secondary metric (harness):")
print(paired_test("MAPPO-greedy vs Dijkstra-dynamic", arms_harness["MAPPO-greedy"], arms_harness["Dijkstra-dynamic"]))
print("  (Dijkstra-dynamic == the congestion-aware selfish equilibrium / paper's dijkstra_dynamic "
      "arm. Beating it is the coordination-gap claim; beating only static is the weaker info-gap claim.)")

# Gap capture on ONE metric definition: arms, T_DUE and T_SO all use tripinfo
# duration+departDelay. Reported with mean (primary) and median (secondary).
mg_mean, mg_med = mean(arms["MAPPO-greedy"]), median(arms["MAPPO-greedy"])
dd_mean = mean(arms["Dijkstra-dynamic"])
frac_fixed = (T_DUE - mg_mean) / (T_DUE - T_SO) if (T_DUE - T_SO) else float("nan")
frac_fixed_med = (T_DUE - mg_med) / (T_DUE - T_SO) if (T_DUE - T_SO) else float("nan")
frac_harness = (dd_mean - mg_mean) / (dd_mean - T_SO) if (dd_mean - T_SO) else float("nan")
print(f"\nMAPPO-greedy mean {mg_mean:.0f}s / median {mg_med:.0f}s vs harness selfish-eq "
      f"(Dijkstra-dynamic) {dd_mean:.0f}s and optimum T_SO {T_SO:.0f}s.")
print(f"  Coordination gap captured (fixed T_DUE ref, mean)  : {frac_fixed*100:.0f}%")
print(f"  Coordination gap captured (fixed T_DUE ref, median): {frac_fixed_med*100:.0f}%")
print(f"  Coordination gap captured (harness DUE floor, mean): {frac_harness*100:.0f}%")
print("  T_SO is a grid-search upper bound, so these fractions are upper bounds.")
