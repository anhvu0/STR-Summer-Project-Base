"""R4 learning-necessity baselines on the Braess map (REVISION_PLAN_2026-07-13).

Identical protocol to eval_braess_inference.py (same harness, same held-out
seeds, same PRIMARY metric: tripinfo duration+departDelay over all road users),
comparing the production MAPPO policy against non-learning controllers:

    MAPPO-greedy    : production checkpoint (marginal_noskip), argmax
    FixedSplit      : R2's best fixed assignment (pb=0, 45/55 up/down), tuned
                      at spawn 1.5 and NOT retuned for other spawns
    RandomSplit     : uniform hash assignment, no tuning
    TollDijkstra    : live-travel-time Dijkstra + queue toll on VAR edges
    Dijkstra-static / Dijkstra-dynamic : as in the main eval

Decision D1 (plan): if FixedSplit >= MAPPO everywhere including demand shifts,
the paper reframes; if MAPPO holds off-nominal where the fixed split breaks,
the learned-coordination claim strengthens.

Usage:
  PYTHONHASHSEED=0 SUMO_HOME=... PYTHONPATH=. .venv/bin/python \
      eval_braess_baselines.py [spawn_interval] [n_seeds]
Writes Selfless_routing/reproduce/artifacts/braess_baselines_spawn<spawn>.csv
"""
import os as _os
import sys

if _os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("ERROR: run with PYTHONHASHSEED=0 (reproducibility protocol)")

try:
    import libsumo as _libsumo
    sys.modules["traci"] = _libsumo
    sys.modules["traci.constants"] = _libsumo.constants
except ImportError:
    pass

import copy
import csv
import os
import xml.etree.ElementTree as ET

import traci
from sumolib import checkBinary

from core.mappo import MAPPOConfig
from core.rl_training_pipeline import RLTrainingPipeline
from core.STR_SUMO import StrSumo
from controller.MAPPOController import MAPPOPolicy
from controller.DijkstraController import DijkstraPolicy
from controller.SplitAndTollControllers import (
    FixedSplitPolicy, RandomSplitPolicy, TollDijkstraPolicy)

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

MODEL = "configurations/model/mappo_policy_braess_marginal_noskip.best.pt"
SPAWN = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SEEDS = [7000 + i for i in range(N_SEEDS)]
N_VEH = 240

pipe = RLTrainingPipeline(
    sumocfg_path="./configurations/braess.sumocfg",
    model_output_path="scratch_braess/_evaldummy.pt",
    episodes=1, spawn_interval=SPAWN,
    mappo_config=MAPPOConfig(),
    target_pattern=4, num_target_vehicles=N_VEH, num_random_vehicles=5,
    reroute_epoch_edges=1, eval_every=0, fast_training_profile=True,
    team_reward_alpha=1.0, team_reward_mode="marginal",
)
SUMO = checkBinary("sumo")
NET = os.path.join(pipe.sumocfg_dir, pipe.net_file)
TRIPINFO_DIR = f"scratch_braess/tripinfo/baselines_spawn{SPAWN:g}"
os.makedirs(TRIPINFO_DIR, exist_ok=True)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    s = sorted(xs)
    n = len(s)
    return float("nan") if not n else (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))


def parse_tripinfo(path):
    tri = ET.parse(path).getroot()
    tts, unfinished = [], 0
    for t in tri.iter("tripinfo"):
        dur = float(t.get("duration", 0.0))
        dd = max(float(t.get("departDelay", 0.0)), 0.0)
        if t.get("arrival", "-1") in ("-1", "-1.00"):
            unfinished += 1
        tts.append(dur + dd)
    return mean(tts), len(tts), unfinished


def run_ctrl(controller, vehicles, sumo_seed, tripinfo_path):
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
    tri_mean, tri_n, tri_unf = parse_tripinfo(tripinfo_path)
    stats["tripinfo_mean_tt"] = tri_mean
    stats["tripinfo_n"] = tri_n
    stats["tripinfo_unfinished"] = tri_unf
    return stats


def build_arms():
    return {
        "MAPPO-greedy": lambda veh: MAPPOPolicy(
            copy.deepcopy(veh), pipe.connection_info, MODEL,
            net_xml_file=NET, deterministic=True, reroute_epoch_edges=1),
        "FixedSplit": lambda veh: FixedSplitPolicy(pipe.connection_info, N_VEH),
        "RandomSplit": lambda veh: RandomSplitPolicy(pipe.connection_info),
        "TollDijkstra": lambda veh: TollDijkstraPolicy(pipe.connection_info),
        "Dijkstra-static": lambda veh: DijkstraPolicy(pipe.connection_info, weight_mode="distance"),
        "Dijkstra-dynamic": lambda veh: DijkstraPolicy(pipe.connection_info, weight_mode="traveltime"),
    }


ARMS = build_arms()
ARM_ORDER = list(ARMS.keys())
arms = {k: [] for k in ARM_ORDER}
rows = []
print(f"spawn={SPAWN}  seeds={SEEDS[0]}..{SEEDS[-1]} (n={N_SEEDS})  model={MODEL}")
print("primary metric: tripinfo duration+departDelay over ALL road users\n")
header = "  ".join(f"{k:>13s}" for k in ARM_ORDER)
print(f"{'seed':>6} {header}")
for seed in SEEDS:
    vehicles = pipe.generate_episode_vehicles(episode_seed=int(seed),
                                              spawn_interval_override=SPAWN)
    vals = []
    for k in ARM_ORDER:
        ctrl = ARMS[k](vehicles)
        s = run_ctrl(ctrl, copy.deepcopy(vehicles), seed,
                     f"{TRIPINFO_DIR}/{k.replace('-', '_')}_{seed}.xml")
        arms[k].append(s["tripinfo_mean_tt"])
        vals.append(s["tripinfo_mean_tt"])
        rows.append({
            "seed": seed, "arm": k, "spawn": SPAWN,
            "tripinfo_mean_tt": round(s["tripinfo_mean_tt"], 3),
            "harness_avg_tt": round(s["avg_travel_time"], 3),
            "tripinfo_unfinished": s["tripinfo_unfinished"],
            "completion_rate": round(s["completion_rate"], 4),
            "timeout_rate": round(s["timeout_rate"], 4),
            "step_limit_reached": int(s["step_limit_reached"]),
        })
    print(f"{seed:>6} " + "  ".join(f"{v:>13.1f}" for v in vals))

ART = "Selfless_routing/reproduce/artifacts"
os.makedirs(ART, exist_ok=True)
art_path = f"{ART}/braess_baselines_spawn{SPAWN:g}.csv"
with open(art_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {art_path}")

print("\n=== per-arm summary (tripinfo, all users) ===")
for k in ARM_ORDER:
    v = arms[k]
    print(f"  {k:16s}: mean {mean(v):7.1f}  median {median(v):7.1f}")

print("\n=== paired one-sided Wilcoxon (MAPPO faster than arm?) ===")
mg = arms["MAPPO-greedy"]
for k in ARM_ORDER[1:]:
    diffs = [x - y for x, y in zip(mg, arms[k])]
    wins = sum(1 for d in diffs if d < 0)
    line = f"  MAPPO < {k:16s}: mean d={mean(diffs):+7.1f}s  wins {wins}/{len(diffs)}"
    if wilcoxon is not None and any(d != 0 for d in diffs):
        try:
            _, p = wilcoxon(mg, arms[k], alternative="less")
            line += f"  p={p:.4g}"
        except Exception:
            pass
    print(line)
