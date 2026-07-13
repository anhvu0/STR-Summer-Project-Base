"""R5: MARL attribution ablations on the Braess map (REVISION_PLAN_2026-07-13).

Braess analogue of the NYC attribution table, through the standard protocol
(held-out seeds 7000-7019, unified tripinfo duration+departDelay metric, all
road users, PYTHONHASHSEED=0). Arms:

    MAPPO            : production checkpoint (marginal_noskip), full stack
    MAPPO-index0     : full stack, always candidate 0 (stack without learned pref)
    MAPPO-untrained  : random actor weights (architecture without training)
    MAPPO-noresv     : production weights, route reservations OFF
    MAPPO-noguard    : production weights, detour guard OFF
    MAPPO-alpha0     : checkpoint trained with team_reward_alpha=0 (selfish reward)
    MAPPO-shared     : checkpoint trained with team_reward_mode=shared
    Dijkstra-dynamic : selfish replanner reference for paired tests

The alpha0/shared checkpoints are produced by scratch_braess/r5_chain.sh; arms
whose checkpoint is missing are skipped with a warning.

Usage:
  PYTHONHASHSEED=0 SUMO_HOME=... PYTHONPATH=. .venv/bin/python \
      eval_braess_ablations.py [n_seeds]
Writes Selfless_routing/reproduce/artifacts/braess_ablations.csv
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
import random
import xml.etree.ElementTree as ET

import numpy as np
import torch
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

PROD = "configurations/model/mappo_policy_braess_marginal_noskip.best.pt"
ALPHA0 = "configurations/model/r5/mappo_braess_alpha0.best.pt"
SHARED = "configurations/model/r5/mappo_braess_shared.best.pt"
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEEDS = [7000 + i for i in range(N_SEEDS)]
SPAWN = 1.5

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
TRIPINFO_DIR = "scratch_braess/tripinfo/ablations"
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
    random.seed(int(sumo_seed))
    np.random.seed(int(sumo_seed))
    torch.manual_seed(int(sumo_seed))
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
    stats["tripinfo_unfinished"] = tri_unf
    return stats


def mappo(veh, model, seed, **kw):
    kwargs = dict(net_xml_file=NET, deterministic=True, reroute_epoch_edges=1)
    kwargs.update(kw)
    return MAPPOPolicy(copy.deepcopy(veh), pipe.connection_info, model, **kwargs)


ARMS = {
    "MAPPO": lambda v, s: mappo(v, PROD, s),
    "MAPPO-index0": lambda v, s: mappo(v, PROD, s, force_index0=True),
    "MAPPO-untrained": lambda v, s: mappo(v, PROD, s, randomize_actor=True, randomize_seed=s),
    "MAPPO-noresv": lambda v, s: mappo(v, PROD, s, route_reservations=False),
    "MAPPO-noguard": lambda v, s: mappo(v, PROD, s, detour_throttle=False),
    "MAPPO-alpha0": lambda v, s: mappo(v, ALPHA0, s),
    "MAPPO-shared": lambda v, s: mappo(v, SHARED, s),
    "Dijkstra-dynamic": lambda v, s: DijkstraPolicy(pipe.connection_info, weight_mode="traveltime"),
}
missing = [k for k, m in (("MAPPO-alpha0", ALPHA0), ("MAPPO-shared", SHARED))
           if not os.path.exists(m)]
for k in missing:
    print(f"WARNING: checkpoint for {k} missing; arm skipped")
    del ARMS[k]
ARM_ORDER = list(ARMS.keys())

arms = {k: [] for k in ARM_ORDER}
rows = []
print(f"seeds={SEEDS[0]}..{SEEDS[-1]} (n={N_SEEDS})  spawn={SPAWN}  prod={PROD}\n")
for seed in SEEDS:
    vehicles = pipe.generate_episode_vehicles(episode_seed=int(seed),
                                              spawn_interval_override=SPAWN)
    vals = []
    for k in ARM_ORDER:
        ctrl = ARMS[k](vehicles, seed)
        s = run_ctrl(ctrl, copy.deepcopy(vehicles), seed,
                     f"{TRIPINFO_DIR}/{k.replace('-', '_')}_{seed}.xml")
        arms[k].append(s["tripinfo_mean_tt"])
        vals.append(s["tripinfo_mean_tt"])
        rows.append({"seed": seed, "arm": k,
                     "tripinfo_mean_tt": round(s["tripinfo_mean_tt"], 3),
                     "harness_avg_tt": round(s["avg_travel_time"], 3),
                     "tripinfo_unfinished": s["tripinfo_unfinished"],
                     "completion_rate": round(s["completion_rate"], 4),
                     "timeout_rate": round(s["timeout_rate"], 4),
                     "step_limit_reached": int(s["step_limit_reached"])})
    print(f"{seed:>6} " + "  ".join(f"{v:>9.1f}" for v in vals))

ART = "Selfless_routing/reproduce/artifacts"
os.makedirs(ART, exist_ok=True)
with open(f"{ART}/braess_ablations.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {ART}/braess_ablations.csv")

print("\n=== per-arm summary (tripinfo, all users) ===")
dd = arms["Dijkstra-dynamic"]
for k in ARM_ORDER:
    v = arms[k]
    diffs = [x - y for x, y in zip(v, dd)]
    wins = sum(1 for d in diffs if d < 0)
    line = (f"  {k:18s}: mean {mean(v):7.1f}  median {median(v):7.1f}"
            f"  vs dyn {mean(diffs):+7.1f}s wins {wins}/{len(diffs)}")
    if k != "Dijkstra-dynamic" and wilcoxon is not None and any(d != 0 for d in diffs):
        try:
            _, p = wilcoxon(v, dd, alternative="less")
            line += f"  p={p:.4g}"
        except Exception:
            pass
    print(line)
