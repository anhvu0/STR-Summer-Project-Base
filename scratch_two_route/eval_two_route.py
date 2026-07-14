"""Deployment eval on the two-route-yield benchmark of Psarou et al.
(arXiv:2502.13188), through the identical StrSumo harness as the Braess evals.

Arms (per SUMO seed 7000-7019, fixed 22-vehicle demand):
  Full-fleet arms (all 22 controlled; fleet objective = system welfare):
    MAPPO-greedy      : trained policy, argmax route
    Dijkstra-static   : free-flow shortest path (here: route 0 = the optimum route)
    Dijkstra-dynamic  : live-travel-time replanner (defects to the priority route)
  Mixed arms (the paper's mutation split: 10 AVs controlled, 12 humans fixed
  on route 0, the paper's "system optimal" start):
    MAPPO-mixed       : trained policy drives the 10 AVs
    Dijkstra-dyn-mixed: live replanner drives the 10 AVs (selfish-AV analog)

Primary metric: tripinfo duration + departDelay, averaged over ALL road users,
with AV / human group means for the mixed arms. References (same metric,
two_route_references.py): SO (all on route 0) 43.4 s, all on route 1 62.0 s,
all-10-AVs-defect 54.9 s fleet / 41.2 AV / 66.4 human.

Usage:
  PYTHONHASHSEED=0 SUMO_HOME=... PYTHONPATH=.:scratch_two_route .venv/bin/python \
      scratch_two_route/eval_two_route.py [model.pt] [n_seeds]
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

_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import copy
import csv
import os
import xml.etree.ElementTree as ET

import traci
from sumolib import checkBinary

from core.mappo import MAPPOConfig
from core.STR_SUMO import StrSumo
from core import Util
from controller.MAPPOController import MAPPOPolicy
from controller.DijkstraController import DijkstraPolicy

from train_two_route import TwoRoutePipeline, write_training_routes
from two_route_common import AV_IDS, DEPARTS, HUMAN_IDS, TEST_SUMO_SEEDS

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

MODEL = sys.argv[1] if len(sys.argv) > 1 else "configurations/model/mappo_two_route_marginal.best.pt"
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SEEDS = TEST_SUMO_SEEDS[:N_SEEDS]
DEST = "E2"
ALL_IDS = {v for v, _ in DEPARTS}
AV_SET = set(AV_IDS)

pipe = TwoRoutePipeline(
    sumocfg_path="./configurations/two_route.sumocfg",
    model_output_path="scratch_two_route/_evaldummy.pt",
    episodes=1, spawn_interval=1.0,
    mappo_config=MAPPOConfig(),
    target_pattern=4, num_target_vehicles=22, num_random_vehicles=0,
    reroute_epoch_edges=1, eval_every=0, fast_training_profile=True,
    team_reward_alpha=1.0, team_reward_mode="marginal",
)
SUMO = checkBinary("sumo")
NET = os.path.join(pipe.sumocfg_dir, pipe.net_file)
ROUTE_PATH = os.path.join(pipe.sumocfg_dir, pipe.route_file)
TRIPINFO_DIR = "scratch_two_route/tripinfo"
os.makedirs(TRIPINFO_DIR, exist_ok=True)


def make_vehicles(controlled_ids):
    return {str(v): Util.Vehicle(str(v), DEST, dep, dep + 300.0)
            for v, dep in DEPARTS if v in controlled_ids}


def parse_tripinfo(path):
    """duration + departDelay per vehicle id, all road users."""
    out = {}
    for t in ET.parse(path).getroot().iter("tripinfo"):
        dd = max(float(t.get("departDelay", 0.0)), 0.0)
        out[int(t.get("id"))] = float(t.get("duration", 0.0)) + dd
    return out


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def run_arm(controller, controlled_ids, sumo_seed, tripinfo_path):
    import random as _random
    import numpy as _np
    import torch as _torch
    _random.seed(int(sumo_seed))
    _np.random.seed(int(sumo_seed))
    _torch.manual_seed(int(sumo_seed))
    write_training_routes(ROUTE_PATH, controlled_ids)
    sim = StrSumo(controller, pipe.connection_info, make_vehicles(controlled_ids))
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
    per_veh = parse_tripinfo(tripinfo_path)
    return stats, per_veh


def controller_for(arm, controlled_ids):
    vehicles = make_vehicles(controlled_ids)
    if arm.startswith("MAPPO"):
        return MAPPOPolicy(copy.deepcopy(vehicles), pipe.connection_info, MODEL,
                           net_xml_file=NET, deterministic=True, reroute_epoch_edges=1)
    if arm.startswith("Dijkstra-static"):
        return DijkstraPolicy(pipe.connection_info, weight_mode="distance")
    return DijkstraPolicy(pipe.connection_info, weight_mode="traveltime")


ARMS = [
    ("MAPPO-greedy", ALL_IDS),
    ("Dijkstra-static", ALL_IDS),
    ("Dijkstra-dynamic", ALL_IDS),
    ("MAPPO-mixed", AV_SET),
    ("Dijkstra-dyn-mixed", AV_SET),
]

res = {a: {"all": [], "av": [], "human": []} for a, _ in ARMS}
rows = []
print(f"model={MODEL}  seeds={SEEDS[0]}..{SEEDS[-1]}  metric: tripinfo dur+departDelay")
for seed in SEEDS:
    line = f"{seed:>6}"
    for arm, ctl in ARMS:
        controller = controller_for(arm, ctl)
        tag = arm.replace("-", "_").lower()
        stats, per_veh = run_arm(controller, ctl, seed,
                                 f"{TRIPINFO_DIR}/{tag}_{seed}.xml")
        m_all = mean(per_veh.values())
        m_av = mean(tt for v, tt in per_veh.items() if v in AV_SET)
        m_hu = mean(tt for v, tt in per_veh.items() if v not in AV_SET)
        res[arm]["all"].append(m_all)
        res[arm]["av"].append(m_av)
        res[arm]["human"].append(m_hu)
        nz = (stats.get("controller_runtime_metrics") or {}).get("route_choice_nonzero_rate", "")
        rows.append({"seed": seed, "arm": arm,
                     "mean_tt_all": round(m_all, 3), "mean_tt_av": round(m_av, 3),
                     "mean_tt_human": round(m_hu, 3),
                     "completion_rate": round(stats["completion_rate"], 4),
                     "nonzero_rate": nz})
        line += f"  {arm[:12]}={m_all:5.1f}"
    print(line, flush=True)

ART = "Selfless_routing/reproduce/artifacts/two_route_eval.csv"
with open(ART, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nwrote", ART)

print("\n=== summary (mean over seeds; AV/human split uses the mutation ids) ===")
print(f"  {'arm':20s} {'all':>7s} {'AV':>7s} {'human':>7s}")
for arm, _ in ARMS:
    print(f"  {arm:20s} {mean(res[arm]['all']):7.1f} {mean(res[arm]['av']):7.1f} "
          f"{mean(res[arm]['human']):7.1f}")

print("\nReferences: SO (all route 0) 43.4 | all route 1 62.0 | "
      "10 AVs defect: 54.9 all / 41.2 AV / 66.4 human")


def paired(name, a, b):
    diffs = [x - y for x, y in zip(a, b)]
    wins = sum(1 for d in diffs if d < 0)
    line = f"  {name:44s} mean d={mean(diffs):+6.1f}s  wins {wins}/{len(diffs)}"
    if wilcoxon is not None and any(d != 0 for d in diffs):
        try:
            _, p = wilcoxon(a, b, alternative="less")
            line += f"  p={p:.4g}"
        except Exception:
            pass
    return line


print("\n=== paired tests ===")
print(paired("MAPPO-greedy vs Dijkstra-dynamic (all)",
             res["MAPPO-greedy"]["all"], res["Dijkstra-dynamic"]["all"]))
print(paired("MAPPO-mixed vs Dijkstra-dyn-mixed (all)",
             res["MAPPO-mixed"]["all"], res["Dijkstra-dyn-mixed"]["all"]))
print(paired("MAPPO-mixed vs Dijkstra-dyn-mixed (human)",
             res["MAPPO-mixed"]["human"], res["Dijkstra-dyn-mixed"]["human"]))
