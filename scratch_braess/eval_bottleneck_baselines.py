"""R7 dose-response point: bottleneck-map baselines (REVISION_PLAN_2026-07-13).

Same protocol as eval_braess_baselines.py (held-out seeds 7000+, PRIMARY metric
tripinfo duration+departDelay over ALL road users), on the 3-path bottleneck map
(docs/bottleneck_map_design.md) with the marginal-cost MAPPO checkpoint trained
2026-07-13 (mappo_bottleneck_marginal.best.pt, episode 29).

Arms:
    MAPPO-greedy    : bottleneck marginal checkpoint, argmax, cadence 2
    FixedSplit      : capacity-proportional 40/30/30 A/B/C assignment (the
                      design doc's coordination reference; retune via
                      eval_bottleneck_so_grid.py before quoting as T_SO)
    TollDijkstra    : live-travel-time Dijkstra + queue toll on 1-lane edges
    Dijkstra-static / Dijkstra-dynamic : as in the Braess eval, cadence 2

Usage:
  PYTHONHASHSEED=0 SUMO_HOME=... PYTHONPATH=. .venv/bin/python \
      scratch_braess/eval_bottleneck_baselines.py [spawn_interval] [n_seeds]
Writes Selfless_routing/reproduce/artifacts/bottleneck_baselines_spawn<spawn>.csv
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
from controller.RouteController import RouteController
from controller.SplitAndTollControllers import TollDijkstraPolicy

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

MODEL = "configurations/model/mappo_bottleneck_marginal.best.pt"
SPAWN = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SEEDS = [7000 + i for i in range(N_SEEDS)]
N_VEH = 300
CADENCE = 2   # decision on the staging edge (docs/bottleneck_map_design.md)

PATHS = {"A": ["a1", "a2"], "B": ["b1", "b2", "b3"], "C": ["c1", "c2", "c3"]}
SOURCES = ("in1", "in2", "in3")

pipe = RLTrainingPipeline(
    sumocfg_path="./configurations/bottleneck.sumocfg",
    model_output_path="scratch_braess/_evaldummy_bn.pt",
    episodes=1, spawn_interval=SPAWN,
    mappo_config=MAPPOConfig(),
    target_pattern=4, num_target_vehicles=N_VEH, num_random_vehicles=100,
    reroute_epoch_edges=CADENCE, eval_every=0, fast_training_profile=True,
    team_reward_alpha=1.0, team_reward_mode="marginal",
)
SUMO = checkBinary("sumo")
NET = os.path.join(pipe.sumocfg_dir, pipe.net_file)
TRIPINFO_DIR = f"scratch_braess/tripinfo/bottleneck_spawn{SPAWN:g}"
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


class BottleneckSplitPolicy(RouteController):
    """Fixed (fA, fB, fC) assignment of controlled vehicles to the three paths,
    interleaved deterministically by first-seen order (same spirit as the
    Braess FixedSplitPolicy)."""

    def __init__(self, connection_info, n_vehicles, fa=0.40, fb=0.30):
        super().__init__(connection_info)
        n = int(n_vehicles)
        na = int(round(fa * n))
        nb = int(round(fb * n))
        legs = (["A"] * na + ["B"] * nb + ["C"] * (n - na - nb))
        # Interleave so the assignment is spread over the departure sequence
        # rather than blocked, mirroring so_grid_fine.py.
        self._legs = [legs[(i * 7 + 3) % n] for i in range(n)]
        self._order = {}
        self._routes = {}
        self._n = n

    def _assignment(self, vehicle_id):
        try:
            idx = int(vehicle_id) % self._n
        except ValueError:
            idx = self._order.setdefault(vehicle_id, len(self._order)) % self._n
        return self._legs[idx]

    def make_decisions(self, vehicles, connection_info):
        decisions = {}
        for vehicle in vehicles:
            vid = vehicle.vehicle_id
            cur = vehicle.current_edge
            if vid not in self._routes:
                src = cur if cur in SOURCES else "in1"
                self._routes[vid] = [src, "stage"] + PATHS[self._assignment(vid)] + ["out"]
            route = self._routes[vid]
            if cur in route:
                decisions[vid] = list(route[route.index(cur):])
            else:
                decisions[vid] = [cur, "out"] if cur != "out" else [cur]
        return decisions


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
            net_xml_file=NET, deterministic=True, reroute_epoch_edges=CADENCE),
        "FixedSplit": lambda veh: BottleneckSplitPolicy(pipe.connection_info, N_VEH),
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
header = "  ".join(f"{k:>15s}" for k in ARM_ORDER)
print(f"{'seed':>6} {header}", flush=True)
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
            "tripinfo_n": s["tripinfo_n"],
            "tripinfo_unfinished": s["tripinfo_unfinished"],
            "completion_rate": round(s["completion_rate"], 4),
            "timeout_rate": round(s["timeout_rate"], 4),
            "step_limit_reached": int(s["step_limit_reached"]),
        })
    print(f"{seed:>6} " + "  ".join(f"{v:>15.1f}" for v in vals), flush=True)

ART_DIR = "Selfless_routing/reproduce/artifacts"
os.makedirs(ART_DIR, exist_ok=True)
out = f"{ART_DIR}/bottleneck_baselines_spawn{SPAWN:g}.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\n{'arm':>15}  {'mean':>8}  {'median':>8}")
for k in ARM_ORDER:
    v = arms[k]
    print(f"{k:>15}  {mean(v):>8.1f}  {median(v):>8.1f}")

mg = arms["MAPPO-greedy"]
for k in ARM_ORDER:
    if k == "MAPPO-greedy":
        continue
    diffs = [x - y for x, y in zip(mg, arms[k])]
    wins = sum(1 for d in diffs if d < 0)
    p = float("nan")
    if wilcoxon is not None and any(d != 0 for d in diffs):
        _, p = wilcoxon(mg, arms[k], alternative="less")
    print(f"MAPPO vs {k:>15}: mean diff {mean(diffs):+8.1f}s  wins {wins}/{len(diffs)}  p(one-sided)={p:.4g}")
print(f"\nwrote {out}")
