"""R7: system-optimum fixed-split search on the bottleneck map, THROUGH the
deployment harness (so demand, background vehicles, and metric are identical to
the controller arms, avoiding the Braess refs' demand asymmetry).

Coarse-to-fine over (fA, fB) with fC = 1 - fA - fB: coarse pass at 0.1 share
steps, then 0.05 refinement around the coarse best. Each combo runs on the
given demand seeds and is scored by mean tripinfo duration+departDelay over ALL
road users.

Usage:
  PYTHONHASHSEED=0 SUMO_HOME=... PYTHONPATH=. .venv/bin/python \
      scratch_braess/eval_bottleneck_so_grid.py [seed0,seed1,...]
Writes Selfless_routing/reproduce/artifacts/bottleneck_so_grid.csv
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
from controller.RouteController import RouteController

SEEDS = [int(s) for s in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["7000", "7001", "7002"])]
SPAWN = 1.0
N_VEH = 300
CADENCE = 2
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
TRIPINFO_DIR = "scratch_braess/tripinfo/bottleneck_so_grid"
os.makedirs(TRIPINFO_DIR, exist_ok=True)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def parse_tripinfo(path):
    tri = ET.parse(path).getroot()
    tts = []
    for t in tri.iter("tripinfo"):
        dur = float(t.get("duration", 0.0))
        dd = max(float(t.get("departDelay", 0.0)), 0.0)
        tts.append(dur + dd)
    return mean(tts)


class BottleneckSplitPolicy(RouteController):
    def __init__(self, connection_info, n_vehicles, fa=0.40, fb=0.30):
        super().__init__(connection_info)
        n = int(n_vehicles)
        na = int(round(fa * n))
        nb = int(round(fb * n))
        legs = (["A"] * na + ["B"] * nb + ["C"] * (n - na - nb))
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


def run_split(fa, fb, vehicles, seed, tripinfo_path):
    ctrl = BottleneckSplitPolicy(pipe.connection_info, N_VEH, fa=fa, fb=fb)
    sim = StrSumo(ctrl, pipe.connection_info, copy.deepcopy(vehicles))
    traci.start([SUMO, "-c", pipe.runtime_sumocfg_path, "--quit-on-end",
                 "--no-step-log", "--no-warnings", "--seed", str(int(seed)),
                 "--tripinfo-output", tripinfo_path,
                 "--tripinfo-output.write-unfinished"])
    try:
        sim.run(verbose=False, return_stats=True, print_runtime_summary=False)
    finally:
        try:
            traci.close()
        except Exception:
            pass
    return parse_tripinfo(tripinfo_path)


VEHICLES = {s: pipe.generate_episode_vehicles(episode_seed=s) for s in SEEDS}

rows = []
best = (None, None, float("inf"))


def sweep(points):
    global best
    for fa, fb in points:
        tts = [run_split(fa, fb, VEHICLES[s], s,
                         f"{TRIPINFO_DIR}/fa{fa:.2f}_fb{fb:.2f}_{s}.xml")
               for s in SEEDS]
        m = mean(tts)
        rows.append({"fa": round(fa, 3), "fb": round(fb, 3),
                     "fc": round(1 - fa - fb, 3), "mean_tt": round(m, 3)})
        if m < best[2]:
            best = (fa, fb, m)
        print(f"fa={fa:.2f} fb={fb:.2f} fc={1-fa-fb:.2f}  mean_tt={m:.1f}"
              + ("  <-- best" if m == best[2] else ""), flush=True)


coarse = [(fa / 10, fb / 10) for fa in range(0, 11) for fb in range(0, 11 - fa)]
print(f"coarse pass: {len(coarse)} combos x {len(SEEDS)} seeds")
sweep(coarse)

fa0, fb0, _ = best
fine = []
for dfa in (-0.05, 0.0, 0.05):
    for dfb in (-0.05, 0.0, 0.05):
        fa, fb = fa0 + dfa, fb0 + dfb
        if (dfa, dfb) != (0, 0) and 0 <= fa <= 1 and 0 <= fb <= 1 - fa:
            fine.append((fa, fb))
print(f"fine pass around fa={fa0:.2f} fb={fb0:.2f}: {len(fine)} combos")
sweep(fine)

ART_DIR = "Selfless_routing/reproduce/artifacts"
os.makedirs(ART_DIR, exist_ok=True)
out = f"{ART_DIR}/bottleneck_so_grid.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["fa", "fb", "fc", "mean_tt"])
    w.writeheader()
    w.writerows(rows)
print(f"\nBEST: fa={best[0]:.2f} fb={best[1]:.2f} fc={1-best[0]-best[1]:.2f} mean_tt={best[2]:.1f}")
print(f"wrote {out}")
