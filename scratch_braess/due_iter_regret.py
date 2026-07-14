"""Review response: equilibrium residual of the ITERATED DUE reference.

The paper's primary Braess equilibrium reference T_DUE_iter (391.7 s) comes from
SUMO duaIterate (mean of the last 5 of 40 iterations) but was never subjected to
the one-vehicle best-response check that rejected the rerouting-device proxy
(architecture.tex: "which we did not subject to the same check"). This script
closes that gap.

Unlike the device DUE, the iterated assignment IS a fixed-route world: replaying
iteration 039's route file reproduces its tripinfo exactly, so frozen-replay
deviations are exact here (no adaptive-population confound):

  1. Base: replay duaiterate/039/braess_trips_039.rou.xml.gz with iteration
     039's processing options; verify fleet mean matches the recorded 390.9 s.
  2. For each sampled vehicle v and each of the 9 fixed (leg1, leg2) combos:
     identical route file except v gets that combo route. regret(v) =
     t_base(v) - min over combos of t_alt(v), floored at 0.

Reports median/p90/max regret and frac > 5 s; writes per-vehicle rows to
Selfless_routing/reproduce/artifacts/braess_due_iter_regret.csv.

Usage:
  SUMO_HOME=... .venv/bin/python scratch_braess/due_iter_regret.py [iter] [n_sample] [dir] [ttt]
"""
import csv
import gzip
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diag_braess_due_so as diag

ITER = sys.argv[1] if len(sys.argv) > 1 else "039"
N_SAMPLE = int(sys.argv[2]) if len(sys.argv) > 2 else 40
DUA_DIR = sys.argv[3] if len(sys.argv) > 3 else "duaiterate"
TTT = sys.argv[4] if len(sys.argv) > 4 else "300"  # 300 = original run; fine run uses -1
SEED = sys.argv[5] if len(sys.argv) > 5 else "1"   # original run pinned seed 1;
                                                   # fine run omits it (SUMO default)
COMBOS = [(l1, l2) for l1 in ("braess", "up", "down") for l2 in ("braess", "up", "down")]
WORKERS = 8
ROU_GZ = f"scratch_braess/{DUA_DIR}/{ITER}/braess_trips_{ITER}.rou.xml.gz"
# Processing options copied from the iteration sumocfg so the replay is exact.
SIM_OPTS = ["--route-steps", "200", "--time-to-teleport", TTT,
            "--time-to-teleport.highways", "0",
            "--no-step-log", "--no-warnings"] + \
           ([] if SEED == "none" else ["--seed", SEED])


def load_vehicles():
    with gzip.open(ROU_GZ, "rt") as fh:
        root = ET.parse(fh).getroot()
    return [(v.get("id"), v.get("depart"), v.find("route").get("edges"))
            for v in root.iter("vehicle")]


def write_routes(path, vehicles, deviant=None, combo=None):
    with open(path, "w") as f:
        f.write("<routes>\n")
        for vid, depart, edges in vehicles:
            if vid == deviant:
                src = edges.split()[0]
                edges = diag.route_edges(src, *combo)
            f.write(f'  <vehicle id="{vid}" depart="{depart}" departLane="best" '
                    f'departSpeed="max"><route edges="{edges}"/></vehicle>\n')
        f.write("</routes>\n")


def run_sim(name, vehicles, deviant=None, combo=None):
    rou, trips = f"_dir_{name}.rou.xml", f"_dir_{name}.tripinfo.xml"
    write_routes(rou, vehicles, deviant, combo)
    cmd = [diag.SUMO, "-n", diag.NET, "-r", rou, "--tripinfo-output", trips,
           "--end", "40000"] + SIM_OPTS
    subprocess.run(cmd, check=True, capture_output=True)
    tt = {t.get("id"): float(t.get("duration")) + float(t.get("departDelay"))
          for t in ET.parse(trips).getroot().iter("tripinfo")}
    for fn in (rou, trips):
        if os.path.exists(fn):
            os.remove(fn)
    return tt


def main():
    diag.SUMO = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo")
    vehicles = load_vehicles()
    n = len(vehicles)

    base_tt = run_sim("base", vehicles)
    base_mean = sum(base_tt.values()) / len(base_tt)
    print(f"replayed {DUA_DIR} iteration {ITER}: {n} vehicles, "
          f"fleet mean {base_mean:.2f}s")

    sample_idx = [int(round(i * (n - 1) / max(N_SAMPLE - 1, 1))) for i in range(N_SAMPLE)]
    sample = [vehicles[i][0] for i in sample_idx]
    jobs = [(vid, c) for vid in sample for c in COMBOS]

    def dev_run(job):
        vid, c = job
        tt = run_sim(f"{vid}_{c[0][:2]}{c[1][:2]}", vehicles, deviant=vid, combo=c)
        return vid, tt.get(vid, float("nan"))

    best_alt = {vid: float("inf") for vid in sample}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for vid, t in ex.map(dev_run, jobs):
            if t == t and t < best_alt[vid]:
                best_alt[vid] = t

    rows, regrets = [], []
    for vid in sample:
        own = base_tt.get(vid, float("nan"))
        reg = max(own - best_alt[vid], 0.0)
        regrets.append(reg)
        rows.append({"vehicle": vid, "own_tt": round(own, 1),
                     "best_alt_tt": round(best_alt[vid], 1), "regret": round(reg, 1)})

    suffix = "" if (DUA_DIR, ITER) == ("duaiterate", "039") else f"_{DUA_DIR}_{ITER}"
    art = f"Selfless_routing/reproduce/artifacts/braess_due_iter_regret{suffix}.csv"
    with open(art, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    regrets.sort()
    m = len(regrets)
    med = regrets[m // 2] if m % 2 else 0.5 * (regrets[m // 2 - 1] + regrets[m // 2])
    p90 = regrets[min(int(0.9 * (m - 1) + 0.999), m - 1)]
    frac5 = sum(1 for r in regrets if r > 5.0) / m
    print(f"sampled {m} vehicles: median regret {med:.1f}s | p90 {p90:.1f}s | "
          f"max {regrets[-1]:.1f}s | frac>5s {frac5:.0%}")
    print(f"wrote {art}")


if __name__ == "__main__":
    main()
