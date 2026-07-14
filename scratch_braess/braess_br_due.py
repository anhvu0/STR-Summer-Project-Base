"""Best-response dynamics DUE for the chained-Braess network.

duaIterate's assignments fail the one-vehicle best-response check even with
30 s cost aggregation (median regret 13-55 s, p90 265-378 s across candidate
iterations): Gawron route mixing keeps a share of vehicles on routes far worse
than their best fixed alternative. This script computes a defensible T_DUE by
explicit best-response dynamics over the 9 fixed (leg1, leg2) route combos:

  assignment <- duaiterate_fine iteration 036 (lowest-regret candidate)
  repeat (round):
    for every vehicle v: simulate all 9 combos with everyone else fixed,
    record regret(v) = t_now(v) - min_alt t_alt(v) and v's best combo
    switch the K vehicles with the largest regret (> EPS) to their best combo
  until no vehicle can improve by more than EPS or MAX_ROUNDS reached

Tracks the best assignment seen (lexicographic: frac regret>5 s, then median
regret). Writes per-round convergence to
Selfless_routing/reproduce/artifacts/braess_br_due_rounds.csv and the final
per-vehicle regrets to .../braess_br_due_regret.csv, plus the final route
assignment to scratch_braess/braess_br_due_final.rou.xml.

Usage:
  SUMO_HOME=... .venv/bin/python scratch_braess/braess_br_due.py [max_rounds] [start]

start: "036" (default) seeds from duaiterate_fine iteration 036, "static" seeds
from the all-Braess free-flow-shortest assignment (fully selfish start, tests
whether the settled band depends on the initialization). Artifacts get a
"_static" suffix for the static start.
"""
import csv
import gzip
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diag_braess_due_so as diag

MAX_ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
START = sys.argv[2] if len(sys.argv) > 2 else "036"
SUFFIX = "" if START == "036" else f"_{START}"
EPS = 5.0        # seconds: improvements at or below this are treated as noise
K_SWITCH = 12    # vehicles switched per round (5% of fleet, damping)
WORKERS = 11
START_ROU = "scratch_braess/duaiterate_fine/036/braess_trips_036.rou.xml.gz"
COMBOS = [(l1, l2) for l1 in ("braess", "up", "down") for l2 in ("braess", "up", "down")]
# Match the duaiterate_fine simulation exactly (verified: replay reproduces
# the recorded per-iteration means byte-for-byte with these options).
SIM_OPTS = ["--route-steps", "200", "--time-to-teleport", "-1",
            "--time-to-teleport.highways", "0",
            "--no-step-log", "--no-warnings"]
ART = "Selfless_routing/reproduce/artifacts"


def load_start():
    with gzip.open(START_ROU, "rt") as fh:
        root = ET.parse(fh).getroot()
    vehicles = [(v.get("id"), v.get("depart"), v.find("route").get("edges"))
                for v in root.iter("vehicle")]
    if START == "static":
        vehicles = [(vid, dep, diag.route_edges(edges.split()[0], "braess", "braess"))
                    for vid, dep, edges in vehicles]
    return vehicles


def write_routes(path, vehicles, override=None):
    over = override or {}
    with open(path, "w") as f:
        f.write("<routes>\n")
        for vid, depart, edges in vehicles:
            f.write(f'  <vehicle id="{vid}" depart="{depart}" departLane="best" '
                    f'departSpeed="max"><route edges="{over.get(vid, edges)}"/></vehicle>\n')
        f.write("</routes>\n")


def run_sim(name, vehicles, override=None):
    rou, trips = f"_br_{name}.rou.xml", f"_br_{name}.tripinfo.xml"
    write_routes(rou, vehicles, override)
    cmd = [diag.SUMO, "-n", diag.NET, "-r", rou, "--tripinfo-output", trips,
           "--end", "40000"] + SIM_OPTS
    subprocess.run(cmd, check=True, capture_output=True)
    tt = {t.get("id"): float(t.get("duration")) + float(t.get("departDelay"))
          for t in ET.parse(trips).getroot().iter("tripinfo")}
    for fn in (rou, trips):
        if os.path.exists(fn):
            os.remove(fn)
    return tt


def stats(regrets):
    r = sorted(regrets)
    n = len(r)
    med = r[n // 2] if n % 2 else 0.5 * (r[n // 2 - 1] + r[n // 2])
    p90 = r[min(int(0.9 * (n - 1) + 0.999), n - 1)]
    frac5 = sum(1 for x in r if x > EPS) / n
    return med, p90, r[-1], frac5


def main():
    diag.SUMO = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo")
    vehicles = load_start()
    routes = {vid: edges for vid, _, edges in vehicles}
    n = len(vehicles)
    best_snapshot = None  # (frac5, med, mean, routes copy, regret rows)

    rounds_path = os.path.join(ART, f"braess_br_due_rounds{SUFFIX}.csv")
    rf = open(rounds_path, "w", newline="")
    rw = csv.writer(rf)
    rw.writerow(["round", "fleet_mean", "median_regret", "p90_regret",
                 "max_regret", "frac_gt5", "n_switched"])

    for rnd in range(MAX_ROUNDS):
        base_tt = run_sim("base", vehicles, routes)
        fleet_mean = sum(base_tt.values()) / len(base_tt)

        def dev_run(job):
            vid, c = job
            src = routes[vid].split()[0]
            alt = diag.route_edges(src, *c)
            if alt == routes[vid]:
                return vid, c, base_tt[vid]
            tt = run_sim(f"{vid}_{c[0][:2]}{c[1][:2]}", vehicles,
                         {**routes, vid: alt})
            return vid, c, tt.get(vid, float("nan"))

        jobs = [(vid, c) for vid, _, _ in vehicles for c in COMBOS]
        best = {vid: (float("inf"), None) for vid, _, _ in vehicles}
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for vid, c, t in ex.map(dev_run, jobs):
                if t == t and t < best[vid][0]:
                    best[vid] = (t, c)

        rows = []
        for vid, _, _ in vehicles:
            reg = max(base_tt[vid] - best[vid][0], 0.0)
            rows.append((vid, base_tt[vid], best[vid][0], reg, best[vid][1]))
        med, p90, mx, frac5 = stats([r[3] for r in rows])

        key = (frac5, med)
        if best_snapshot is None or key < best_snapshot[0]:
            best_snapshot = (key, fleet_mean, dict(routes), list(rows))

        improvers = sorted((r for r in rows if r[3] > EPS),
                           key=lambda r: -r[3])[:K_SWITCH]
        rw.writerow([rnd, round(fleet_mean, 2), round(med, 1), round(p90, 1),
                     round(mx, 1), round(frac5, 3), len(improvers)])
        rf.flush()
        print(f"round {rnd}: mean {fleet_mean:.1f}s | med regret {med:.1f} | "
              f"p90 {p90:.1f} | max {mx:.1f} | frac>{EPS:.0f}s {frac5:.0%} | "
              f"switching {len(improvers)}", flush=True)
        if not improvers:
            print("converged: no vehicle improves by more than EPS")
            break
        for vid, _, _, _, combo in improvers:
            src = routes[vid].split()[0]
            routes[vid] = diag.route_edges(src, *combo)
    rf.close()

    (_, bmean, broutes, brows) = best_snapshot
    with open(os.path.join(ART, f"braess_br_due_regret{SUFFIX}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["vehicle", "own_tt", "best_alt_tt", "regret"])
        for vid, own, alt, reg, _ in brows:
            w.writerow([vid, round(own, 1), round(alt, 1), round(reg, 1)])
    write_routes(f"scratch_braess/braess_br_due_final{SUFFIX}.rou.xml", vehicles, broutes)
    med, p90, mx, frac5 = stats([r[3] for r in brows])
    print(f"BEST assignment: fleet mean {bmean:.2f}s | med regret {med:.1f} | "
          f"p90 {p90:.1f} | max {mx:.1f} | frac>{EPS:.0f}s {frac5:.0%}")
    print("wrote braess_br_due_rounds.csv, braess_br_due_regret.csv, "
          "braess_br_due_final.rou.xml")


if __name__ == "__main__":
    main()
