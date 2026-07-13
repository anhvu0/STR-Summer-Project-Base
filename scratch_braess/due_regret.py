"""R2: equilibrium residual of the rerouting-device DUE proxy (REVISION_PLAN R2).

The paper uses SUMO's rerouting device as the dynamic-user-equilibrium reference
T_DUE. This script quantifies how far that proxy is from an actual equilibrium via
sampled one-vehicle best-response deviations IN the DUE world:

  1. DUE run (all vehicles start on the braess route, rerouting device on);
     record each vehicle's travel time t_DUE(v).
  2. For each sampled vehicle v and each of the 9 fixed (leg1, leg2) combos:
     rerun the SAME simulation but give v that fixed route with its rerouting
     device OFF, everyone else unchanged (adaptive). regret(v) =
     t_DUE(v) - min over combos of t_alt(v), floored at 0.

(A frozen-replay variant was tried first and rejected: fixing everyone's final
route from departure changes merge timing and does not reproduce the DUE run,
fleet mean 479 vs 406 s, so regrets measured there are meaningless.)

Positive regret means v could have unilaterally improved on its adaptive outcome
with a fixed route: the residual by which the proxy falls short of equilibrium.
Reports median/p90/max regret and the fraction of sampled vehicles with
regret > 5 s; writes per-vehicle rows to
Selfless_routing/reproduce/artifacts/braess_due_regret.csv.

Usage:
  SUMO_HOME=... .venv/bin/python scratch_braess/due_regret.py [n_vehicles] [spawn] [n_sample]
"""
import csv
import os
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess

import diag_braess_due_so as diag

N = int(sys.argv[1]) if len(sys.argv) > 1 else 240
SPAWN = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
N_SAMPLE = int(sys.argv[3]) if len(sys.argv) > 3 else 40
COMBOS = [(l1, l2) for l1 in ("braess", "up", "down") for l2 in ("braess", "up", "down")]
WORKERS = 8


def write_routes(path, deviant=None, combo=None):
    """DUE demand: everyone departs on the braess route with the rerouting device.
    If deviant is set, vehicle v<deviant> instead gets the fixed `combo` route and
    NO device (a unilateral fixed-route deviation against the adaptive population)."""
    with open(path, "w") as f:
        f.write("<routes>\n")
        # Equip via per-vehicle param ONLY. A global --device.rerouting.probability 1
        # would force the device onto the deviant too and replan its fixed route away
        # (verified: with it, every deviation run is bit-identical to the DUE run).
        dev_on = ' <param key="has.rerouting.device" value="true"/>'
        dev_off = ' <param key="has.rerouting.device" value="false"/>'
        for i in range(N):
            src = diag.SOURCES[i % len(diag.SOURCES)]
            l1, l2 = ("braess", "braess") if i != deviant else combo
            d = dev_on if i != deviant else dev_off
            f.write(f'  <vehicle id="v{i}" depart="{i*SPAWN:.2f}" departLane="best" '
                    f'departSpeed="max"><route edges="{diag.route_edges(src, l1, l2)}"/>{d}</vehicle>\n')
        f.write("</routes>\n")


def run_sim(name, deviant=None, combo=None):
    rou, trips = f"_dr_{name}.rou.xml", f"_dr_{name}.tripinfo.xml"
    write_routes(rou, deviant, combo)
    cmd = [diag.SUMO, "-n", diag.NET, "-r", rou, "--tripinfo-output", trips,
           "--no-step-log", "--no-warnings", "--time-to-teleport", "-1", "--end", "40000",
           "--device.rerouting.period", "8",
           "--device.rerouting.pre-period", "1",
           "--device.rerouting.adaptation-interval", "1",
           "--device.rerouting.adaptation-steps", "12"]
    subprocess.run(cmd, check=True, capture_output=True)
    tt = {t.get("id"): float(t.get("duration")) + float(t.get("departDelay"))
          for t in ET.parse(trips).getroot().iter("tripinfo")}
    for fn in (rou, trips):
        if os.path.exists(fn):
            os.remove(fn)
    return tt


def main():
    diag.SUMO = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo")

    due_tt = run_sim("due")
    due_mean = sum(due_tt.values()) / len(due_tt)
    print(f"DUE fleet mean {due_mean:.1f}s (reference T_DUE)")

    sample = [int(round(i * (N - 1) / max(N_SAMPLE - 1, 1))) for i in range(N_SAMPLE)]
    jobs = [(v, c) for v in sample for c in COMBOS]

    def dev_run(job):
        v, c = job
        tt = run_sim(f"v{v}_{c[0][:2]}{c[1][:2]}", deviant=v, combo=c)
        return v, c, tt.get(f"v{v}", float("nan"))

    best_alt = {v: float("inf") for v in sample}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for v, c, t in ex.map(dev_run, jobs):
            if t == t and t < best_alt[v]:
                best_alt[v] = t

    rows, regrets = [], []
    for v in sample:
        own = due_tt.get(f"v{v}", float("nan"))
        reg = max(own - best_alt[v], 0.0)
        regrets.append(reg)
        rows.append({"vehicle": v,
                     "own_tt": round(own, 1), "best_alt_tt": round(best_alt[v], 1),
                     "regret": round(reg, 1)})

    art = "Selfless_routing/reproduce/artifacts/braess_due_regret.csv"
    os.makedirs(os.path.dirname(art), exist_ok=True)
    with open(art, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    regrets.sort()
    n = len(regrets)
    med = regrets[n // 2] if n % 2 else 0.5 * (regrets[n // 2 - 1] + regrets[n // 2])
    p90 = regrets[min(int(0.9 * (n - 1) + 0.999), n - 1)]
    frac5 = sum(1 for r in regrets if r > 5.0) / n
    print(f"sampled {n} vehicles: median regret {med:.1f}s | p90 {p90:.1f}s | "
          f"max {regrets[-1]:.1f}s | frac>5s {frac5:.0%}")
    print(f"wrote {art}")


if __name__ == "__main__":
    main()
