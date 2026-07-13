"""R2: tighter T_SO for the Braess map (REVISION_PLAN_2026-07-13 R2).

Round-1 T_SO (diag_braess_due_so.py) searched pb in {0,.1,...,.5}, the SAME pb for
both diamonds, and forced an even up/down split of the rest. This script relaxes all
three restrictions, coarse-to-fine:

  stage 1 (coarse): pb per diamond in {0,.1,...,.6} x up-share u in {.4,.5,.6},
                    independent per diamond -> 21 x 21 combos.
  stage 2 (fine)  : +-0.05 neighborhood around the stage-1 best, pb step 0.05,
                    u step 0.05.

Metric identical to diag_braess_due_so.py: tripinfo duration + departDelay, mean/veh
(all vehicles; this demand is 100% controlled-equivalent fixed routes). The best avg
is a (tighter) upper bound on the true system optimum. Writes the full grid to
Selfless_routing/reproduce/artifacts/braess_so_grid.csv and prints the best combo.

Usage:
  SUMO_HOME=... .venv/bin/python scratch_braess/so_grid_fine.py [num_vehicles] [spawn]
"""
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diag_braess_due_so as diag

N = int(sys.argv[1]) if len(sys.argv) > 1 else 240
SPAWN = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
WORKERS = 8


def diamond_legs(pb, u, n, offset):
    nb = int(round(pb * n))
    rest = n - nb
    nup = int(round(u * rest))
    legs = ["braess"] * nb + ["up"] * nup + ["down"] * (rest - nup)
    return [legs[(i * 7 + offset) % n] for i in range(n)]


def run_combo(combo):
    pb1, u1, pb2, u2 = combo
    l1s = diamond_legs(pb1, u1, N, 0)
    l2s = diamond_legs(pb2, u2, N, 3)
    name = f"so_{int(pb1*100):02d}_{int(u1*100):02d}_{int(pb2*100):02d}_{int(u2*100):02d}"
    st = diag.run_fixed(name, list(zip(l1s, l2s)), SPAWN)
    return combo, st


def frange(lo, hi, step):
    out, x = [], lo
    while x <= hi + 1e-9:
        out.append(round(x, 2))
        x += step
    return out


def search(combos, seen, results):
    todo = [c for c in combos if c not in seen]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for combo, st in ex.map(run_combo, todo):
            seen.add(combo)
            results.append((combo, st))
    return min(results, key=lambda r: r[1]["avg"])


def main():
    diag.SUMO = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo")
    seen, results = set(), []

    coarse = [(pb1, u1, pb2, u2)
              for pb1 in frange(0.0, 0.6, 0.1) for u1 in (0.4, 0.5, 0.6)
              for pb2 in frange(0.0, 0.6, 0.1) for u2 in (0.4, 0.5, 0.6)]
    best = search(coarse, seen, results)
    print(f"coarse best: {best[0]} avg {best[1]['avg']:.1f}s  ({len(results)} runs)")

    (pb1, u1, pb2, u2) = best[0]
    fine = [(a, b, c, d)
            for a in frange(max(pb1 - 0.05, 0.0), min(pb1 + 0.05, 0.6), 0.05)
            for b in frange(max(u1 - 0.05, 0.0), min(u1 + 0.05, 1.0), 0.05)
            for c in frange(max(pb2 - 0.05, 0.0), min(pb2 + 0.05, 0.6), 0.05)
            for d in frange(max(u2 - 0.05, 0.0), min(u2 + 0.05, 1.0), 0.05)]
    best = search(fine, seen, results)
    print(f"fine best  : {best[0]} avg {best[1]['avg']:.1f}s  ({len(results)} runs total)")

    art = "Selfless_routing/reproduce/artifacts/braess_so_grid.csv"
    os.makedirs(os.path.dirname(art), exist_ok=True)
    with open(art, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pb1", "u1", "pb2", "u2", "avg_tt", "p90_tt", "max_tt", "n_done", "n"])
        for (a, b, c, d), st in sorted(results, key=lambda r: r[1]["avg"]):
            w.writerow([a, b, c, d, round(st["avg"], 2), round(st["p90"], 2),
                        round(st["max"], 2), st["n"], N])
    print(f"wrote {art}")

    (a, b, c, d), st = best
    print(f"\nT_SO (tightened upper bound) = {st['avg']:.1f}s at "
          f"pb1={a} up-share1={b} pb2={c} up-share2={d}  (round-1 ref: 303)")
    print(f"with T_DUE=406: coordination gap = {406.0/st['avg']:.3f} (round-1: 1.34)")


if __name__ == "__main__":
    main()
