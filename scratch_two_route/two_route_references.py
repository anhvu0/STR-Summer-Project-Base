"""Enumerate route splits on the two-route-yield benchmark.

Part 1: whole-population splits (all 22 vehicles), k vehicles on route 1,
        assigned in departure order round-robin  -> locates SO and checks the
        all-route-0 / all-route-1 equilibria of Psarou et al.
Part 2: humans fixed on route 0 (the paper's "system optimal" start),
        enumerate k of the 10 AVs on route 1 -> the best achievable outcome
        for any AV controller in this scenario (exact SO upper bound).

Every configuration runs over the 20 held-out SUMO seeds (7000-7019).
Writes Selfless_routing/reproduce/artifacts/two_route_references.csv
"""

import csv
import itertools
import os
import statistics as st

from two_route_common import (AV_IDS, DEPARTS, HUMAN_IDS, TEST_SUMO_SEEDS,
                              group_means, run_fixed)

WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "Selfless_routing", "reproduce", "artifacts",
                   "two_route_references.csv")

def pop_split(k1):
    """k1 vehicles on route 1, spread evenly over departure order."""
    order = [v for v, _ in DEPARTS]
    on1 = set()
    if k1 > 0:
        step = len(order) / k1
        on1 = {order[min(21, round(i * step + step / 2 - 0.5))] for i in range(k1)}
        # fix collisions by walking forward
        on1 = set()
        pos = [round((i + 0.5) * len(order) / k1 - 0.5) for i in range(k1)]
        used = set()
        for p in pos:
            while p in used:
                p += 1
            used.add(min(p, 21))
        on1 = {order[p] for p in used}
    return {v: (1 if v in on1 else 0) for v, _ in DEPARTS}


def av_split(k1):
    """Humans on route 0; first k1 AVs (departure order) on route 1."""
    a = {v: 0 for v, _ in DEPARTS}
    for v in AV_IDS[:k1]:
        a[v] = 1
    return a


def main():
    rows = []
    for name, splitter, ks in [("pop", pop_split, range(0, 23)),
                               ("av", av_split, range(0, 11))]:
        for k in ks:
            assignment = splitter(k)
            per_seed = []
            for s in TEST_SUMO_SEEDS:
                r = run_fixed(WORK, f"{name}{k}_s{s}", assignment, s)
                per_seed.append(group_means(r))
            mall = [x[0] for x in per_seed]
            mav = [x[1] for x in per_seed]
            mhu = [x[2] for x in per_seed]
            rows.append(dict(family=name, k_route1=k,
                             mean_all=st.mean(mall), sd_all=st.pstdev(mall),
                             mean_av=st.mean(mav), mean_human=st.mean(mhu)))
            print(f"{name} k1={k:2d}  all={st.mean(mall):6.1f} "
                  f"(sd {st.pstdev(mall):4.1f})  av={st.mean(mav):6.1f} "
                  f"hum={st.mean(mhu):6.1f}", flush=True)
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    with open(ART, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", ART)


if __name__ == "__main__":
    main()
