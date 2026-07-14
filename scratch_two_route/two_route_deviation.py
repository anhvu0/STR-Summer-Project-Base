"""Unilateral-deviation audit for the two-route-yield benchmark.

From the all-route-0 assignment (the system optimum), switch one AV at a time
to the priority route and record its own gain and the fleet cost, over the 20
held-out SUMO seeds. Writes artifacts/two_route_deviation.csv.
"""

import csv
import os
import statistics as st

from two_route_common import AV_IDS, DEPARTS, TEST_SUMO_SEEDS, run_fixed

WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_dev")
ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "Selfless_routing", "reproduce", "artifacts",
                   "two_route_deviation.csv")


def main():
    base = {v: 0 for v, _ in DEPARTS}
    base_times = {}
    base_fleet = []
    for s in TEST_SUMO_SEEDS:
        r = dict(run_fixed(WORK, f"base_s{s}", base, s))
        for v, tt in r.items():
            base_times.setdefault(v, []).append(tt)
        base_fleet.append(st.mean(r.values()))

    rows = []
    for dev in AV_IDS:
        a = dict(base)
        a[dev] = 1
        own, fleet = [], []
        for s in TEST_SUMO_SEEDS:
            r = dict(run_fixed(WORK, f"dev{dev}_s{s}", a, s))
            own.append(r[dev])
            fleet.append(st.mean(r.values()))
        rows.append(dict(
            deviator=dev, depart=dict(DEPARTS)[dev],
            own_base=st.mean(base_times[dev]), own_dev=st.mean(own),
            own_gain=st.mean(base_times[dev]) - st.mean(own),
            fleet_base=st.mean(base_fleet), fleet_dev=st.mean(fleet),
            fleet_cost=st.mean(fleet) - st.mean(base_fleet)))
        print(rows[-1], flush=True)
    with open(ART, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", ART)


if __name__ == "__main__":
    main()
