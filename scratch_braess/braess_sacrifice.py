"""R3: per-vehicle sacrifice accounting on Braess (REVISION_PLAN_2026-07-13).

Pairs each vehicle's realized travel time (tripinfo duration + departDelay) under
MAPPO-greedy against the SAME vehicle in the SAME seed under Dijkstra-dynamic
(the selfish-replanner reference). Reads the per-arm tripinfo files written by
eval_braess_inference.py (mg_<seed>.xml / dd_<seed>.xml).

Reported per the plan, pooled over seeds and per-seed:
  harmed fraction (delta > 1 s and > 30 s), mean sacrifice per harmed vehicle,
  mean gain per beneficiary, net fleet gain, max / p95 sacrifice,
  Gini of travel times under each arm, altruism efficiency
  (total beneficiary seconds gained per total harmed second sacrificed).

Definition used for the paper: selfless = individually costly AND socially
beneficial. This script measures the "individually costly" half against the
equilibrium counterfactual.

Usage:
  .venv/bin/python scratch_braess/braess_sacrifice.py <tripinfo_dir> [out_csv_tag]
"""
import csv
import os
import sys
import xml.etree.ElementTree as ET

TDIR = sys.argv[1] if len(sys.argv) > 1 else "scratch_braess/tripinfo"
TAG = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(TDIR.rstrip("/"))
SEEDS = [7000 + i for i in range(20)]


def per_vehicle(path):
    return {t.get("id"): float(t.get("duration")) + max(float(t.get("departDelay")), 0.0)
            for t in ET.parse(path).getroot().iter("tripinfo")}


def gini(xs):
    s = sorted(xs)
    n = len(s)
    tot = sum(s)
    if n == 0 or tot == 0:
        return float("nan")
    cum = 0.0
    for i, x in enumerate(s, 1):
        cum += i * x
    return (2.0 * cum) / (n * tot) - (n + 1.0) / n


def main():
    pooled = []
    rows = []
    gini_mg_all, gini_dd_all = [], []
    for seed in SEEDS:
        mg = per_vehicle(os.path.join(TDIR, f"mg_{seed}.xml"))
        dd = per_vehicle(os.path.join(TDIR, f"dd_{seed}.xml"))
        ids = sorted(set(mg) & set(dd), key=str)
        deltas = [(v, mg[v] - dd[v]) for v in ids]
        pooled.extend(d for _, d in deltas)
        harmed = [d for _, d in deltas if d > 1.0]
        helped = [-d for _, d in deltas if d < -1.0]
        gini_mg_all.append(gini([mg[v] for v in ids]))
        gini_dd_all.append(gini([dd[v] for v in ids]))
        rows.append({
            "seed": seed, "n_paired": len(ids),
            "harmed_frac": round(len(harmed) / len(ids), 4),
            "harmed_frac_30s": round(sum(1 for d in harmed if d > 30.0) / len(ids), 4),
            "mean_sacrifice_s": round(sum(harmed) / len(harmed), 2) if harmed else 0.0,
            "max_sacrifice_s": round(max(harmed), 1) if harmed else 0.0,
            "beneficiary_frac": round(len(helped) / len(ids), 4),
            "mean_gain_s": round(sum(helped) / len(helped), 2) if helped else 0.0,
            "net_fleet_gain_s": round(-sum(d for _, d in deltas) / len(ids), 2),
            "altruism_efficiency": round(sum(helped) / sum(harmed), 2) if harmed else float("inf"),
            "gini_mappo": round(gini_mg_all[-1], 4),
            "gini_dijkstra_dyn": round(gini_dd_all[-1], 4),
        })

    art = f"Selfless_routing/reproduce/artifacts/braess_sacrifice_{TAG}.csv"
    os.makedirs(os.path.dirname(art), exist_ok=True)
    with open(art, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    pooled.sort()
    n = len(pooled)
    harmed = [d for d in pooled if d > 1.0]
    helped = [-d for d in pooled if d < -1.0]
    p95_sac = ([d for d in pooled if d > 0] or [0.0])
    p95_sac = pooled[min(int(0.95 * (n - 1) + 0.5), n - 1)]
    print(f"pooled over {len(SEEDS)} seeds, {n} paired vehicle-trips  [{TAG}]")
    print(f"  harmed (>1s slower than own Dijkstra-dynamic time) : {len(harmed)/n:6.1%}"
          f"   (>30s: {sum(1 for d in harmed if d>30)/n:.1%})")
    print(f"  mean sacrifice per harmed vehicle                  : {sum(harmed)/max(len(harmed),1):6.1f}s")
    print(f"  p95 of the paired delta (positive = harmed)        : {p95_sac:6.1f}s")
    print(f"  max sacrifice                                      : {max(harmed) if harmed else 0.0:6.1f}s")
    print(f"  beneficiaries (>1s faster)                         : {len(helped)/n:6.1%}")
    print(f"  mean gain per beneficiary                          : {sum(helped)/max(len(helped),1):6.1f}s")
    print(f"  net fleet gain per vehicle                         : {-sum(pooled)/n:6.1f}s")
    print(f"  altruism efficiency (gain-s per sacrificed-s)      : {sum(helped)/sum(harmed) if harmed else float('inf'):6.2f}")
    print(f"  Gini (MAPPO)    mean over seeds                    : {sum(gini_mg_all)/len(gini_mg_all):.4f}")
    print(f"  Gini (Dij-dyn)  mean over seeds                    : {sum(gini_dd_all)/len(gini_dd_all):.4f}")
    print(f"wrote {art}")


if __name__ == "__main__":
    main()
