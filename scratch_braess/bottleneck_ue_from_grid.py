"""R7: empirical Wardrop (user-equilibrium) split on the bottleneck map, from
the SO-grid tripinfo artifacts (no new sims).

For each (fA, fB) combo of eval_bottleneck_so_grid.py we classify controlled
vehicles to their path by routeLength (A/B/C differ by ~360/440 m) and compute
per-path mean tripinfo tt (duration + departDelay). The empirical UE is the
combo minimizing the Wardrop residual: max over USED paths of
(overall-best-path tt advantage a switcher could gain). T_DUE = fleet mean tt
at that combo, demand-matched to every controller arm.

Usage: PYTHONHASHSEED not needed (pure analysis).
Writes Selfless_routing/reproduce/artifacts/bottleneck_ue_grid.csv
"""
import csv
import glob
import os
import re
import xml.etree.ElementTree as ET

TRIPINFO_DIR = "scratch_braess/tripinfo/bottleneck_so_grid"
SRC_LEN = {"in1": 353.6, "in2": 250.0, "in3": 353.6}
PATH_EXTRA = {"A": 1600.0, "B": 1960.0, "C": 2040.0}
TOL = 90.0

rows = []
for f in sorted(glob.glob(f"{TRIPINFO_DIR}/fa*_fb*_*.xml")):
    m = re.match(r".*/fa([0-9.]+)_fb([0-9.]+)_(\d+)\.xml", f)
    fa, fb, seed = float(m.group(1)), float(m.group(2)), int(m.group(3))
    per_path = {"A": [], "B": [], "C": []}
    per_path_dur = {"A": [], "B": [], "C": []}
    all_tts = []
    for t in ET.parse(f).getroot().iter("tripinfo"):
        dur = float(t.get("duration", 0.0))
        dd = max(float(t.get("departDelay", "0")), 0.0)
        tt = dur + dd
        all_tts.append(tt)
        src = t.get("departLane").rsplit("_", 1)[0]
        if src not in SRC_LEN:
            continue
        rl = float(t.get("routeLength"))
        path, dist = min(((p, abs(rl - (SRC_LEN[src] + extra)))
                          for p, extra in PATH_EXTRA.items()), key=lambda x: x[1])
        if dist <= TOL:
            per_path[path].append(tt)
            per_path_dur[path].append(dur)
    rows.append({
        "fa": fa, "fb": fb, "fc": round(1 - fa - fb, 3), "seed": seed,
        "mean_tt": sum(all_tts) / len(all_tts),
        **{f"tt_{p}": (sum(v) / len(v) if v else float("nan")) for p, v in per_path.items()},
        **{f"dur_{p}": (sum(v) / len(v) if v else float("nan")) for p, v in per_path_dur.items()},
        **{f"n_{p}": len(v) for p, v in per_path.items()},
    })

# aggregate over seeds per combo
combos = {}
for r in rows:
    combos.setdefault((r["fa"], r["fb"]), []).append(r)


def agg(vals):
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


# Wardrop residual on trip DURATION (excludes departDelay, which is common-mode
# ramp queueing a route switch cannot avoid). Unused paths count at their
# free-flow duration: the (optimistic) time a unilateral switcher would see, so
# the residual is an upper bound and an all-on-one-path split cannot fake
# equilibrium.
MEAN_SRC = sum(SRC_LEN.values()) / len(SRC_LEN)
FF_DUR = {p: (MEAN_SRC + extra) / 13.89 for p, extra in PATH_EXTRA.items()}

print(f"{'fa':>5} {'fb':>5} {'fc':>5} {'mean':>7} {'dur_A':>7} {'dur_B':>7} {'dur_C':>7} {'resid':>7}")
best = None
out_rows = []
for (fa, fb), rs in sorted(combos.items()):
    fc = round(1 - fa - fb, 3)
    mean_tt = agg([r["mean_tt"] for r in rs])
    durs = {p: agg([r[f"dur_{p}"] for r in rs]) for p in "ABC"}
    shares = {"A": fa, "B": fb, "C": fc}
    used = {p: durs[p] for p in "ABC" if shares[p] > 0.01 and durs[p] == durs[p]}
    if not used:
        continue
    # what a switcher could get: best of (observed used-path durations, free-flow
    # of unused paths)
    options = dict(used)
    for p in "ABC":
        if p not in options:
            options[p] = FF_DUR[p]
    best_opt = min(options.values())
    resid = max(v - best_opt for v in used.values())
    out_rows.append({"fa": fa, "fb": fb, "fc": fc, "mean_tt": round(mean_tt, 2),
                     "dur_A": round(durs["A"], 2) if durs["A"] == durs["A"] else "",
                     "dur_B": round(durs["B"], 2) if durs["B"] == durs["B"] else "",
                     "dur_C": round(durs["C"], 2) if durs["C"] == durs["C"] else "",
                     "wardrop_residual": round(resid, 2)})
    flag = ""
    if best is None or resid < best[0]:
        best = (resid, fa, fb, fc, mean_tt)
        flag = " <-- lowest residual"
    def _f(p):
        return f"{durs[p]:>7.1f}" if durs[p] == durs[p] else f"{'--':>7}"
    print(f"{fa:>5.2f} {fb:>5.2f} {fc:>5.2f} {mean_tt:>7.1f} "
          f"{_f('A')} {_f('B')} {_f('C')} {resid:>7.1f}{flag}")

os.makedirs("Selfless_routing/reproduce/artifacts", exist_ok=True)
out = "Selfless_routing/reproduce/artifacts/bottleneck_ue_grid.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)
print(f"\nUE estimate: fa={best[1]:.2f} fb={best[2]:.2f} fc={best[3]:.2f} "
      f"residual={best[0]:.1f}s  T_DUE={best[4]:.1f}s")
print(f"wrote {out}")
