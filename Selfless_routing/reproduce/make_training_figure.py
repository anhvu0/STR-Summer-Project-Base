#!/usr/bin/env python
"""Build the training-dynamics figure (paper Fig. 2) from the live episode log.

Reads configurations/rl_episode_metrics.csv (written by train_rl.py; currently the
Phase 2b 150-episode retrain) and writes ../Images/training_dynamics.pdf.

Panels:
  (a) fleet travel time per episode  -- demand-driven variance
  (b) travel time vs teleports       -- outcomes track congestion spikes (r)
  (c) policy entropy + score margin  -- the policy sharpens as entropy anneals
  (d) approx. KL vs target KL        -- weak per-update signal (updates below target)
"""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV_PATH = os.path.join(REPO, "configurations", "rl_episode_metrics.csv")
OUT_PATH = os.path.join(HERE, "..", "Images", "training_dynamics.pdf")
TARGET_KL = 0.015

ORANGE = "#E8720C"
VIOLET = "#5B3E96"
INK = "#222222"


def load():
    rows = list(csv.DictReader(open(CSV_PATH)))
    def col(name):
        out = []
        for r in rows:
            v = r.get(name, "")
            try:
                out.append((int(r["episode"]), float(v)))
            except (ValueError, TypeError):
                pass
        return out
    return col


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return cov / (sx * sy) if sx * sy else float("nan")


def main():
    col = load()
    ep_tt = col("avg_travel_time")
    ep_tel = col("teleports")
    ep_ent = col("entropy")
    ep_margin = col("route_mean_logit_margin")
    ep_kl = col("approx_kl")

    plt.rcParams.update({
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.75))
    ax_a, ax_b, ax_c, ax_d = axes

    # (a) fleet travel time per episode
    ea, ta = zip(*ep_tt)
    ax_a.plot(ea, ta, color=ORANGE, linewidth=1.0)
    imin, imax = ta.index(min(ta)), ta.index(max(ta))
    ax_a.set_title("(a) Fleet travel time", loc="left")
    ax_a.set_xlabel("episode")
    ax_a.set_ylabel("mean travel time (s)")
    ax_a.annotate(f"min {min(ta):.0f} (ep{ea[imin]})", xy=(0.30, 0.90),
                  xycoords="axes fraction", fontsize=6, color="0.35")

    # (b) travel time vs teleports (demand-driven)
    tel = {e: v for e, v in ep_tel}
    xs = [tel[e] for e, _ in ep_tt if e in tel]
    ys = [v for e, v in ep_tt if e in tel]
    r = pearson(xs, ys)
    ax_b.scatter(xs, ys, s=7, color=VIOLET, alpha=0.65, edgecolors="none")
    ax_b.set_title("(b) Outcome vs congestion", loc="left")
    ax_b.set_xlabel("teleports (episode)")
    ax_b.set_ylabel("mean travel time (s)")
    ax_b.annotate(f"$r={r:.2f}$", xy=(0.60, 0.10), xycoords="axes fraction",
                  fontsize=7.5, color=INK)

    # (c) entropy anneal + score margin sharpening
    ec, en = zip(*ep_ent)
    ax_c.plot(ec, en, color=ORANGE, label="entropy")
    ax_c.set_title("(c) Policy sharpens", loc="left")
    ax_c.set_xlabel("episode")
    ax_c.set_ylabel("entropy", color=ORANGE)
    ax_c.tick_params(axis="y", labelcolor=ORANGE)
    ax_m = ax_c.twinx()
    em, mg = zip(*ep_margin)
    ax_m.plot(em, mg, color=VIOLET, linestyle="--", label="score margin")
    ax_m.set_ylabel("score margin", color=VIOLET)
    ax_m.tick_params(axis="y", labelcolor=VIOLET)

    # (d) approx KL vs target
    ed, kl = zip(*ep_kl)
    ax_d.plot(ed, kl, color=VIOLET, linewidth=1.0)
    ax_d.axhline(TARGET_KL, color="0.4", linestyle=":", linewidth=1.0)
    ax_d.set_ylim(0, TARGET_KL * 1.15)
    ax_d.set_title("(d) PPO update size", loc="left")
    ax_d.set_xlabel("episode")
    ax_d.set_ylabel("approx. KL")
    med = sorted(k for _, k in ep_kl)[len(ep_kl) // 2]
    ax_d.annotate(f"target {TARGET_KL:g}", xy=(0.04, 0.90), xycoords="axes fraction",
                  fontsize=6, color="0.35")
    ax_d.annotate(f"median\n{med:.4f}", xy=(0.55, 0.45), xycoords="axes fraction",
                  fontsize=6, color=VIOLET, ha="left")

    for ax in list(axes) + [ax_m]:
        ax.spines[["top"]].set_visible(False)
    for ax in (ax_a, ax_b, ax_d):
        ax.spines[["right"]].set_visible(False)

    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print("wrote", os.path.abspath(OUT_PATH))
    print(f"tt min {min(ta):.1f} (ep{ea[imin]})  max {max(ta):.1f} (ep{ea[imax]})")
    print(f"r(tt, teleports) = {r:.3f}   entropy {en[0]:.2f}->{en[-1]:.2f}   "
          f"margin {mg[0]:.2f}->{mg[-1]:.2f}   median approx_kl {med:.4f}")


if __name__ == "__main__":
    main()
