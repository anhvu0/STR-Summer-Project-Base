#!/usr/bin/env python
"""Build the deployment figure (Fig. 4) for the paper.

Reads results/penetration_sweep.csv (written by penetration_sweep.py) at the
100% penetration level, which is exactly the deployment configuration (all 450
corridor vehicles route-controlled), and writes ../Images/deployment.pdf.

Two panels, sized so neither is compressed:
Panel (a): per-scenario fleet travel time, Dijkstra vs greedy MAPPO (detour
guard on), one row per held-out seed sorted by congestion; green connectors
mark MAPPO wins, red losses.
Panel (b): detour-guard ablation. Per-scenario gain (MAPPO minus Dijkstra) vs
scenario congestion for the deployed guard-on policy and the guard-off ablation.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullLocator
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "results", "penetration_sweep.csv")
OUT_PATH = os.path.join(HERE, "..", "Images", "deployment.pdf")

C_DIJK = "#1f77b4"   # Dijkstra baseline (blue)
C_ON = "#E8720C"     # MAPPO, detour guard on (orange); matches penetration fig
C_OFF = "#5B3E96"    # MAPPO, detour guard off (violet); matches penetration fig
C_WIN = "#2f9e44"    # MAPPO faster than Dijkstra
C_LOSS = "#e03131"   # MAPPO slower than Dijkstra


def main():
    df = pd.read_csv(CSV_PATH)
    piv = df[df.level_pct == 100].pivot_table(
        index="seed", columns="arm", values="fleet_mean_tt")
    piv = piv.sort_values("dijkstra", ascending=False)  # most congested first
    seeds = list(piv.index)
    dij = piv["dijkstra"].values
    on = piv["mappo_on"].values
    off = piv["mappo_off"].values
    wins = int((on < dij).sum())

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7, "legend.fontsize": 7.5,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2, "pdf.fonttype": 42,
    })
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(6.5, 2.7),
        gridspec_kw={"wspace": 0.28, "width_ratios": [1.0, 1.05]},
    )

    # ---- Panel (a): per-seed comparison ---------------------------------
    y = range(len(seeds))
    for yi, d, o in zip(y, dij, on):
        ax_a.plot([d, o], [yi, yi], color=(C_WIN if o < d else C_LOSS),
                  linewidth=1.3, zorder=2)
    ax_a.scatter(dij, list(y), s=18, color=C_DIJK, zorder=3, label="Dijkstra")
    ax_a.scatter(on, list(y), s=18, color=C_ON, zorder=3, label="MAPPO")
    ax_a.set_yticks(list(y))
    ax_a.set_yticklabels(seeds)
    ax_a.invert_yaxis()
    ax_a.set_ylabel("seed (sorted by congestion)")
    ax_a.set_xlabel("avg travel time (s)")
    ax_a.set_title("(a) Per-seed comparison", loc="left")
    ax_a.legend(frameon=False, loc="lower right", handletextpad=0.3)
    ax_a.grid(axis="x", color="0.9", linewidth=0.5, zorder=0)

    # ---- Panel (b): detour-guard ablation -------------------------------
    ax_b.axhline(0, color="0.55", linewidth=0.7, zorder=1)
    ax_b.scatter(dij, on - dij, s=26, color=C_ON, zorder=3,
                 label="guard on")
    ax_b.scatter(dij, off - dij, s=30, color=C_OFF, marker="D", zorder=3,
                 label="guard off")
    ax_b.set_xscale("log")
    ax_b.xaxis.set_major_formatter(ScalarFormatter())
    ax_b.xaxis.set_minor_locator(NullLocator())
    ax_b.set_xticks([200, 300, 500, 800, 1200])
    ax_b.set_xlabel("scenario congestion (s, log)")
    ax_b.set_ylabel("MAPPO $-$ Dijkstra (s)")
    ax_b.set_title("(b) Detour-guard ablation", loc="left")
    ax_b.legend(frameon=False, loc="lower left", handletextpad=0.3)
    ax_b.annotate("detours hurt", xy=(0.03, 0.96), xycoords="axes fraction",
                  ha="left", va="top", fontsize=7, color="0.35")
    ax_b.annotate("detours help", xy=(0.97, 0.05), xycoords="axes fraction",
                  ha="right", va="bottom", fontsize=7, color="0.35")
    ax_b.grid(axis="y", color="0.9", linewidth=0.5, zorder=0)

    for ax in (ax_a, ax_b):
        ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(OUT_PATH, bbox_inches="tight")
    print("wrote", os.path.abspath(OUT_PATH))
    print(f"seeds={len(seeds)}  MAPPO wins={wins}/{len(seeds)}")
    print(f"means  dijkstra={dij.mean():.1f}  guard-on={on.mean():.1f}  "
          f"guard-off={off.mean():.1f}")


if __name__ == "__main__":
    main()
