#!/usr/bin/env python
"""Build the penetration-sweep figure for the paper from penetration_sweep.csv.

Reads results/penetration_sweep.csv (written by penetration_sweep.py) and writes
../Images/penetration_sweep.pdf (vector, column width).

Panel (a): median per-scenario fleet travel-time delta vs Dijkstra, by
controlled-vehicle penetration, for the deployed policy (detour guard on) and
the guard-off ablation.
Panel (b): executed non-shortest-route rate by penetration for the same arms.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "results", "penetration_sweep.csv")
OUT_PATH = os.path.join(HERE, "..", "Images", "penetration_sweep.pdf")

LEVELS = [10, 25, 50, 75, 100]
# Same series identities as Fig. 4c (deployment.pdf): orange = guard on,
# violet = guard off. Marker shape + line style carry identity in grayscale.
STYLE = {
    "mappo_on":  dict(color="#E8720C", marker="o", linestyle="-",
                      label="MAPPO (guard on)"),
    "mappo_off": dict(color="#5B3E96", marker="D", linestyle="--",
                      label="MAPPO (guard off)"),
}


def main():
    df = pd.read_csv(CSV_PATH)
    piv = df.pivot_table(index=["seed", "level_pct"], columns="arm",
                         values="fleet_mean_tt").reset_index()
    med_delta, wins = {}, {}
    for arm in STYLE:
        med_delta[arm] = [
            (piv.loc[piv.level_pct == l, arm]
             - piv.loc[piv.level_pct == l, "dijkstra"]).median()
            for l in LEVELS
        ]
        wins[arm] = [
            int(((piv.loc[piv.level_pct == l, arm]
                  - piv.loc[piv.level_pct == l, "dijkstra"]) < 0).sum())
            for l in LEVELS
        ]
    detour = df[df.arm != "dijkstra"].pivot_table(
        index="level_pct", columns="arm", values="route_choice_nonzero_rate")

    plt.rcParams.update({
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.4,
        "pdf.fonttype": 42,
    })
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(3.3, 2.3), sharex=True,
        gridspec_kw={"hspace": 0.32},
    )

    ax_a.axhline(0, color="0.55", linewidth=0.7, zorder=1)
    for arm, style in STYLE.items():
        ax_a.plot(LEVELS, med_delta[arm], markersize=4, zorder=3, **style)
    ax_a.set_ylabel("median $\\Delta$ travel time (s)")
    ax_a.set_title("(a) Typical per-scenario gain vs. Dijkstra", loc="left")
    ax_a.annotate("lower = MAPPO faster", xy=(0.02, 0.97), xycoords="axes fraction",
                  ha="left", va="top", fontsize=6.5, color="0.35")

    for arm, style in STYLE.items():
        ax_b.plot(LEVELS, [detour.loc[l, arm] for l in LEVELS],
                  markersize=4, zorder=3, **style)
    ax_b.set_ylabel("non-shortest rate")
    ax_b.legend(frameon=False, loc=(0.30, 0.62), handlelength=2.2)
    ax_b.set_title("(b) Detour rate rises as control share falls", loc="left")
    ax_b.set_xlabel("controlled share of the 450-vehicle corridor fleet (%)")
    ax_b.set_ylim(0.0, 0.60)
    ax_b.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax_b.annotate("guard reverts only pile-on detours",
                  xy=(55, 0.06), fontsize=6.5, color="0.35", ha="center")

    for ax in (ax_a, ax_b):
        ax.set_xticks(LEVELS)
        ax.grid(axis="y", color="0.9", linewidth=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(OUT_PATH, bbox_inches="tight")
    print("wrote", os.path.abspath(OUT_PATH))
    print("guard-on  med delta:", [round(v, 1) for v in med_delta["mappo_on"]],
          "wins:", wins["mappo_on"])
    print("guard-off med delta:", [round(v, 1) for v in med_delta["mappo_off"]],
          "wins:", wins["mappo_off"])


if __name__ == "__main__":
    main()
