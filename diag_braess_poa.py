"""True price-of-anarchy probe for configurations/maps/braess.net.xml.

Unlike diag_bottleneck_poa.py (which compared all-on-A vs a hand-split and so
measured mostly an *information* gap T_static/T_split), this probe targets the
distinction that matters for the selfless-routing thesis:

  T_static  : everyone on the free-flow-shortest route (the double-Braess route
              that uses all four single-lane VAR edges). = static Dijkstra / naive
              selfish assignment. Herds every VAR edge.
  T_SO      : best coordinated fixed assignment found here (the up/down split that
              avoids the cross links + a small grid search). Upper bound on the
              system optimum.
  (T_DUE, the congestion-aware selfish equilibrium, is measured separately with
   duaIterate.py -- see diag_braess_due_so.sh; that ratio T_DUE/T_SO is the TRUE
   price of anarchy / coordination gap that Braess -- unlike a parallel-path Pigou
   net -- keeps above 1.)

Fleet travel time uses duration + departDelay (insertion backlog counts), per
Selfless_routing/EXPERIMENT_PLAN.md E2.

Usage:
  SUMO_HOME=.venv/lib/python3.14/site-packages/sumo .venv/bin/python \
      diag_braess_poa.py [num_vehicles] [spawn_interval]
"""
import itertools
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

NET = "configurations/maps/braess.net.xml"
SOURCES = ["in1", "in2"]

# Per-diamond legs (d in {1,2}). Braess leg uses BOTH VAR edges via the cross link.
def leg(kind, d):
    return {
        "braess": f"f_up{d} cross{d} g_dn{d}",
        "up":     f"f_up{d} g_up{d}",
        "down":   f"f_dn{d} g_dn{d}",
    }[kind]

def route_edges(src, leg1, leg2):
    return f"{src} stage {leg(leg1,1)} link1 {leg(leg2,2)} out"


def write_routes(path, assignments, spawn_interval):
    """assignments: list of (leg1, leg2) tuples, one per vehicle."""
    with open(path, "w") as f:
        f.write("<routes>\n")
        for i, (l1, l2) in enumerate(assignments):
            src = SOURCES[i % len(SOURCES)]
            f.write(
                f'    <vehicle id="v{i}" depart="{i*spawn_interval:.2f}" '
                f'departLane="best" departSpeed="max">\n'
                f'        <route edges="{route_edges(src, l1, l2)}"/>\n'
                f'    </vehicle>\n'
            )
        f.write("</routes>\n")


def run_scenario(name, assignments, spawn_interval):
    rou = f"diag_braess_{name}.rou.xml"
    trips = f"diag_braess_{name}.trips.xml"
    stats = f"diag_braess_{name}.stats.xml"
    write_routes(rou, assignments, spawn_interval)
    sumo = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo")
    subprocess.run(
        [sumo, "-n", NET, "-r", rou, "--tripinfo-output", trips,
         "--statistic-output", stats, "--no-step-log", "--no-warnings",
         "--time-to-teleport", "300"],
        check=True, capture_output=True,
    )
    tri = ET.parse(trips).getroot()
    tt = sorted(float(t.get("duration")) + float(t.get("departDelay"))
                for t in tri.iter("tripinfo"))
    n = len(tt)
    st = ET.parse(stats).getroot()
    teleports = st.find("teleports")
    tele = int(teleports.get("total")) if teleports is not None else 0
    veh = st.find("vehicles")
    loaded = int(veh.get("loaded")) if veh is not None else n
    res = {
        "n": n, "loaded": loaded, "teleports": tele,
        "avg": sum(tt) / n if n else float("nan"),
        "p50": tt[n // 2] if n else float("nan"),
        "p90": tt[min(int(n * 0.9), n - 1)] if n else float("nan"),
        "max": tt[-1] if n else float("nan"),
    }
    for fn in (rou, trips, stats):
        if os.path.exists(fn):
            os.remove(fn)
    return res


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    spawn = float(sys.argv[2]) if len(sys.argv) > 2 else 1.1

    scenarios = {
        # T_static: naive selfish -> everyone on the free-flow-shortest route,
        # which is the double-Braess route (both VAR edges in both diamonds).
        "static_all_braess": [("braess", "braess")] * n,
        # reference selfish variants
        "all_up":            [("up", "up")] * n,
        "all_down":          [("down", "down")] * n,
        # SO candidate: 50/50 up/down split per diamond (avoids the cross links);
        # cycles the 4 combinations so every VAR edge carries ~half the fleet.
        "so_split_5050":     [c for c in itertools.islice(
                                  itertools.cycle([("up", "up"), ("up", "down"),
                                                   ("down", "up"), ("down", "down")]), n)],
    }

    print(f"demand: {n} vehicles @ {spawn:.2f}s spawn ({1/spawn:.2f} veh/s), map={NET}\n")
    results = {}
    for name, asg in scenarios.items():
        r = run_scenario(name, asg, spawn)
        results[name] = r
        flag = "  <-- TELEPORTS!" if r["teleports"] else ""
        print(f"{name:20s} avg {r['avg']:7.1f}s  p50 {r['p50']:7.1f}  p90 {r['p90']:7.1f}  "
              f"max {r['max']:7.1f}  done {r['n']}/{r['loaded']}  tele {r['teleports']}{flag}")

    t_static = results["static_all_braess"]["avg"]
    t_so = min(results["so_split_5050"]["avg"], results["all_up"]["avg"],
               results["all_down"]["avg"])
    print(f"\nT_static (naive selfish) = {t_static:.1f}s")
    print(f"T_SO (best fixed assignment found) = {t_so:.1f}s")
    print(f"Information+coordination gap T_static/T_SO = {t_static / t_so:.2f}")
    print("NOTE: the TRUE price of anarchy is T_DUE/T_SO (congestion-aware selfish "
          "vs optimum); run diag_braess_due_so.sh next for T_DUE via duaIterate.")


if __name__ == "__main__":
    main()
