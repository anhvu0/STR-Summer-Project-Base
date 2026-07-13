"""True price-of-anarchy for the Braess map: T_DUE / T_SO.

This is the number Gate G2 (Selfless_routing/EXPERIMENT_PLAN.md) turns on -- the
*coordination* gap that survives a congestion-aware selfish equilibrium, as
opposed to the *information* gap (T_static / T_DUE) that any replanner captures.

  T_static : whole fleet on the free-flow-shortest route (double-Braess). = naive
             static Dijkstra / selfish assignment.
  T_DUE    : congestion-aware selfish equilibrium, measured with SUMO's rerouting
             device (every vehicle greedily re-plans on live edge travel times at
             a fixed cadence). This is also the paper's `dijkstra_dynamic` arm, so
             information parity with MAPPO is explicit.
  T_SO     : best coordinated fixed assignment found by a grid search over the
             per-diamond {braess, up, down} split (upper bound on the system
             optimum). The Braess optimum avoids the cross links.

All travel times use duration + departDelay, summed over the fleet, mean/veh.

Usage:
  SUMO_HOME=.venv/lib/python3.14/site-packages/sumo .venv/bin/python \
      diag_braess_due_so.py [num_vehicles] [spawn_interval]
"""
import itertools
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

NET = "configurations/maps/braess.net.xml"
SOURCES = ["in1", "in2"]
SUMO = None  # set in main


def leg(kind, d):
    return {"braess": f"f_up{d} cross{d} g_dn{d}",
            "up": f"f_up{d} g_up{d}",
            "down": f"f_dn{d} g_dn{d}"}[kind]


def route_edges(src, l1, l2):
    return f"{src} stage {leg(l1, 1)} link1 {leg(l2, 2)} out"


def _stats(trips_file):
    tri = ET.parse(trips_file).getroot()
    tt = sorted(float(t.get("duration")) + float(t.get("departDelay"))
                for t in tri.iter("tripinfo"))
    n = len(tt)
    return {"n": n, "avg": sum(tt) / n if n else float("nan"),
            "p90": tt[min(int(n * 0.9), n - 1)] if n else float("nan"),
            "max": tt[-1] if n else float("nan")}


def run_fixed(name, assignments, spawn, reroute=False):
    rou = f"_b_{name}.rou.xml"
    trips = f"_b_{name}.tripinfo.xml"
    with open(rou, "w") as f:
        f.write("<routes>\n")
        dev = ' <param key="has.rerouting.device" value="true"/>' if reroute else ""
        for i, (l1, l2) in enumerate(assignments):
            src = SOURCES[i % len(SOURCES)]
            f.write(f'  <vehicle id="v{i}" depart="{i*spawn:.2f}" departLane="best" '
                    f'departSpeed="max"><route edges="{route_edges(src, l1, l2)}"/>{dev}</vehicle>\n')
        f.write("</routes>\n")
    cmd = [SUMO, "-n", NET, "-r", rou, "--tripinfo-output", trips,
           "--no-step-log", "--no-warnings", "--time-to-teleport", "-1", "--end", "40000"]
    if reroute:
        cmd += ["--device.rerouting.probability", "1",
                "--device.rerouting.period", "8",
                "--device.rerouting.pre-period", "1",
                "--device.rerouting.adaptation-interval", "1",
                "--device.rerouting.adaptation-steps", "12"]
    subprocess.run(cmd, check=True, capture_output=True)
    st = _stats(trips)
    for fn in (rou, trips):
        if os.path.exists(fn):
            os.remove(fn)
    return st


def main():
    global SUMO
    SUMO = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    spawn = float(sys.argv[2]) if len(sys.argv) > 2 else 1.8

    # T_static: fleet on the free-flow-shortest route (double-Braess).
    static = run_fixed("static", [("braess", "braess")] * n, spawn)

    # T_DUE: same demand, initial route = braess, rerouting device on -> vehicles
    # re-plan on live travel times (congestion-aware selfish equilibrium proxy).
    due = run_fixed("due", [("braess", "braess")] * n, spawn, reroute=True)

    # T_SO: grid search over the fraction of the fleet on the braess leg (rest
    # split evenly up/down), balanced INDEPENDENTLY per diamond so each VAR edge
    # carries ~half the non-braess share. Best (lowest avg) = SO upper bound.
    def diamond_legs(pb, offset):
        nb = int(round(pb * n))
        rest = n - nb
        legs = ["braess"] * nb + ["up"] * (rest // 2) + ["down"] * (rest - rest // 2)
        # deterministic interleave so the pattern is stationary over time
        return [legs[(i * 7 + offset) % n] for i in range(n)]

    best = None
    grid = []
    for pb in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        l1s = diamond_legs(pb, 0)
        l2s = diamond_legs(pb, 3)
        asg = list(zip(l1s, l2s))
        st = run_fixed(f"so_pb{int(pb*100)}", asg, spawn)
        grid.append((pb, st))
        if best is None or st["avg"] < best[1]["avg"]:
            best = (pb, st)

    print(f"demand: {n} vehicles @ {spawn:.2f}s spawn ({1/spawn:.2f} veh/s)\n")
    print(f"  T_static (all braess, free-flow shortest) : avg {static['avg']:7.1f}s  "
          f"p90 {static['p90']:7.1f}  done {static['n']}/{n}")
    print(f"  T_DUE    (rerouting device, selfish-dyn)   : avg {due['avg']:7.1f}s  "
          f"p90 {due['p90']:7.1f}  done {due['n']}/{n}")
    print("  SO grid (fixed split, frac on braess leg):")
    for pb, st in grid:
        mark = "  <== best (T_SO bound)" if (pb, st) == best else ""
        print(f"      p_braess={pb:.1f}: avg {st['avg']:7.1f}s  p90 {st['p90']:7.1f}"
              f"  done {st['n']}/{n}{mark}")
    t_so = best[1]["avg"]
    print(f"\n  T_static / T_SO = {static['avg']/t_so:5.2f}   (information + coordination gap)")
    print(f"  T_DUE    / T_SO = {due['avg']/t_so:5.2f}   <== TRUE price of anarchy "
          f"(coordination gap; Gate G2 wants >= ~1.2)")
    print(f"  T_static / T_DUE = {static['avg']/due['avg']:5.2f}  (information gap alone, "
          f"captured by any replanner)")


if __name__ == "__main__":
    main()
