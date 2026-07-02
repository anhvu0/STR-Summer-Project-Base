"""Price-of-anarchy probe for configurations/maps/bottleneck.net.xml.

Runs plain SUMO (no RL stack) on two fixed routings of the same demand:
  selfish  - every vehicle takes its individually-fastest path A (what static
             shortest-path / Dijkstra assigns, and what a selfish equilibrium
             herds toward);
  split    - a capacity-proportional coordinated split (A at ~capacity, the
             rest detouring via B/C), approximating the system optimum.

The ratio selfish/split avg travel time is a lower bound on the map's price of
anarchy. This is the number that must be well above 1 for selfless routing to
have measurable headroom (docs/selfless_routing_analysis.md §4.3/§5.0).

Usage: SUMO_HOME=.venv/lib/python3.14/site-packages/sumo .venv/bin/python \
           diag_bottleneck_poa.py [num_vehicles] [spawn_interval]
"""
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

NET = "configurations/maps/bottleneck.net.xml"
ENTRIES = ["in1", "in2", "in3"]
PATHS = {
    "A": "stage a1 a2 out",
    "B": "stage b1 b2 b3 out",
    "C": "stage c1 c2 c3 out",
}


def write_routes(path, assignments, spawn_interval):
    with open(path, "w") as f:
        f.write("<routes>\n")
        for i, route_key in enumerate(assignments):
            entry = ENTRIES[i % len(ENTRIES)]
            f.write(
                '    <vehicle id="v{i}" depart="{t:.2f}" departLane="best" departSpeed="max">\n'
                '        <route edges="{entry} {path}"/>\n'
                "    </vehicle>\n".format(i=i, t=i * spawn_interval, entry=entry, path=PATHS[route_key])
            )
        f.write("</routes>\n")


def run_scenario(name, assignments, spawn_interval):
    rou = "diag_poa_{}.rou.xml".format(name)
    trips = "diag_poa_{}.trips.xml".format(name)
    write_routes(rou, assignments, spawn_interval)
    sumo = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo")
    subprocess.run(
        [sumo, "-n", NET, "-r", rou, "--tripinfo-output", trips,
         "--no-step-log", "--no-warnings", "--time-to-teleport", "300"],
        check=True, capture_output=True,
    )
    durations = sorted(
        float(t.get("duration")) for t in ET.parse(trips).getroot().iter("tripinfo")
    )
    n = len(durations)
    stats = {
        "n": n,
        "avg": sum(durations) / n,
        "p50": durations[n // 2],
        "p90": durations[int(n * 0.9)],
        "max": durations[-1],
    }
    for f_ in (rou, trips):
        os.remove(f_)
    return stats


def main():
    num_vehicles = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    spawn_interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    selfish = ["A"] * num_vehicles
    # Capacity-proportional split: A ~= its 0.5 veh/s saturation share of the
    # 1/spawn_interval demand rate; remainder alternates over the two detours.
    split_cycle = ["A", "B", "A", "C", "B", "C", "A", "B", "A", "C"]  # 40% A / 30% B / 30% C
    split = [split_cycle[i % len(split_cycle)] for i in range(num_vehicles)]

    s = run_scenario("selfish", selfish, spawn_interval)
    o = run_scenario("split", split, spawn_interval)
    print("demand: {} vehicles @ {:.2f}s spawn interval".format(num_vehicles, spawn_interval))
    print("selfish (all via A) : avg {avg:7.1f}s  p50 {p50:7.1f}  p90 {p90:7.1f}  max {max:7.1f}  (n={n})".format(**s))
    print("coordinated split   : avg {avg:7.1f}s  p50 {p50:7.1f}  p90 {p90:7.1f}  max {max:7.1f}  (n={n})".format(**o))
    print("price-of-anarchy lower bound (avg ratio): {:.2f}".format(s["avg"] / o["avg"]))


if __name__ == "__main__":
    main()
