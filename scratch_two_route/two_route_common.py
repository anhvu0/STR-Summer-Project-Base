"""Shared definitions for the two-route-yield benchmark scenario.

Scenario ported verbatim from Psarou et al. (arXiv:2502.13188), via the
authors' experiment repository (COeXISTENCE-PROJECT/RouteRL_two_route_net):
network file network.net.xml (copied to configurations/maps/two_route_yield.net.xml),
route set {route 0: E0 E1 E7 E2 (no priority at the merge),
           route 1: E0 E3 E4 E5 E2 (priority)},
and the 22-vehicle demand of agents_data.csv (ids 0,20,21 depart at t=0,
ids 1..19 at t=1..19 s).

AV fleet: 10 of the 22 vehicles, per the paper's constraint that the first
two vehicles in departure order never mutate and no two AVs are consecutive.
We use the deterministic alternating assignment that satisfies it: every
second vehicle in departure order starting from the third.
"""

import os
import subprocess
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET = os.path.join(REPO, "configurations", "maps", "two_route_yield.net.xml")

ROUTE0 = "E0 E1 E7 E2"     # shorter, yields at merge
ROUTE1 = "E0 E3 E4 E5 E2"  # longer, has priority

# (vehicle id, depart time) in the departure order of agents_data.csv
DEPARTS = [(0, 0.0), (20, 0.0), (21, 0.0)] + [(i, float(i)) for i in range(1, 20)]

# every second vehicle in departure order, starting from position 3
AV_IDS = [DEPARTS[p][0] for p in range(2, 22, 2)]
HUMAN_IDS = [v for v, _ in DEPARTS if v not in AV_IDS]

TEST_SUMO_SEEDS = list(range(7000, 7020))


def sumo_binary():
    return os.path.join(os.environ["SUMO_HOME"], "bin", "sumo")


def write_routes(path, assignment):
    """assignment: dict vehicle id -> 0 or 1 (route index)."""
    with open(path, "w") as f:
        f.write("<routes>\n")
        f.write('  <vType id="car"/>\n')
        f.write(f'  <route id="r0" edges="{ROUTE0}"/>\n')
        f.write(f'  <route id="r1" edges="{ROUTE1}"/>\n')
        for vid, dep in DEPARTS:
            r = "r%d" % assignment[vid]
            f.write(f'  <vehicle id="v{vid}" type="car" route="{r}" '
                    f'depart="{dep:.1f}" departLane="best" departSpeed="max"/>\n')
        f.write("</routes>\n")


def run_fixed(workdir, tag, assignment, seed):
    """Run one deterministic SUMO rollout; return list of (vid, tt) with
    tt = tripinfo duration + departDelay (the paper-matched travel time)."""
    os.makedirs(workdir, exist_ok=True)
    rou = os.path.join(workdir, f"{tag}.rou.xml")
    trips = os.path.join(workdir, f"{tag}.tripinfo.xml")
    write_routes(rou, assignment)
    cmd = [sumo_binary(), "-n", NET, "-r", rou,
           "--tripinfo-output", trips, "--tripinfo-output.write-unfinished",
           "--seed", str(seed), "--no-step-log", "--no-warnings",
           "--time-to-teleport", "300", "--end", "3600"]
    subprocess.run(cmd, check=True, capture_output=True)
    out = []
    for t in ET.parse(trips).getroot().iter("tripinfo"):
        vid = int(t.get("id")[1:])
        out.append((vid, float(t.get("duration")) + float(t.get("departDelay"))))
    return out


def group_means(rows):
    av = [tt for v, tt in rows if v in AV_IDS]
    hu = [tt for v, tt in rows if v in HUMAN_IDS]
    allv = [tt for _, tt in rows]
    m = lambda x: sum(x) / len(x)
    return m(allv), m(av), m(hu)
