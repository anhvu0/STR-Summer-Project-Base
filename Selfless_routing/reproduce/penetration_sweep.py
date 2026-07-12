#!/usr/bin/env python
"""Controlled-vehicle penetration sweep for the Selfless Routing paper.

Professor's request (2026-07-08 review): "experiment with different percentages
of controlled vehicles (say from 10% to 100%) and observe the impact on the
approach's effectiveness in addition to studying the effect of the congestion."

Design
------
Demand is held IDENTICAL to the paper at every sweep point: for each seed in
4010-4029 we generate the paper's scenario (450 corridor vehicles routed to one
shared destination + 150 random-trip background vehicles, pattern 2, 0.5 s
spawn interval).  The penetration level p in {10, 25, 50, 75, 100} percent
selects WHICH fraction of the 450 corridor vehicles the controller routes.
The remaining corridor vehicles follow a fixed free-flow shortest path written
into the route file (the generator only writes the start edge, so an
un-controlled corridor vehicle would otherwise stop immediately).  Subsets are
nested per seed (10% subset of the 25% subset of ...), so adjacent levels differ
only by which vehicles gain control.  p=100 with arm=dijkstra / mappo_on
reproduces the paper's Table II configuration exactly.

Arms: dijkstra (distance-only live shortest path), mappo_on (deployed MAPPO,
detour guard active), mappo_off (guard ablation).

Metrics: fleet metrics are computed from SUMO tripinfo over ALL 450 corridor
vehicles so they are comparable across penetration levels; controller-subset
and static-subset breakdowns plus the StrSumo stats dict are also recorded.

Harness notes carried over from earlier debugging:
- a FRESH ConnectionInfo is built per run (a shared one leaks
  edge_vehicle_count state between runs and corrupts results);
- workers install the libsumo shim before importing core/controller modules
  (in-process SUMO, no TraCI ports, safe across worker processes);
- route files and sumocfgs are pre-generated sequentially, then simulations
  run in parallel, so nothing races on the shared configurations/ tree.

Usage (from anywhere; paths are self-anchoring):
    .venv/bin/python Selfless_routing/reproduce/penetration_sweep.py --phase all
    ... --phase gen            # scenario generation + route patching only
    ... --phase run --workers 10
    ... --seeds 4010 --levels 100 --arms dijkstra,mappo_on   # smoke test
Resumable: finished (seed, level, arm) rows in the CSV are skipped on rerun.
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS_DIR = os.path.join(HERE, "results")
ROUTES_DIR = os.path.join(RESULTS_DIR, "routes")
CFG_DIR = os.path.join(RESULTS_DIR, "cfg")
TRIP_DIR = os.path.join(RESULTS_DIR, "tripinfo")
SCENARIO_DIR = os.path.join(RESULTS_DIR, "scenarios")
CSV_PATH = os.path.join(RESULTS_DIR, "penetration_sweep.csv")

NET_FILE = os.path.join(REPO, "configurations", "maps", "simple_nyc.net.xml")
BASE_ROU = os.path.join(REPO, "configurations", "rou", "str_sumo_nyc.rou.xml")
# Paper model: the Phase 2b retrain (recalibrated Layer A + congestion-gated
# reward). Override with --model to reproduce a different checkpoint.
MODEL_PATH = os.path.join(REPO, "configurations", "model", "mappo_policy_nyc_phase2b.best.pt")

DEFAULT_SEEDS = list(range(4010, 4030))
DEFAULT_LEVELS = [10, 25, 50, 75, 100]
DEFAULT_ARMS = ["dijkstra", "mappo_on", "mappo_off"]
N_CONTROLLED = 450
N_BACKGROUND = 150
PATTERN = 2
SPAWN_INTERVAL = 0.5
SUBSET_SEED_BASE = 900000  # rng seed offset for the per-seed control-subset shuffle

CSV_FIELDS = [
    "seed", "level_pct", "arm", "n_subset", "n_static",
    "fleet_mean_tt", "fleet_p50_tt", "fleet_p90_tt", "fleet_p95_p50",
    "fleet_completion", "fleet_deadline_misses", "fleet_unfinished",
    "subset_mean_tt", "subset_completion",
    "static_mean_tt", "static_completion",
    "strsumo_avg_tt", "strsumo_completion", "strsumo_deadlines",
    "strsumo_timeout_rate", "step_limit_reached",
    "route_choice_nonzero_rate", "wall_seconds",
]

_ENV_READY = False


def _env_setup():
    """Idempotent per-process setup: SUMO_HOME, cwd, sys.path, libsumo shim.

    Must run before any core/controller import; they import traci and read
    SUMO_HOME at import time, and MAPPOController resolves relative paths
    against the repo root.
    """
    global _ENV_READY
    if _ENV_READY:
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if "SUMO_HOME" not in os.environ:
        candidate = os.path.join(REPO, ".venv", "lib", "python3.14", "site-packages", "sumo")
        if os.path.isdir(candidate):
            os.environ["SUMO_HOME"] = candidate
    os.chdir(REPO)
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    try:
        import libsumo as _libsumo
        sys.modules["traci"] = _libsumo
        sys.modules["traci.constants"] = _libsumo.constants
    except ImportError:
        pass
    _ENV_READY = True


def scenario_paths(seed):
    return (
        os.path.join(SCENARIO_DIR, "scenario_s%d.json" % seed),
        os.path.join(ROUTES_DIR, "base_s%d.rou.xml" % seed),
    )


def patched_route_path(seed, level):
    return os.path.join(ROUTES_DIR, "rou_s%d_p%03d.rou.xml" % (seed, level))


def cfg_path(seed, level):
    return os.path.join(CFG_DIR, "cfg_s%d_p%03d.sumocfg" % (seed, level))


def generate_scenarios(seeds, levels):
    """Sequential phase: paper-identical demand per seed + patched route files."""
    _env_setup()
    import sumolib
    from core.target_vehicles_generation_protocols import target_vehicles_generator

    net = sumolib.net.readNet(NET_FILE)
    for d in (RESULTS_DIR, ROUTES_DIR, CFG_DIR, TRIP_DIR, SCENARIO_DIR):
        os.makedirs(d, exist_ok=True)

    for seed in seeds:
        meta_path, base_path = scenario_paths(seed)
        if os.path.exists(meta_path) and os.path.exists(base_path):
            meta = json.load(open(meta_path))
        else:
            print("[gen] seed %d: generating %d+%d vehicles (pattern %d, spawn %.1f)"
                  % (seed, N_CONTROLLED, N_BACKGROUND, PATTERN, SPAWN_INTERVAL))
            generator = target_vehicles_generator(NET_FILE)
            vehicle_list = generator.generate_vehicles(
                N_CONTROLLED, N_BACKGROUND, PATTERN, BASE_ROU, NET_FILE,
                spawn_interval=SPAWN_INTERVAL, seed=seed,
            )
            if vehicle_list is None:
                raise RuntimeError("vehicle generation failed for seed %d" % seed)
            shutil.copyfile(BASE_ROU, base_path)
            meta = {
                "seed": seed,
                "vehicles": [
                    {
                        "id": v.vehicle_id,
                        "destination": v.destination,
                        "start_time": float(v.start_time),
                        "deadline": float(v.deadline),
                    }
                    for v in vehicle_list
                ],
                "subsets": {},
            }

        # Nested control subsets: one seeded shuffle, prefixes of it.
        ids = [v["id"] for v in meta["vehicles"]]
        rng = random.Random(SUBSET_SEED_BASE + seed)
        order = list(ids)
        rng.shuffle(order)
        for level in levels:
            k = int(round(len(ids) * level / 100.0))
            meta["subsets"][str(level)] = sorted(order[:k], key=int)
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        _patch_routes_for_seed(seed, levels, meta, net)
    print("[gen] done: %d seeds x %d levels" % (len(seeds), len(levels)))


def _patch_routes_for_seed(seed, levels, meta, net):
    """Write per-level route files: static corridor vehicles get a fixed
    free-flow shortest path; controlled-subset vehicles keep the start-edge
    stub that the controller replaces at runtime."""
    _, base_path = scenario_paths(seed)
    dest_by_id = {v["id"]: v["destination"] for v in meta["vehicles"]}
    sp_cache = {}

    def shortest_edges(start_edge_id, dest_edge_id):
        key = (start_edge_id, dest_edge_id)
        if key not in sp_cache:
            path, _cost = net.getShortestPath(
                net.getEdge(start_edge_id), net.getEdge(dest_edge_id))
            if path is None:
                raise RuntimeError("no path %s -> %s (seed %d)"
                                   % (start_edge_id, dest_edge_id, seed))
            sp_cache[key] = " ".join(e.getID() for e in path)
        return sp_cache[key]

    for level in levels:
        out_path = patched_route_path(seed, level)
        subset = set(meta["subsets"][str(level)])
        tree = ET.parse(base_path)
        for veh in tree.getroot().iter("vehicle"):
            vid = veh.get("id")
            if vid in dest_by_id and vid not in subset:
                route = veh.find("route")
                start_edge_id = route.get("edges").split()[0]
                route.set("edges", shortest_edges(start_edge_id, dest_by_id[vid]))
        tree.write(out_path, encoding="unicode")
        _write_cfg(seed, level)


def _write_cfg(seed, level):
    cfg = ET.Element("configuration")
    inp = ET.SubElement(cfg, "input")
    ET.SubElement(inp, "net-file", value=NET_FILE)
    ET.SubElement(inp, "route-files", value=patched_route_path(seed, level))
    t = ET.SubElement(cfg, "time")
    ET.SubElement(t, "begin", value="0")
    ET.SubElement(t, "end", value="200000")
    ET.ElementTree(cfg).write(cfg_path(seed, level), encoding="unicode")


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    idx = (len(sorted_vals) - 1) * q
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def run_one(task):
    """Worker: one (seed, level, arm, model_path) simulation. Fresh ConnectionInfo per run."""
    seed, level, arm, model_path = task
    _env_setup()
    t0 = time.time()
    import traci
    from sumolib import checkBinary
    from core.STR_SUMO import StrSumo
    from core.Util import ConnectionInfo, Vehicle

    meta_path, _ = scenario_paths(seed)
    meta = json.load(open(meta_path))
    subset = set(meta["subsets"][str(level)])
    by_id = {v["id"]: v for v in meta["vehicles"]}
    vehicles = {
        vid: Vehicle(vid, by_id[vid]["destination"],
                     by_id[vid]["start_time"], by_id[vid]["deadline"])
        for vid in sorted(subset, key=int)
    }

    connection_info = ConnectionInfo(NET_FILE)
    if arm == "dijkstra":
        from controller.DijkstraController import DijkstraPolicy
        scheduler = DijkstraPolicy(connection_info)
    else:
        from controller.MAPPOController import MAPPOPolicy
        scheduler = MAPPOPolicy(
            vehicles, connection_info, model_path, deterministic=True,
            detour_throttle=(arm == "mappo_on"), route_reservations=True,
        )

    trip_path = os.path.join(TRIP_DIR, "trip_s%d_p%03d_%s.xml" % (seed, level, arm))
    simulation = StrSumo(scheduler, connection_info, vehicles)
    traci.start([
        checkBinary("sumo"), "-c", cfg_path(seed, level), "--quit-on-end",
        "--no-step-log", "--no-warnings",
        "--tripinfo-output", trip_path,
        "--tripinfo-output.write-unfinished",
    ])
    try:
        _total, end_number, deadlines_missed, stats = simulation.run(
            verbose=False, return_stats=True, print_runtime_summary=False)
    finally:
        try:
            traci.close()
        except Exception:
            pass

    row = _metrics_from_tripinfo(trip_path, meta, subset)
    row.update({
        "seed": seed, "level_pct": level, "arm": arm,
        "n_subset": len(subset), "n_static": len(by_id) - len(subset),
        "strsumo_avg_tt": round(float(stats.get("avg_travel_time", float("nan"))), 3),
        "strsumo_completion": round(float(stats.get("completion_rate", 0.0)), 4),
        "strsumo_deadlines": int(deadlines_missed),
        "strsumo_timeout_rate": round(float(stats.get("timeout_rate", 0.0)), 4),
        "step_limit_reached": int(bool(stats.get("step_limit_reached", False))),
        "route_choice_nonzero_rate": "",
        "wall_seconds": round(time.time() - t0, 1),
    })
    if arm != "dijkstra":
        metrics = scheduler.get_runtime_metrics()
        row["route_choice_nonzero_rate"] = round(
            float(metrics.get("route_choice_nonzero_rate", float("nan"))), 4)
    return row


def _metrics_from_tripinfo(trip_path, meta, subset):
    dead_by_id = {v["id"]: v["deadline"] for v in meta["vehicles"]}
    fleet_ids = set(dead_by_id)
    durations, misses, unfinished = {}, 0, 0
    for _event, elem in ET.iterparse(trip_path):
        if elem.tag != "tripinfo":
            continue
        vid = elem.get("id")
        if vid not in fleet_ids:
            elem.clear()
            continue
        arrival = float(elem.get("arrival", "-1"))
        if arrival < 0:
            unfinished += 1
        else:
            durations[vid] = float(elem.get("duration"))
            if arrival > dead_by_id[vid]:
                misses += 1
        elem.clear()

    def group(ids):
        vals = sorted(durations[v] for v in ids if v in durations)
        mean = sum(vals) / len(vals) if vals else float("nan")
        return vals, mean

    fleet_vals, fleet_mean = group(fleet_ids)
    _, subset_mean = group(subset)
    static_ids = fleet_ids - subset
    _, static_mean = group(static_ids)
    p50 = _percentile(fleet_vals, 0.50)
    p90 = _percentile(fleet_vals, 0.90)
    p95 = _percentile(fleet_vals, 0.95)
    return {
        "fleet_mean_tt": round(fleet_mean, 3),
        "fleet_p50_tt": round(p50, 3),
        "fleet_p90_tt": round(p90, 3),
        "fleet_p95_p50": round(p95 / p50, 4) if p50 else float("nan"),
        "fleet_completion": round(len(fleet_vals) / float(len(fleet_ids)), 4),
        "fleet_deadline_misses": misses,
        "fleet_unfinished": unfinished + (len(fleet_ids) - len(fleet_vals) - unfinished),
        "subset_mean_tt": round(subset_mean, 3) if subset else "",
        "subset_completion": round(
            sum(1 for v in subset if v in durations) / float(len(subset)), 4) if subset else "",
        "static_mean_tt": round(static_mean, 3) if static_ids else "",
        "static_completion": round(
            sum(1 for v in static_ids if v in durations) / float(len(static_ids)), 4)
            if static_ids else "",
    }


def _done_keys():
    done = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH) as f:
            for row in csv.DictReader(f):
                done.add((int(row["seed"]), int(row["level_pct"]), row["arm"]))
    return done


def run_sweep(seeds, levels, arms, workers, model_path):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    done = _done_keys()
    tasks = [(s, l, a, model_path) for s in seeds for l in levels for a in arms
             if (s, l, a) not in done]
    rng = random.Random(0)
    rng.shuffle(tasks)  # spread heavy seeds across workers
    print("[run] %d tasks (%d already done), %d workers"
          % (len(tasks), len(done), workers))
    if not tasks:
        return

    write_header = not os.path.exists(CSV_PATH)
    t0 = time.time()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
            f.flush()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_one, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                task = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    print("[run] FAILED %s: %r" % (task, exc))
                    continue
                writer.writerow(row)
                f.flush()
                print("[run] %3d/%d s%d p%03d %-9s fleet=%.1fs subset=%s wall=%.0fs (elapsed %.0fs)"
                      % (i, len(tasks), row["seed"], row["level_pct"], row["arm"],
                         row["fleet_mean_tt"], row["subset_mean_tt"],
                         row["wall_seconds"], time.time() - t0))
    print("[run] sweep complete -> %s" % CSV_PATH)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=["gen", "run", "all"], default="all")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--levels", default=",".join(str(l) for l in DEFAULT_LEVELS))
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--model", default=MODEL_PATH,
                        help="MAPPO checkpoint for mappo_* arms (default: Phase 2b)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    levels = [int(l) for l in args.levels.split(",") if l.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        if arm not in DEFAULT_ARMS:
            raise SystemExit("unknown arm %r (choose from %s)" % (arm, DEFAULT_ARMS))

    if args.phase in ("gen", "all"):
        generate_scenarios(seeds, levels)
    if args.phase in ("run", "all"):
        run_sweep(seeds, levels, arms, args.workers, args.model)


if __name__ == "__main__":
    main()
