"""Diagnostic: compare ep39(best) / ep59(frozen_eval_current) / ep73(final) checkpoints
per-seed on the held-out eval seeds, greedy and/or stochastic, vs a shared Dijkstra baseline.

Usage:
  python diag_eval.py --seeds 6000,6001 --policies greedy --checkpoints best,cur,final
"""
import argparse, copy, os, sys, time, json

# eclipse-sumo wheel layout
if not os.environ.get("SUMO_HOME"):
    import sumolib
    os.environ["SUMO_HOME"] = os.path.dirname(os.path.dirname(sumolib.__file__)) + "/sumo" \
        if os.path.isdir(os.path.dirname(os.path.dirname(sumolib.__file__)) + "/sumo") \
        else os.path.join(os.path.dirname(sys.executable), "..", "lib")
# robust: find site-packages/sumo
import sumolib
_cand = os.path.join(os.path.dirname(os.path.dirname(sumolib.__file__)), "sumo")
if os.path.isdir(_cand):
    os.environ["SUMO_HOME"] = _cand
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

from sumolib import checkBinary
import traci
from core.STR_SUMO import StrSumo, build_runtime_sumocfg
from core.Util import ConnectionInfo
from controller.DijkstraController import DijkstraPolicy
from controller.MAPPOController import MAPPOPolicy
from core.target_vehicles_generation_protocols import target_vehicles_generator

CKPTS = {
    "best":  "./configurations/model/mappo_policy_nyc.best.pt",            # ep39
    "cur":   "./configurations/model/mappo_policy_nyc.frozen_eval_current.pt",  # ep59
    "final": "./configurations/model/mappo_policy_nyc.pt",                 # ep73
}

def gen_vehicles(route_file, conn, n_ctrl, n_unc, pattern, spawn, seed):
    g = target_vehicles_generator(conn.net_filename)
    vlist = g.generate_vehicles(n_ctrl, n_unc, pattern, route_file, conn.net_filename,
                                spawn_interval=spawn, seed=seed)
    return {str(v.vehicle_id): v for v in vlist}

def run(scheduler, vehicles, conn, runtime_cfg, port, binary):
    sim = StrSumo(scheduler, conn, vehicles)
    traci.start([binary, "-c", runtime_cfg, "--quit-on-end", "--no-step-log", "--no-warnings"], port=port)
    try:
        _, _, _, stats = sim.run(verbose=False, return_stats=True, print_runtime_summary=False)
    finally:
        if traci.isLoaded():
            traci.close()
    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="6000,6001,6002,6003,6004,6005,6006,6007,6008,6009,6010")
    ap.add_argument("--policies", default="greedy")  # greedy / stochastic / greedy,stochastic
    ap.add_argument("--checkpoints", default="best,cur,final")
    ap.add_argument("--pattern", type=int, default=2)
    ap.add_argument("--ctrl", type=int, default=350)
    ap.add_argument("--unc", type=int, default=150)
    ap.add_argument("--spawn", type=float, default=0.5)
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--out", default="diag_eval_results.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    policies = [p for p in args.policies.split(",") if p.strip()]
    ckpt_keys = [c for c in args.checkpoints.split(",") if c.strip()]

    binary = checkBinary("sumo")
    cfg = "./configurations/myconfig.sumocfg"
    from xml.dom.minidom import parse
    dom = parse(cfg)
    net_file = dom.getElementsByTagName("net-file")[0].attributes["value"].nodeValue
    route_file = "./configurations/" + dom.getElementsByTagName("route-files")[0].attributes["value"].nodeValue
    conn = ConnectionInfo("./configurations/" + net_file)
    runtime_cfg = build_runtime_sumocfg(cfg, fast_mode=True)

    results = []
    for seed in seeds:
        veh = gen_vehicles(route_file, conn, args.ctrl, args.unc, args.pattern, args.spawn, seed)
        t0 = time.time()
        base = run(DijkstraPolicy(conn), copy.deepcopy(veh), conn, runtime_cfg, args.port, binary)
        row = {"seed": seed, "dijkstra": {"avg": base["avg_travel_time"], "p90": base["p90_travel_time"],
               "compl": base["completion_rate"], "p95p50": base["p95_to_p50_travel_ratio"]}}
        for ck in ckpt_keys:
            for pol in policies:
                det = (pol == "greedy")
                pol_obj = MAPPOPolicy(copy.deepcopy(veh), conn, CKPTS[ck],
                                      net_xml_file=os.path.join("./configurations", net_file), deterministic=det)
                st = run(pol_obj, copy.deepcopy(veh), conn, runtime_cfg, args.port, binary)
                rm = st.get("controller_runtime_metrics") or {}
                key = f"{ck}.{pol}"
                row[key] = {"avg": st["avg_travel_time"], "p90": st["p90_travel_time"],
                            "compl": st["completion_rate"], "p95p50": st["p95_to_p50_travel_ratio"],
                            "nonzero": rm.get("route_choice_nonzero_rate", 0.0),
                            "margin": rm.get("route_mean_logit_margin", 0.0),
                            "eta_delta": rm.get("route_mean_eta_delta_steps", 0.0),
                            "delta_vs_base": st["avg_travel_time"] - base["avg_travel_time"]}
        row["secs"] = round(time.time() - t0, 1)
        results.append(row)
        # live print
        msg = f"seed {seed}  dij {base['avg_travel_time']:.0f}"
        for ck in ckpt_keys:
            for pol in policies:
                r = row[f"{ck}.{pol}"]
                msg += f"  | {ck}.{pol} {r['avg']:.0f} ({r['delta_vs_base']:+.0f})"
        msg += f"  [{row['secs']}s]"
        print(msg, flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    # aggregate
    print("\n=== AGGREGATE (mean over %d seeds) ===" % len(seeds))
    print("dijkstra avg %.1f" % (sum(r["dijkstra"]["avg"] for r in results)/len(results)))
    for ck in ckpt_keys:
        for pol in policies:
            k = f"{ck}.{pol}"
            avg = sum(r[k]["avg"] for r in results)/len(results)
            d = sum(r[k]["delta_vs_base"] for r in results)/len(results)
            wins = sum(1 for r in results if r[k]["avg"] < r["dijkstra"]["avg"])
            nz = sum(r[k]["nonzero"] for r in results)/len(results)
            mg = sum(r[k]["margin"] for r in results)/len(results)
            print(f"{k:14s} avg {avg:7.1f}  delta {d:+7.1f}  wins {wins}/{len(seeds)}  nonzero {nz:.3f}  margin {mg:.2f}")

if __name__ == "__main__":
    main()
