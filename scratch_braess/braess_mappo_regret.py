"""Ex-post assignment-level regret + fleet externality for the production MAPPO
(marginal, no-skip) outcomes on the 20 held-out Braess seeds.

Review round 2 asks whether MAPPO's equilibrium-band welfare is (a) an
approximate equilibrium, (b) plain policy suboptimality, or (c) sustained by
individually costly route holds. The instrument is the same one-vehicle
best-response check that validated the BR-dynamics band (braess_br_due.py),
pointed at MAPPO's realized routes, extended to record the FLEET effect of each
profitable unilateral deviation:

  regret(v)      = t_base(v) - min over the 9 fixed (leg1, leg2) combos t_dev(v)
  fleet_delta(v) = fleet_total(best deviation of v) - fleet_total(base)
  rest_delta(v)  = fleet_delta(v) - (t_best(v) - t_base(v))   [externality on others]

This is ex-post ASSIGNMENT-level regret: realized routes are frozen and replayed
open-loop, so neither the deciding policy nor the other agents react to the
deviation. Metric everywhere: tripinfo duration + departDelay over ALL road
users, unfinished included (same as eval_braess_inference.py / diag_braess_due_so).

Frozen runs use the SAME instrument as the BR-dynamics band (braess_br_due.py):
duaiterate replay options + SUMO default seed, single runs. Replaying with
--seed <episode seed> instead puts some seeds on a chaos knife-edge (seed 7000:
538 s vs 350-379 s under five other draws, live 337.6 s); the default-seed BR
instrument reproduces every live mean within +38/-4 s (mean +9.5 s, the open-loop
replay cost) and keeps the 13.6 s band residual comparable.

Phases (run in order; each is resumable and skips seeds already done):
  extract   PYTHONHASHSEED=0 SUMO_HOME=... PYTHONPATH=. .venv/bin/python \
                scratch_braess/braess_mappo_regret.py extract [n_seeds]
            Re-runs the MAPPO-greedy arm per seed with --vehroute-output and
            verifies the tripinfo mean against the production artifact
            (braess_eval_mappo_policy_braess_marginal_noskip.best.csv).
  deviate   SUMO_HOME=... .venv/bin/python scratch_braess/braess_mappo_regret.py deviate [n_seeds]
            Frozen base replay + 240 vehicles x 9 combos per seed (parallel).
  summarize .venv/bin/python scratch_braess/braess_mappo_regret.py summarize
            Writes Selfless_routing/reproduce/artifacts/braess_mappo_regret_{per_vehicle,summary}.csv
"""
import csv
import json
import os
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diag_braess_due_so as diag

PHASE = sys.argv[1] if len(sys.argv) > 1 else "summarize"
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SEEDS = [7000 + i for i in range(N_SEEDS)]
SPAWN = 1.5
MODEL = "configurations/model/mappo_policy_braess_marginal_noskip.best.pt"
REF_ART = ("Selfless_routing/reproduce/artifacts/"
           "braess_eval_mappo_policy_braess_marginal_noskip.best.csv")
WORK = "scratch_braess/mappo_regret"
ART = "Selfless_routing/reproduce/artifacts"
TMP = os.environ.get("TMPDIR", WORK)
COMBOS = [(l1, l2) for l1 in ("braess", "up", "down") for l2 in ("braess", "up", "down")]
EPS = 5.0          # braess_br_due.py noise threshold
FLOOR = 13.6       # BR-band settled median-regret residual (numbers.tex BraessBRMedRegret)
WORKERS = 11
os.makedirs(WORK, exist_ok=True)


def parse_tripinfo(path):
    """Unified welfare metric rows: id -> duration + max(departDelay, 0)."""
    tri = ET.parse(path).getroot()
    tts, unfinished = {}, 0
    for t in tri.iter("tripinfo"):
        dd = max(float(t.get("departDelay", 0.0)), 0.0)
        tts[t.get("id")] = float(t.get("duration", 0.0)) + dd
        if t.get("arrival", "-1") in ("-1", "-1.00"):
            unfinished += 1
    return tts, unfinished


# --------------------------------------------------------------------------- extract
def extract():
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("ERROR: extract needs PYTHONHASHSEED=0 (paper protocol)")
    try:
        import libsumo as _l
        sys.modules["traci"] = _l
        sys.modules["traci.constants"] = _l.constants
    except ImportError:
        pass
    import copy
    import random as _random

    import numpy as _np
    import torch as _torch
    import traci
    from sumolib import checkBinary

    from controller.MAPPOController import MAPPOPolicy
    from core.mappo import MAPPOConfig
    from core.rl_training_pipeline import RLTrainingPipeline
    from core.STR_SUMO import StrSumo

    pipe = RLTrainingPipeline(
        sumocfg_path="./configurations/braess.sumocfg",
        model_output_path="scratch_braess/_regretdummy.pt",
        episodes=1, spawn_interval=SPAWN, mappo_config=MAPPOConfig(),
        target_pattern=4, num_target_vehicles=240, num_random_vehicles=5,
        reroute_epoch_edges=1, eval_every=0, fast_training_profile=True,
        team_reward_alpha=1.0, team_reward_mode="marginal")
    sumo = checkBinary("sumo")
    net = os.path.join(pipe.sumocfg_dir, pipe.net_file)

    ref = {}
    with open(REF_ART) as fh:
        for r in csv.DictReader(fh):
            if r["arm"] == "MAPPO-greedy":
                ref[int(r["seed"])] = float(r["tripinfo_mean_tt"])

    for seed in SEEDS:
        vr = f"{WORK}/{seed}.vehroutes.xml"
        if os.path.exists(vr):
            print(f"seed {seed}: already extracted, skip")
            continue
        vehicles = pipe.generate_episode_vehicles(episode_seed=int(seed),
                                                  spawn_interval_override=SPAWN)
        ctrl = MAPPOPolicy(copy.deepcopy(vehicles), pipe.connection_info, MODEL,
                           net_xml_file=net, deterministic=True, reroute_epoch_edges=1)
        _random.seed(int(seed)); _np.random.seed(int(seed)); _torch.manual_seed(int(seed))
        sim = StrSumo(ctrl, pipe.connection_info, copy.deepcopy(vehicles))
        tp = f"{WORK}/{seed}.live.tripinfo.xml"
        traci.start([sumo, "-c", pipe.runtime_sumocfg_path, "--quit-on-end",
                     "--no-step-log", "--no-warnings", "--seed", str(int(seed)),
                     "--tripinfo-output", tp, "--tripinfo-output.write-unfinished",
                     "--vehroute-output", vr + ".tmp",
                     "--vehroute-output.intended-depart",
                     "--vehroute-output.write-unfinished"])
        try:
            sim.run(verbose=False, return_stats=True, print_runtime_summary=False)
        finally:
            try:
                traci.close()
            except Exception:
                pass
        tts, unfinished = parse_tripinfo(tp)
        live_mean = sum(tts.values()) / len(tts)
        delta = live_mean - ref.get(seed, float("nan"))
        with open(f"{WORK}/{seed}.controlled.json", "w") as fh:
            json.dump({"controlled": sorted(vehicles.keys(), key=int),
                       "live_mean": live_mean, "n": len(tts),
                       "unfinished": unfinished}, fh)
        os.replace(vr + ".tmp", vr)
        print(f"seed {seed}: live mean {live_mean:.3f}s vs artifact "
              f"{ref.get(seed, float('nan')):.3f}s (delta {delta:+.3f}s), "
              f"n={len(tts)}, unfinished={unfinished}", flush=True)


# --------------------------------------------------------------------------- deviate
def load_realized(seed):
    root = ET.parse(f"{WORK}/{seed}.vehroutes.xml").getroot()
    vehs = []
    for v in root.iter("vehicle"):
        rt = v.find("route")
        if rt is None:  # rerouted vehicles carry a routeDistribution; final = last
            dist = v.find("routeDistribution")
            rt = [r for r in dist.iter("route")][-1]
        vehs.append((v.get("id"), float(v.get("depart")), rt.get("edges")))
    vehs.sort(key=lambda x: (x[1], int(x[0])))
    return vehs


def write_rou(path, vehs, override=None):
    over = override or {}
    with open(path, "w") as f:
        f.write("<routes>\n")
        for vid, depart, edges in vehs:
            f.write(f'  <vehicle id="{vid}" depart="{depart:.2f}">'
                    f'<route edges="{over.get(vid, edges)}"/></vehicle>\n')
        f.write("</routes>\n")


def deviate():
    diag.SUMO = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo")

    def run_frozen(name, seed, vehs, override=None):
        rou = os.path.join(TMP, f"_mr_{name}.rou.xml")
        trips = os.path.join(TMP, f"_mr_{name}.tripinfo.xml")
        write_rou(rou, vehs, override)
        import subprocess
        subprocess.run([diag.SUMO, "-n", diag.NET, "-r", rou,
                        "--tripinfo-output", trips,
                        "--tripinfo-output.write-unfinished",
                        "--end", "200000",
                        "--route-steps", "200", "--time-to-teleport", "-1",
                        "--time-to-teleport.highways", "0",
                        "--no-step-log", "--no-warnings"],
                       check=True, capture_output=True)
        tts, unfinished = parse_tripinfo(trips)
        for fn in (rou, trips):
            if os.path.exists(fn):
                os.remove(fn)
        return tts, unfinished

    for seed in SEEDS:
        out = f"{WORK}/{seed}.dev.csv"
        if os.path.exists(out):
            print(f"seed {seed}: deviations already done, skip")
            continue
        if not os.path.exists(f"{WORK}/{seed}.vehroutes.xml"):
            print(f"seed {seed}: no vehroutes yet, skip (run extract first)")
            continue
        with open(f"{WORK}/{seed}.controlled.json") as fh:
            meta = json.load(fh)
        controlled = set(meta["controlled"])
        vehs = load_realized(seed)
        realized = {vid: edges for vid, _, edges in vehs}

        base_tts, base_unf = run_frozen(f"{seed}_base", seed, vehs)
        base_fleet = sum(base_tts.values())
        base_mean = base_fleet / len(base_tts)
        print(f"seed {seed}: frozen base mean {base_mean:.2f}s "
              f"(live {meta['live_mean']:.2f}s, replay delta "
              f"{base_mean - meta['live_mean']:+.2f}s), unfinished={base_unf}", flush=True)

        def dev_run(job):
            vid, c = job
            src = realized[vid].split()[0]
            alt = diag.route_edges(src, *c)
            if alt == realized[vid]:
                return vid, c, base_tts[vid], base_fleet, base_unf, 1
            tts, unf = run_frozen(f"{seed}_{vid}_{c[0][:2]}{c[1][:2]}", seed, vehs,
                                  {vid: alt})
            return vid, c, tts.get(vid, float("nan")), sum(tts.values()), unf, 0

        ctrl_ids = [vid for vid, _, _ in vehs if vid in controlled]
        jobs = [(vid, c) for vid in ctrl_ids for c in COMBOS]
        rows = []
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for i, (vid, c, tt, fleet, unf, is_base) in enumerate(ex.map(dev_run, jobs)):
                rows.append([seed, vid, f"{c[0]}|{c[1]}", round(tt, 2),
                             round(fleet, 2), unf, is_base])
                if (i + 1) % 500 == 0:
                    print(f"  seed {seed}: {i + 1}/{len(jobs)} deviations", flush=True)
        with open(out + ".tmp", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["seed", "vehicle", "combo", "dev_tt", "fleet_sum",
                        "unfinished", "is_realized"])
            w.writerow([seed, "__base__", "", round(base_mean, 3),
                        round(base_fleet, 2), base_unf, 1])
            for vid in ctrl_ids:
                w.writerow([seed, f"__basett__{vid}", "", round(base_tts[vid], 2),
                            round(base_fleet, 2), base_unf, 1])
            w.writerows(rows)
        os.replace(out + ".tmp", out)
        regs = []
        for vid in ctrl_ids:
            best = min(tt for s, v, c, tt, *_ in
                       [(r[0], r[1], r[2], r[3]) for r in rows if r[1] == vid])
            regs.append(max(base_tts[vid] - best, 0.0))
        regs.sort()
        med = regs[len(regs) // 2]
        frac5 = sum(1 for r in regs if r > EPS) / len(regs)
        print(f"seed {seed}: median regret {med:.1f}s | frac>{EPS:.0f}s {frac5:.0%}",
              flush=True)


# ------------------------------------------------------------------------- summarize
def summarize():
    per_vehicle, per_seed = [], []
    # Fragility control: rest-of-fleet effect of EVERY alternate, grouped by
    # whether the switch helps the deviator. If rest-harm is generic (all
    # groups alike), positive fleet deltas would be chaos fragility, not
    # evidence that the profitable holds specifically protect the fleet.
    rest_groups = {"profitable": [], "neutral": [], "self_harm": []}
    combo_gainers, combo_realized = {}, {}
    for seed in SEEDS:
        path = f"{WORK}/{seed}.dev.csv"
        if not os.path.exists(path):
            continue
        with open(f"{WORK}/{seed}.controlled.json") as fh:
            meta = json.load(fh)
        base_tt, devs, base_fleet = {}, {}, None
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if r["vehicle"] == "__base__":
                    base_fleet = float(r["fleet_sum"])
                    base_mean = float(r["dev_tt"])
                elif r["vehicle"].startswith("__basett__"):
                    base_tt[r["vehicle"][10:]] = float(r["dev_tt"])
                else:
                    devs.setdefault(r["vehicle"], []).append(
                        (float(r["dev_tt"]), float(r["fleet_sum"]), r["combo"],
                         int(r["unfinished"])))
        regs = []
        for vid, alts in devs.items():
            for tt, fleet, combo, unf in alts:
                own_d = tt - base_tt[vid]
                rest_d = (fleet - base_fleet) - own_d
                if own_d < -FLOOR:
                    rest_groups["profitable"].append(rest_d)
                elif own_d > FLOOR:
                    rest_groups["self_harm"].append(rest_d)
                else:
                    rest_groups["neutral"].append(rest_d)
            t_best, fleet_best, combo_best, unf_best = min(alts)
            reg = max(base_tt[vid] - t_best, 0.0)
            if reg > FLOOR:
                combo_gainers[combo_best] = combo_gainers.get(combo_best, 0) + 1
            fleet_delta = fleet_best - base_fleet
            own_delta = t_best - base_tt[vid]
            rest_delta = fleet_delta - own_delta
            regs.append(reg)
            per_vehicle.append({
                "seed": seed, "vehicle": vid, "base_tt": round(base_tt[vid], 2),
                "best_alt_tt": round(t_best, 2), "regret": round(reg, 2),
                "best_combo": combo_best, "fleet_delta": round(fleet_delta, 2),
                "rest_delta": round(rest_delta, 2), "dev_unfinished": unf_best})
        regs.sort()
        n = len(regs)
        med = regs[n // 2] if n % 2 else 0.5 * (regs[n // 2 - 1] + regs[n // 2])
        p90 = regs[min(int(0.9 * (n - 1) + 0.999), n - 1)]
        gainers = [pv for pv in per_vehicle if pv["seed"] == seed
                   and pv["regret"] > FLOOR]
        per_seed.append({
            "seed": seed, "live_mean": round(meta["live_mean"], 2),
            "frozen_base_mean": round(base_mean, 2),
            "median_regret": round(med, 2), "p90_regret": round(p90, 2),
            "frac_gt5": round(sum(1 for r in regs if r > EPS) / n, 4),
            "frac_gt_floor": round(sum(1 for r in regs if r > FLOOR) / n, 4),
            "n_gainers_floor": len(gainers),
            "gainers_fleet_worse": sum(1 for g in gainers if g["fleet_delta"] > 0),
            "gainers_fleet_better": sum(1 for g in gainers if g["fleet_delta"] < 0),
            "mean_fleet_delta_gainers": round(
                sum(g["fleet_delta"] for g in gainers) / len(gainers), 2)
            if gainers else 0.0})
    if not per_seed:
        sys.exit("no per-seed deviation files found; run deviate first")

    os.makedirs(ART, exist_ok=True)
    with open(f"{ART}/braess_mappo_regret_per_vehicle.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_vehicle[0].keys()))
        w.writeheader(); w.writerows(per_vehicle)
    with open(f"{ART}/braess_mappo_regret_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_seed[0].keys()))
        w.writeheader(); w.writerows(per_seed)

    meds = sorted(r["median_regret"] for r in per_seed)
    pooled = sorted(pv["regret"] for pv in per_vehicle)
    np_ = len(pooled)
    pooled_med = pooled[np_ // 2]
    gainers = [pv for pv in per_vehicle if pv["regret"] > FLOOR]
    worse = [g for g in gainers if g["fleet_delta"] > 0]
    better = [g for g in gainers if g["fleet_delta"] < 0]
    replay_deltas = [r["frozen_base_mean"] - r["live_mean"] for r in per_seed]
    print(f"seeds analyzed: {len(per_seed)}")
    print(f"replay fidelity: frozen-base minus live mean, per-seed "
          f"mean {sum(replay_deltas)/len(replay_deltas):+.2f}s "
          f"range [{min(replay_deltas):+.2f}, {max(replay_deltas):+.2f}]")
    print(f"per-seed median regret: mean {sum(meds)/len(meds):.2f}s "
          f"range [{meds[0]:.2f}, {meds[-1]:.2f}]  (BR-band residual {FLOOR})")
    print(f"pooled per-vehicle regret: median {pooled_med:.2f}s | "
          f"frac>{EPS:.0f}s {sum(1 for r in pooled if r > EPS)/np_:.1%} | "
          f"frac>{FLOOR}s {len(gainers)/np_:.1%}")
    if gainers:
        mean_fd = sum(g["fleet_delta"] for g in gainers) / len(gainers)
        mean_rd = sum(g["rest_delta"] for g in gainers) / len(gainers)
        mean_own = sum(g["regret"] for g in gainers) / len(gainers)
        print(f"gainers (regret>{FLOOR}s): {len(gainers)} vehicles "
              f"({len(gainers)/np_:.1%} of fleet)")
        print(f"  fleet WORSE after their best deviation: {len(worse)} "
              f"({len(worse)/len(gainers):.0%})")
        print(f"  fleet BETTER after their best deviation: {len(better)} "
              f"({len(better)/len(gainers):.0%})")
        print(f"  mean own gain {mean_own:.1f}s | mean fleet delta {mean_fd:+.1f}s "
              f"| mean rest-of-fleet delta {mean_rd:+.1f}s")
        try:
            from scipy.stats import wilcoxon
            _, p = wilcoxon([g["fleet_delta"] for g in gainers],
                            alternative="greater")
            print(f"  Wilcoxon fleet_delta > 0 (selfless-hold direction): p={p:.4g}")
        except Exception:
            pass
    print("fragility control, mean rest-of-fleet delta per alternate group:")
    for grp, xs in rest_groups.items():
        if xs:
            xs_s = sorted(xs)
            print(f"  {grp:11s}: n={len(xs):6d}  mean {sum(xs)/len(xs):+8.1f}s  "
                  f"median {xs_s[len(xs_s)//2]:+8.1f}s")
    if combo_gainers:
        cross = sum(v for k, v in combo_gainers.items() if "braess" in k)
        print(f"best deviation of gainers uses a cross (braess) leg: "
              f"{cross}/{sum(combo_gainers.values())} "
              f"({100.0*cross/sum(combo_gainers.values()):.0f}%)")
        top = sorted(combo_gainers.items(), key=lambda x: -x[1])[:4]
        print(f"  top target combos: {top}")


if PHASE == "extract":
    extract()
elif PHASE == "deviate":
    deviate()
elif PHASE == "summarize":
    summarize()
else:
    sys.exit(f"unknown phase {PHASE}")
