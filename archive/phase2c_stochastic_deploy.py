"""Phase 2c: stochastic-DEPLOYMENT eval of the phase2b checkpoints.

The frozen-eval stochastic pass during training is a *diagnostic* (sampled routing,
Layer A OFF). This harness measures the deployable variant instead: sampled routing
with the recalibrated Layer A ON, so the fleet spreads across alternatives while the
spare-capacity veto still guards pointless/pile-on detours.

Motivation (phase2b, 150 eps / 30 seeds): the sampled policy improved near-monotonically
while greedy argmax oscillated, and at the final checkpoint the Layer-A-off sampled eval
BEAT the greedy deployment (449.5 vs 469.2 avg). If sampling also wins with Layer A on,
stochastic deployment is a zero-retrain fix for the argmax-herding churn.

Runs 2 checkpoints (ep74 best, ep149 final) x 30 held-out seeds x 3 samples.
Dijkstra baselines and ep74 greedy per-seed numbers are reused from the best-checkpoint
metadata (they are deterministic given seed + config), so no baseline reruns.

Writes per-run rows to configurations/phase2c_stochastic_deploy.csv (a NEW file; the
live rl_*_metrics.csv files are never opened) and prints paired aggregates at the end.

Run:
  PYTHONPATH=. SUMO_HOME=.venv/lib/python3.14/site-packages/sumo .venv/bin/python \
      phase2c_stochastic_deploy.py
"""

import copy
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
import traci
from sumolib import checkBinary

from controller.MAPPOController import MAPPOPolicy
from core.STR_SUMO import StrSumo
from core.mappo import MAPPOConfig
from core.rl_training_pipeline import RLTrainingPipeline

EVAL_SEEDS = list(range(6000, 6030))
SAMPLES_PER_SEED = 3
CHECKPOINTS = {
    "ep74_best": "configurations/model/mappo_policy_nyc_phase2b.best.pt",
    "ep149_final": "configurations/model/mappo_policy_nyc_phase2b.pt",
}
BEST_META = "configurations/model/mappo_policy_nyc_phase2b.best.pt.meta.json"
OUTPUT_CSV = "configurations/phase2c_stochastic_deploy.csv"


def build_pipeline():
    # Only used for vehicle generation, the runtime sumocfg, and connection_info.
    # CSV truncation happens inside .train()/.run(), which is never called here.
    return RLTrainingPipeline(
        sumocfg_path="./configurations/myconfig.sumocfg",
        model_output_path="configurations/model/phase2c_unused.pt",
        best_model_output_path="configurations/model/phase2c_unused.best.pt",
        episodes=0,
        spawn_interval=0.5,
        mappo_config=MAPPOConfig(),
        eval_every=0,
        frozen_eval_seeds=EVAL_SEEDS,
        eval_spawn_interval=0.5,
        fast_training_profile=True,
        target_pattern=2,
        num_target_vehicles=450,
        num_random_vehicles=150,
    )


def run_episode(pipeline, sumo_binary, controller, vehicles):
    simulation = StrSumo(controller, pipeline.connection_info, vehicles)
    try:
        traci.start([
            sumo_binary,
            "-c", pipeline.runtime_sumocfg_path,
            "--quit-on-end",
            "--no-step-log",
            "--no-warnings",
        ])
        _, _, _, stats = simulation.run(verbose=False, return_stats=True, print_runtime_summary=False)
    finally:
        try:
            traci.close()
        except Exception:
            pass
    return stats


def sign_test_p(deltas):
    """Two-sided sign test on paired deltas (ignores zeros)."""
    wins = sum(1 for d in deltas if d < 0)
    losses = sum(1 for d in deltas if d > 0)
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return min(p, 1.0)


def main():
    with open(BEST_META) as f:
        meta = json.load(f)
    reference = {int(row["seed"]): row for row in meta["per_seed"]}
    assert set(reference) == set(EVAL_SEEDS), "meta.json seeds do not match EVAL_SEEDS"

    pipeline = build_pipeline()
    sumo_binary = checkBinary("sumo")
    net_xml = os.path.join(pipeline.sumocfg_dir, pipeline.net_file)

    fields = [
        "checkpoint", "seed", "sample_idx",
        "avg_travel_time", "p50_travel_time", "p90_travel_time",
        "completion_rate", "route_choice_nonzero_rate",
        "baseline_avg_travel_time", "baseline_p90_travel_time",
        "greedy_ep74_avg_travel_time", "greedy_ep74_p90_travel_time",
    ]
    out = open(OUTPUT_CSV, "w", newline="")
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()

    results = defaultdict(lambda: defaultdict(list))  # ckpt -> seed -> [stats]
    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        for seed in EVAL_SEEDS:
            vehicles = pipeline.generate_episode_vehicles(
                episode_seed=int(seed),
                spawn_interval_override=pipeline.eval_spawn_interval,
            )
            for sample_idx in range(SAMPLES_PER_SEED):
                np.random.seed(int(seed) * 1000 + sample_idx)
                sample_vehicles = copy.deepcopy(vehicles)
                policy = MAPPOPolicy(
                    sample_vehicles,
                    pipeline.connection_info,
                    ckpt_path,
                    net_xml_file=net_xml,
                    deterministic=False,
                    # detour_throttle left at default True: Layer A ON = deployment config.
                )
                stats = run_episode(pipeline, sumo_binary, policy, sample_vehicles)
                runtime = stats.get("controller_runtime_metrics") or {}
                ref = reference[int(seed)]
                row = {
                    "checkpoint": ckpt_name,
                    "seed": int(seed),
                    "sample_idx": sample_idx,
                    "avg_travel_time": float(stats["avg_travel_time"]),
                    "p50_travel_time": float(stats["p50_travel_time"]),
                    "p90_travel_time": float(stats["p90_travel_time"]),
                    "completion_rate": float(stats["completion_rate"]),
                    "route_choice_nonzero_rate": float(runtime.get("route_choice_nonzero_rate", 0.0)),
                    "baseline_avg_travel_time": float(ref["baseline_avg_travel_time"]),
                    "baseline_p90_travel_time": float(ref["baseline_p90_travel_time"]),
                    "greedy_ep74_avg_travel_time": float(ref["avg_travel_time"]),
                    "greedy_ep74_p90_travel_time": float(ref["p90_travel_time"]),
                }
                writer.writerow(row)
                out.flush()
                results[ckpt_name][int(seed)].append(row)
                print(
                    f"[{ckpt_name}] seed {seed} sample {sample_idx}: "
                    f"avg {row['avg_travel_time']:.1f} p90 {row['p90_travel_time']:.1f} "
                    f"compl {row['completion_rate']:.3f} detour {row['route_choice_nonzero_rate']:.3f}",
                    flush=True,
                )
    out.close()

    print("\n===== PHASE 2C SUMMARY (stochastic deployment: sampled + Layer A ON) =====")
    for ckpt_name in CHECKPOINTS:
        per_seed = results[ckpt_name]
        seed_avg = {s: float(np.mean([r["avg_travel_time"] for r in rows])) for s, rows in per_seed.items()}
        seed_p90 = {s: float(np.mean([r["p90_travel_time"] for r in rows])) for s, rows in per_seed.items()}
        seed_compl = {s: float(np.mean([r["completion_rate"] for r in rows])) for s, rows in per_seed.items()}
        seed_rate = {s: float(np.mean([r["route_choice_nonzero_rate"] for r in rows])) for s, rows in per_seed.items()}

        base_avg = {s: float(reference[s]["baseline_avg_travel_time"]) for s in per_seed}
        base_p90 = {s: float(reference[s]["baseline_p90_travel_time"]) for s in per_seed}
        greedy_avg = {s: float(reference[s]["avg_travel_time"]) for s in per_seed}
        greedy_p90 = {s: float(reference[s]["p90_travel_time"]) for s in per_seed}

        d_avg_vs_greedy = [seed_avg[s] - greedy_avg[s] for s in per_seed]
        d_p90_vs_greedy = [seed_p90[s] - greedy_p90[s] for s in per_seed]

        print(f"\n--- {ckpt_name} ({CHECKPOINTS[ckpt_name]}) ---")
        print(f"avg  : {np.mean(list(seed_avg.values())):.1f} "
              f"(Dijkstra {np.mean(list(base_avg.values())):.1f}, greedy-ep74 {np.mean(list(greedy_avg.values())):.1f})")
        print(f"p90  : {np.mean(list(seed_p90.values())):.1f} "
              f"(Dijkstra {np.mean(list(base_p90.values())):.1f}, greedy-ep74 {np.mean(list(greedy_p90.values())):.1f})")
        print(f"compl: {np.mean(list(seed_compl.values())):.4f}   detour rate: {np.mean(list(seed_rate.values())):.3f}")
        print(f"wins vs Dijkstra   (avg): {sum(seed_avg[s] < base_avg[s] for s in per_seed)}/{len(per_seed)}   "
              f"(p90): {sum(seed_p90[s] < base_p90[s] for s in per_seed)}/{len(per_seed)}")
        print(f"wins vs greedy-ep74(avg): {sum(seed_avg[s] < greedy_avg[s] for s in per_seed)}/{len(per_seed)}   "
              f"(p90): {sum(seed_p90[s] < greedy_p90[s] for s in per_seed)}/{len(per_seed)}")
        print(f"paired delta vs greedy-ep74: avg {np.mean(d_avg_vs_greedy):+.1f}s "
              f"(sign-test p={sign_test_p(d_avg_vs_greedy):.3f}), "
              f"p90 {np.mean(d_p90_vs_greedy):+.1f}s (sign-test p={sign_test_p(d_p90_vs_greedy):.3f})")


if __name__ == "__main__":
    main()
