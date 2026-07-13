#!/bin/bash
# R6 (REVISION_PLAN_2026-07-13): 5 independent training seeds of the Braess
# production config (marginal-cost reward). Sequential on purpose: training and
# eval share trips.trips.xml / configurations/rou/str_sumo_braess.rou.xml, so
# nothing else touching those files may run concurrently. Waits for any running
# eval_braess_inference.py first for the same reason.
set -u
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
export SUMO_HOME="$PWD/.venv/lib/python3.14/site-packages/sumo"
export PYTHONPATH="$PWD"

while pgrep -f eval_braess_inference.py >/dev/null; do sleep 60; done

mkdir -p configurations/model/r6
# preserve the production training logs before R6 runs overwrite the shared CSVs
[ -f configurations/model/r6/production_episode_metrics.csv ] || \
  cp configurations/rl_episode_metrics.csv configurations/model/r6/production_episode_metrics.csv
[ -f configurations/model/r6/production_frozen_eval_metrics.csv ] || \
  cp configurations/rl_frozen_eval_metrics.csv configurations/model/r6/production_frozen_eval_metrics.csv

for s in 101 102 103 104 105; do
  echo "=== R6 torch-seed $s start $(date) ==="
  .venv/bin/python train_rl.py --sumocfg ./configurations/braess.sumocfg \
    --target-pattern 4 --reroute-epoch-edges 1 --num-target-vehicles 240 \
    --num-random-vehicles 5 --spawn-interval 1.5 --team-reward-alpha 1.0 \
    --team-reward-mode marginal --marginal-cost-scale 0.015 --actor-lr 1.0e-3 \
    --update-epochs 8 --min-transitions-per-update 1024 --episodes 50 --eval-every 5 \
    --eval-spawn-interval 1.5 --eval-seeds "6000,6002,6004,6006,6008" \
    --torch-seed "$s" \
    --model-output "./configurations/model/r6/mappo_braess_marginal_ts$s.pt" \
    --best-model-output "./configurations/model/r6/mappo_braess_marginal_ts$s.best.pt" \
    > "scratch_braess/r6_train_ts$s.log" 2>&1 || echo "seed $s FAILED (exit $?)"
  cp configurations/rl_episode_metrics.csv "configurations/model/r6/episode_metrics_ts$s.csv"
  cp configurations/rl_frozen_eval_metrics.csv "configurations/model/r6/frozen_eval_metrics_ts$s.csv"
  echo "=== R6 torch-seed $s done $(date) ==="
done
echo "R6 all seeds done $(date)"
