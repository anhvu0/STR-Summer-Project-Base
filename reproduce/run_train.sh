#!/usr/bin/env bash
# 2. Train the production selfless-routing policy (MAPPO, marginal-cost externality reward).
#
# The best checkpoint emerges early (~ep9) then drifts, so early-stop on the frozen eval is
# essential -- --best-model-output snapshots the best-so-far. The frozen eval runs at the
# paradox demand (--eval-spawn-interval 1.5); the CLI default 0.5 gridlocks and makes
# checkpoint selection meaningless.
#
# WARNING: never run training and an eval at the same time -- both regenerate the shared
# route files (configurations/rou/... and trips.trips.xml) and corrupt each other. Sequence
# them. Kill a run with:  pkill -9 -f 'python.*train_rl.py'
#
# Outputs: configurations/model/mappo_policy_braess.pt        (final)
#          configurations/model/mappo_policy_braess.best.pt   (best frozen-eval = paper model)
#          configurations/rl_episode_metrics.csv              (per-episode, flushed live)
# Runtime: ~1-2 h on CPU.
#
# Usage: reproduce/run_train.sh [extra train_rl.py flags...]   (appended to the recipe below)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
"$PY" train_rl.py \
  --sumocfg ./configurations/braess.sumocfg \
  --target-pattern 4 --reroute-epoch-edges 1 \
  --num-target-vehicles 240 --num-random-vehicles 5 --spawn-interval 1.5 \
  --team-reward-alpha 1.0 --team-reward-mode marginal --marginal-cost-scale 0.015 \
  --actor-lr 1.0e-3 --update-epochs 8 --min-transitions-per-update 1024 \
  --episodes 50 --eval-every 5 --eval-spawn-interval 1.5 --eval-stochastic-samples 0 \
  --eval-seeds "6000,6002,6004,6006,6008" \
  --model-output ./configurations/model/mappo_policy_braess.pt \
  --best-model-output ./configurations/model/mappo_policy_braess.best.pt \
  "$@"
