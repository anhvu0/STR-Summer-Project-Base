#!/bin/bash
# R5 chain: train the two reward-ablation policies (alpha=0 selfish, shared team
# reward) with the production config, then run the full ablation eval. Serialized
# because training and eval regenerate shared demand files.
set -u
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
export SUMO_HOME="$PWD/.venv/lib/python3.14/site-packages/sumo"
export PYTHONPATH="$PWD"
PY=.venv/bin/python

while pgrep -f "eval_braess_|train_rl.py" >/dev/null; do sleep 60; done
mkdir -p configurations/model/r5

train () {  # $1 tag, extra args after
  local tag=$1; shift
  echo "=== R5 train $tag start $(date) ==="
  $PY train_rl.py --sumocfg ./configurations/braess.sumocfg \
    --target-pattern 4 --reroute-epoch-edges 1 --num-target-vehicles 240 \
    --num-random-vehicles 5 --spawn-interval 1.5 \
    --marginal-cost-scale 0.015 --actor-lr 1.0e-3 \
    --update-epochs 8 --min-transitions-per-update 1024 --episodes 50 --eval-every 5 \
    --eval-spawn-interval 1.5 --eval-seeds "6000,6002,6004,6006,6008" \
    --torch-seed 101 \
    --model-output "./configurations/model/r5/mappo_braess_$tag.pt" \
    --best-model-output "./configurations/model/r5/mappo_braess_$tag.best.pt" \
    "$@" > "scratch_braess/r5_train_$tag.log" 2>&1 || echo "train $tag FAILED"
  cp configurations/rl_episode_metrics.csv "configurations/model/r5/episode_metrics_$tag.csv"
  cp configurations/rl_frozen_eval_metrics.csv "configurations/model/r5/frozen_eval_metrics_$tag.csv"
  echo "=== R5 train $tag done $(date) ==="
}

# alpha=0: reward mode irrelevant when alpha=0, keep marginal for parity.
train alpha0 --team-reward-alpha 0.0 --team-reward-mode marginal
train shared --team-reward-alpha 1.0 --team-reward-mode shared

echo "=== R5 ablation eval start $(date) ==="
$PY eval_braess_ablations.py 20 > scratch_braess/eval_r5_ablations.log 2>&1 \
  || echo "ablation eval FAILED"
tail -14 scratch_braess/eval_r5_ablations.log
echo "r5 all done $(date)"
