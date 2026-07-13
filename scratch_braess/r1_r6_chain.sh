#!/bin/bash
# Serialized R1+R6 chain. Training and eval share trips.trips.xml and
# configurations/rou/str_sumo_braess.rou.xml, so everything that regenerates
# demand must run one at a time:
#   1. wait for the currently running marginal-model eval (launched separately)
#   2. R1: eval the difference-reward model on the unified metric (Table 3 arm)
#   3. R6: 5 independent-seed trainings (scratch_braess/r6_train_seeds.sh)
set -u
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
export SUMO_HOME="$PWD/.venv/lib/python3.14/site-packages/sumo"
export PYTHONPATH="$PWD"

while pgrep -f eval_braess_inference.py >/dev/null; do sleep 60; done

# PRODUCTION checkpoints per scratch_braess/eval_headtohead.py (the run behind the
# paper's Table 3): marginal_noskip.best.pt and braess.best.pt.bak_difference.
# (REVISION_PLAN's snapshot names mappo_policy_braess_marginal.best.pt; that file is
# a different, weaker checkpoint.)
echo "=== R1 marginal(noskip, production) eval start $(date) ==="
.venv/bin/python eval_braess_inference.py \
  configurations/model/mappo_policy_braess_marginal_noskip.best.pt 20 \
  > scratch_braess/eval_r1_marginal_noskip.log 2>&1 || echo "marginal_noskip eval FAILED"
echo "=== R1 difference-model eval start $(date) ==="
.venv/bin/python eval_braess_inference.py \
  configurations/model/mappo_policy_braess.best.pt.bak_difference 20 \
  > scratch_braess/eval_r1_difference.log 2>&1 || echo "difference eval FAILED"
echo "=== R1 evals done $(date) ==="

bash scratch_braess/r6_train_seeds.sh
