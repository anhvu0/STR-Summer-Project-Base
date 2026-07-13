#!/bin/bash
# Post-R6 serialized runs (everything here regenerates shared demand files, so
# strictly one at a time). Waits for r1_r6_chain.sh to log completion first.
set -u
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
export SUMO_HOME="$PWD/.venv/lib/python3.14/site-packages/sumo"
export PYTHONPATH="$PWD"
PY=.venv/bin/python
ART=Selfless_routing/reproduce/artifacts

until grep -q "R6 all seeds done" scratch_braess/r1_r6_chain.log 2>/dev/null; do sleep 60; done

echo "=== 1. production eval rerun + determinism check + decision log $(date) ==="
cp "$ART/braess_eval_mappo_policy_braess_marginal_noskip.best.csv" \
   "$ART/.det_check_prev.csv"
rm -f "$ART/braess_decisions_marginal_noskip.csv"
STR_DECISION_LOG="$ART/braess_decisions_marginal_noskip.csv" \
  $PY eval_braess_inference.py \
  configurations/model/mappo_policy_braess_marginal_noskip.best.pt 20 \
  > scratch_braess/eval_r1_marginal_noskip_rerun.log 2>&1
if cmp -s "$ART/braess_eval_mappo_policy_braess_marginal_noskip.best.csv" "$ART/.det_check_prev.csv"; then
  echo "DETERMINISM CHECK: PASS (artifact byte-identical across runs)"
else
  echo "DETERMINISM CHECK: FAIL (artifact differs; diff follows)"
  diff "$ART/.det_check_prev.csv" "$ART/braess_eval_mappo_policy_braess_marginal_noskip.best.csv" | head -20
fi

echo "=== 2. R3 sacrifice accounting on production $(date) ==="
$PY scratch_braess/braess_sacrifice.py \
  scratch_braess/tripinfo/mappo_policy_braess_marginal_noskip.best marginal_noskip \
  > scratch_braess/sacrifice_marginal_noskip.log 2>&1
cat scratch_braess/sacrifice_marginal_noskip.log

echo "=== 3. R6 held-out evals of the 5 seed checkpoints $(date) ==="
for s in 101 102 103 104 105; do
  $PY eval_braess_inference.py \
    "configurations/model/r6/mappo_braess_marginal_ts$s.best.pt" 20 \
    > "scratch_braess/eval_r6_ts$s.log" 2>&1 || echo "ts$s eval FAILED"
  tail -12 "scratch_braess/eval_r6_ts$s.log" | head -4
done

echo "=== 4. R4 baselines at nominal demand (spawn 1.5) $(date) ==="
$PY eval_braess_baselines.py 1.5 20 > scratch_braess/baselines_spawn1.5.log 2>&1 \
  || echo "baselines 1.5 FAILED"
tail -18 scratch_braess/baselines_spawn1.5.log

echo "=== 5. R4 demand sweep $(date) ==="
for sp in 1.2 1.35 1.8 2.1; do
  $PY eval_braess_baselines.py "$sp" 20 > "scratch_braess/baselines_spawn$sp.log" 2>&1 \
    || echo "baselines $sp FAILED"
  echo "--- spawn $sp:"
  grep -A7 "per-arm summary" "scratch_braess/baselines_spawn$sp.log" | tail -7
done

echo "post_r6 all done $(date)"
