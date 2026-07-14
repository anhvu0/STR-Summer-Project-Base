#!/usr/bin/env bash
# 3. Held-out evaluation: does the trained policy capture the coordination gap
#    (approach T_SO) or herd like the selfish equilibrium?
#
# Runs the production MAPPO policy plus both Dijkstra arms (static + dynamic/live-
# traveltime) through the SAME StrSumo harness on held-out seeds 7000+, seed-pinned,
# with a paired one-sided Wilcoxon test. Primary metric: tripinfo duration+departDelay
# over all road users (s/veh). Also writes per-arm tripinfo (mg_<seed>.xml / dd_<seed>.xml)
# under scratch_braess/tripinfo, which the `sacrifice` audit reads.
#
# Usage:   reproduce/run_eval.sh [MODEL] [N_SEEDS]
# Default: configurations/model/mappo_policy_braess.best.pt, 20 seeds.
# Runtime: ~10-20 min.
#
# Head-to-head variant (marginal vs difference reward + both baselines, one pass/seed):
#   $PY scratch_braess/eval_headtohead.py 20
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
"$PY" eval_braess_inference.py \
  "${1:-configurations/model/mappo_policy_braess.best.pt}" "${2:-20}"
