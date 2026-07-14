#!/usr/bin/env bash
# 4. Learning-necessity baselines (REVISION_PLAN R4).
#
# The production MAPPO policy vs non-learning controllers under the identical protocol as
# run_eval.sh (same harness, same held-out seeds, same tripinfo duration+departDelay metric
# over all road users). Answers "does the learning buy anything a fixed rule doesn't?".
#
# Usage:   reproduce/run_baselines.sh [SPAWN] [N_SEEDS]
# Default: spawn 1.5, 20 seeds.
# Runtime: ~10-20 min.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
"$PY" eval_braess_baselines.py "${1:-1.5}" "${2:-20}"
