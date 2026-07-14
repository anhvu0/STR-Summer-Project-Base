#!/usr/bin/env bash
# 5. MARL attribution ablations (REVISION_PLAN R5).
#
# The Braess analogue of the NYC attribution table, through the standard protocol
# (held-out seeds 7000-7019, unified tripinfo metric over all road users, PYTHONHASHSEED=0).
# Isolates which components of the learned policy carry the gain.
#
# Usage:   reproduce/run_ablations.sh [N_SEEDS]
# Default: 20 seeds.
# Runtime: ~15-30 min (several arms).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
"$PY" eval_braess_ablations.py "${1:-20}"
