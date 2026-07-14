#!/usr/bin/env bash
# 1. Measured price of anarchy on the chained-Braess map.
#
# Reports T_static (naive selfish), T_DUE (congestion-aware selfish equilibrium) and
# T_SO (system optimum, via a grid over the per-diamond split). The coordination gap is
# T_DUE/T_SO -- the "is there a paradox to solve" number (Gate G2). Pure SUMO, no RL stack.
#
# Usage:   reproduce/run_poa.sh [N_VEHICLES] [SPAWN_INTERVAL]
# Default: 240 vehicles @ 1.5 s spawn (the training demand; coordination gap ~1.34).
# Runtime: ~2-4 min.
#
# Related probes (run directly if wanted):
#   $PY diag_braess_poa.py N SPAWN        # split-based probe: information vs coordination gap
#   $PY diag_bottleneck_poa.py N SPAWN    # the alternate single-bottleneck map
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
"$PY" diag_braess_due_so.py "${1:-240}" "${2:-1.5}"
