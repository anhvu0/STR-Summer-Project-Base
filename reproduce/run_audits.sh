#!/usr/bin/env bash
# 6. Equilibrium / deviation audits (one-vehicle best-response check).
#
# Two questions the reviewers asked:
#   (a) Is the MAPPO outcome an approximate equilibrium, plain policy suboptimality, or
#       sustained by individually costly route holds?  -> mappo-regret
#   (b) Are the DUE references actual equilibria, or does route mixing leave vehicles with
#       large regret?  -> br-due / due-iter / due-device
#
# Usage: reproduce/run_audits.sh <audit> [args...]
#   mappo-regret [summarize|compute] [N_SEEDS]   ex-post assignment-level regret + fleet
#                                                externality of the production MAPPO routes
#                                                (round-2 headline). `summarize` is fast and
#                                                reads cached per-vehicle results.
#   br-due       [MAX_ROUNDS] [START]            best-response-dynamics T_DUE (a defensible
#                                                equilibrium; duaIterate itself fails the check)
#   due-iter     [ITER] [N_SAMPLE]               equilibrium residual of the iterated-DUE reference
#   due-device   [N] [SPAWN] [N_SAMPLE]          equilibrium residual of the rerouting-device DUE proxy
#   sacrifice    [TRIPINFO_DIR] [TAG]            per-vehicle sacrifice/gain accounting vs
#                                                Dijkstra-dynamic (reads tripinfo from run_eval.sh;
#                                                default dir scratch_braess/tripinfo)
#
# Runtime: br-due / due-* resimulate many times and are heavy (tens of min); mappo-regret
#          summarize and sacrifice are fast.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
audit="${1:-mappo-regret}"; shift || true
case "$audit" in
  mappo-regret) "$PY" scratch_braess/braess_mappo_regret.py "$@" ;;
  br-due)       "$PY" scratch_braess/braess_br_due.py "$@" ;;
  due-iter)     "$PY" scratch_braess/due_iter_regret.py "$@" ;;
  due-device)   "$PY" scratch_braess/due_regret.py "$@" ;;
  sacrifice)    "$PY" scratch_braess/braess_sacrifice.py "$@" ;;
  *) echo "unknown audit '$audit' (see header for options)" >&2; exit 2 ;;
esac
