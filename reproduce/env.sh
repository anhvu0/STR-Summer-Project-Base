#!/usr/bin/env bash
# Shared environment for every reproduce/ wrapper. `source` this; do not execute it.
#
# It cd's to the repo root (so configurations/, scratch_braess/, ... resolve) and exports
# the three variables the harness needs to be deterministic:
#
#   PYTHONHASHSEED=0  Pins Python dict/set ordering, which fixes the controller's
#                     route-commit order. Without it, near-gridlock seeds are chaotically
#                     sensitive: two runs of the same seed can differ by hundreds of
#                     seconds on one or two scenarios that then dominate a 20-seed mean.
#   SUMO_HOME         Points at the SUMO installed as a pip package inside .venv.
#   PYTHONPATH=.      Lets the scripts import core/ and controller/ from the repo root.
#
# It also defines PY, the project interpreter (torch, numpy, and the SUMO python client
# are installed only in this venv).

REPRODUCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$REPRODUCE_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONHASHSEED=0
export SUMO_HOME="${SUMO_HOME:-$REPO_ROOT/.venv/lib/python3.14/site-packages/sumo}"
export PYTHONPATH="${PYTHONPATH:-.}"

PY="$REPO_ROOT/.venv/bin/python"
