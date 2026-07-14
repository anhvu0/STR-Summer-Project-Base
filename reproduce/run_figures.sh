#!/usr/bin/env bash
# 7. Regenerate the paper's numbers macros and figures.
#
# NOTE: the generators live inside Selfless_routing/reproduce/, which is the LaTeX paper
# working directory and is gitignored -- it ships separately from the code. If that
# directory is absent this wrapper warns and exits cleanly.
#
# Most generators read cached CSVs under Selfless_routing/reproduce/artifacts/, so they run
# in seconds and need no simulation. Regenerate those CSVs first with run_eval.sh /
# run_baselines.sh / run_ablations.sh / run_audits.sh if you changed the model or protocol.
#
# Usage: reproduce/run_figures.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
REPRO="Selfless_routing/reproduce"
if [[ ! -d "$REPRO" ]]; then
  echo "warning: $REPRO not present (paper dir is gitignored); skipping figures." >&2
  exit 0
fi
"$PY" "$REPRO/braess_numbers.py"        # -> Selfless_routing/numbers.tex (paper \newcommand macros)
"$PY" "$REPRO/make_braess_figure.py"    # -> Selfless_routing/Images/braess.pdf
# Legacy NYC / penetration figures (uncomment only if rebuilding those sections):
#   "$PY" "$REPRO/make_poa_figure.py"
#   "$PY" "$REPRO/make_penetration_figure.py"
#   "$PY" "$REPRO/make_training_figure.py"
#   "$PY" "$REPRO/make_deployment_figure.py"
