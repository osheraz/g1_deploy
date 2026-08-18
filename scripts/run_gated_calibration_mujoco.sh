#!/usr/bin/env bash
# Review-gated, two-shoulder suspended 23-DoF MuJoCo calibration rehearsal.
set -euo pipefail

ROOT="/home/clhsieh/e_skin"
PYTHON="/data/clhsieh/environments/python/venvs/e-skin-4090-verify/bin/python"
[[ -x "$PYTHON" ]] || { echo "ERROR: missing Python: $PYTHON" >&2; exit 2; }

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output="/data/clhsieh/logs/e_skin/calibration/review_gated_mujoco/${stamp}"

exec "$PYTHON" \
    "$ROOT/g1_deploy/scripts/run_gated_calibration_mujoco.py" \
    --view \
    --protocol pilot \
    --output "$output" \
    "$@"
