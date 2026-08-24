#!/usr/bin/env bash
# Interactive rehearsal only: suspended 23-DoF MuJoCo, no robot/DDS channel.
set -euo pipefail

ROOT="/home/clhsieh/e_skin"
PYTHON="/data/clhsieh/environments/python/venvs/e-skin-4090-verify/bin/python"

[[ -x "$PYTHON" ]] || {
    echo "ERROR: missing calibration Python: $PYTHON" >&2
    exit 2
}

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output="/data/clhsieh/logs/e_skin/calibration/interactive_mujoco/${stamp}"

exec "$PYTHON" \
    "$ROOT/g1_deploy/sonic/scripts/run_interactive_calibration_mujoco.py" \
    --view \
    --max-envelope screening \
    --output "$output" \
    "$@"
