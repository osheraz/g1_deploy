#!/usr/bin/env bash
# Interactive rehearsal only: suspended 23-DoF MuJoCo, no robot/DDS channel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON="${G1_PYTHON:-$ROOT/.venv/bin/python}"

[[ -x "$PYTHON" ]] || {
    echo "ERROR: missing calibration Python: $PYTHON" >&2
    exit 2
}

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output="$ROOT/logs/sonic/calibration/interactive_mujoco/${stamp}"

exec "$PYTHON" \
    "$ROOT/g1_deploy/sonic/scripts/calibration/run_interactive_calibration_mujoco.py" \
    --view \
    --max-envelope screening \
    --output "$output" \
    "$@"
