#!/usr/bin/env bash
# Review-gated, two-shoulder suspended 23-DoF MuJoCo calibration rehearsal.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON="${G1_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || { echo "ERROR: missing Python: $PYTHON" >&2; exit 2; }

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output="$ROOT/logs/sonic/calibration/review_gated_mujoco/${stamp}"

exec "$PYTHON" \
    "$ROOT/g1_deploy/sonic/scripts/calibration/run_gated_calibration_mujoco.py" \
    --view \
    --protocol pilot \
    --output "$output" \
    "$@"
