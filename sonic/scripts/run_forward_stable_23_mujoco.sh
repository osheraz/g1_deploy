#!/usr/bin/env bash
# Compatibility entrypoint for the corrected-forward MuJoCo preset.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/../workflows/joystick/presets/forward_stable_23_mujoco.sh" "$@"
