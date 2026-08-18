#!/usr/bin/env bash
# Compatibility entrypoint for the Python robust physical/preflight preset.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/../workflows/joystick/presets/robust_small_gait_23_robot_python.sh" "$@"
