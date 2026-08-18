#!/usr/bin/env bash
# Compatibility entrypoint. Canonical preset: workflows/joystick/presets/.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/../workflows/joystick/presets/robust_small_gait_23_robot_cpp.sh" "$@"
