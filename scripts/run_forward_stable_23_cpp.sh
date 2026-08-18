#!/usr/bin/env bash
# Compatibility entrypoint. Canonical preset: workflows/joystick/presets/.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/../workflows/joystick/presets/forward_stable_23_robot_cpp.sh" "$@"
