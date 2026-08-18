#!/usr/bin/env bash
# Compatibility entrypoint. Canonical implementation: workflows/joystick/robot_cpp.sh
# Physical C++ only. Never starts MuJoCo.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/../workflows/joystick/robot_cpp.sh" "$@"
