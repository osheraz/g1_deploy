#!/usr/bin/env bash
# Fixed physical entry point for the MuJoCo-selected Original Release profile.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORWARD_ARGS=()
while (( $# > 0 )); do
    case "$1" in
        --interface)
            [[ -n "${2:-}" && "$2" != --* ]] || {
                echo "ERROR: --interface requires a value" >&2
                exit 2
            }
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        --arm-real-robot|--preflight-only)
            FORWARD_ARGS+=("$1")
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--interface NAME] [--arm-real-robot|--preflight-only]"
            echo "Default: complete no-DDS preflight; add --arm-real-robot for physical motion."
            exit 0
            ;;
        *)
            echo "ERROR: unsupported fixed-profile option: $1" >&2
            exit 2
            ;;
    esac
done

# Keep the selected model, embodiment, and adaptation atomic. The delegated
# launcher defaults to a no-DDS preflight; --arm-real-robot is the only switch
# that permits it to proceed to Unitree DDS and physical commands.
exec "$SCRIPT_DIR/../workflows/joystick/robot.sh" \
    --model-profile original-release \
    --embodiment g1_23dof \
    --adaptation stable-slow-23 \
    "${FORWARD_ARGS[@]}"
