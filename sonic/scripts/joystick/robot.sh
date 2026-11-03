#!/usr/bin/env bash
# Planner-only Python controller driven by the Unitree wireless remote.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck source=../../../sonic_cpp/teleop/core.common.sh
if [[ ! -f "$ROOT/sonic_cpp/teleop/core.common.sh" ]]; then
    echo "Missing SONIC C++ companion: $ROOT/sonic_cpp/teleop/core.common.sh" >&2
    echo "See g1_deploy/docs/sonic/joystick.md; Python entry points do not need this launcher." >&2
    exit 2
fi
source "$ROOT/sonic_cpp/teleop/core.common.sh"

INTERFACE="enp132s0"
MODEL_PROFILE="low-latency"
EMBODIMENT="g1_23dof"
ADAPTATION="official-parity"
PREFLIGHT_ONLY=0
ARM_REAL_ROBOT=0
while (( $# > 0 )); do
    case "$1" in
        --interface)
            [[ -n "${2:-}" && "$2" != --* ]] || {
                echo "ERROR: --interface requires a value" >&2
                exit 2
            }
            INTERFACE="$2"
            shift 2
            ;;
        --model-profile)
            [[ -n "${2:-}" ]] && validate_sonic_model_profile_name "$2" || {
                echo "ERROR: unsupported --model-profile: ${2:-<missing>}" >&2
                exit 2
            }
            MODEL_PROFILE="$2"
            shift 2
            ;;
        --embodiment)
            [[ -n "${2:-}" && "$2" != --* ]] || {
                echo "ERROR: --embodiment requires a value" >&2
                exit 2
            }
            EMBODIMENT="$2"
            shift 2
            ;;
        --adaptation|--adaptation-profile)
            [[ -n "${2:-}" && "$2" != --* ]] || {
                echo "ERROR: --adaptation requires a value" >&2
                exit 2
            }
            if ! ADAPTATION="$(normalize_sonic_adaptation "$2")"; then
                echo "ERROR: unsupported --adaptation: $2" >&2
                exit 2
            fi
            shift 2
            ;;
        --hip-pitch-scale|--hip-pitch-scale=*)
            echo "ERROR: --hip-pitch-scale is supported on the C++ robot launcher, not this Python path." >&2
            echo "Use $ROOT/g1_deploy/sonic/scripts/joystick/robot_cpp.sh" >&2
            exit 2
            ;;
        --preflight-only)
            PREFLIGHT_ONLY=1
            shift
            ;;
        --arm-real-robot)
            ARM_REAL_ROBOT=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--arm-real-robot] [--interface NAME] [--model-profile low-latency|original-release|sonic-v1.1] [--embodiment g1_23dof|g1_29dof] [--adaptation official-parity|forward-stable-23] [--preflight-only]"
            echo "Without --arm-real-robot, the launcher performs a no-DDS preflight and exits."
            exit 0
            ;;
        *)
            echo "ERROR: unsupported argument: $1" >&2
            exit 2
            ;;
    esac
done

if (( PREFLIGHT_ONLY && ARM_REAL_ROBOT )); then
    echo "ERROR: --preflight-only and --arm-real-robot are mutually exclusive" >&2
    exit 2
fi
if (( ! ARM_REAL_ROBOT )); then
    PREFLIGHT_ONLY=1
fi

[[ "$INTERFACE" =~ ^[A-Za-z0-9_.:-]+$ ]] || {
    echo "ERROR: invalid interface token: $INTERFACE" >&2
    exit 2
}

python_robot_network_preflight() {
    local controller_ip="192.168.123.161"
    local onboard_ip="192.168.123.164"
    local host_ip="192.168.123.222"
    local route
    local conflicts

    command -v ip >/dev/null 2>&1 || {
        echo "ERROR: iproute2 is required" >&2
        return 1
    }
    [[ -d "/sys/class/net/$INTERFACE" ]] || {
        echo "ERROR: network interface does not exist: $INTERFACE" >&2
        return 1
    }
    [[ "$(cat "/sys/class/net/$INTERFACE/operstate")" == "up" ]] || {
        echo "ERROR: $INTERFACE is not operationally up" >&2
        return 1
    }
    [[ "$(cat "/sys/class/net/$INTERFACE/carrier")" == "1" ]] || {
        echo "ERROR: $INTERFACE has no Ethernet carrier" >&2
        return 1
    }
    ip -o -4 addr show dev "$INTERFACE" | awk '{print $4}' \
        | grep -Fxq "$host_ip/24" || {
        echo "ERROR: $INTERFACE does not own $host_ip/24" >&2
        return 1
    }
    for peer in "$controller_ip" "$onboard_ip"; do
        route="$(ip -4 route get "$peer")"
        [[ "$route" == *" dev $INTERFACE "* && "$route" == *" src $host_ip "* ]] || {
            echo "ERROR: route to $peer is not pinned to $INTERFACE/$host_ip" >&2
            return 1
        }
    done
    conflicts="$(pgrep -af '[/]g1_deploy_onnx_ref( |$)|[/]sonic.deploy\.py|[/]run_sim_(loop|23dof|29dof_validation)\.py' 2>/dev/null || true)"
    [[ -z "$conflicts" ]] || {
        echo "$conflicts" >&2
        echo "ERROR: another low-level controller or G1 simulator is running" >&2
        return 1
    }
    echo "PASS: robot Ethernet carrier, address, routes, and exclusive controller ownership"
}

python_robot_network_preflight
resolved_stack="$($E_SKIN_TELEOP_PYTHON "$E_SKIN_DEPLOYMENT_MATRIX" resolve \
    --model "$MODEL_PROFILE" --operation joystick --embodiment "$EMBODIMENT" --shell)" || exit $?
eval "$resolved_stack"

case "$ADAPTATION" in
    official-parity)
        SELECTED_CONFIG="$E_SKIN_SELECTED_PYTHON_CONFIG"
        ;;
    forward-stable-23)
        if [[ "$EMBODIMENT" != "g1_23dof" ]]; then
            echo "ERROR: forward-stable-23 is defined only for --embodiment g1_23dof" >&2
            exit 2
        fi
        SELECTED_CONFIG="$E_SKIN_SELECTED_FORWARD_STABLE_23DOF_CONFIG"
        echo "WARNING: forward-stable-23 Python physical path is for A/B only; C++ is the recommended robot runtime" >&2
        ;;
    *)
        echo "ERROR: unsupported --adaptation: $ADAPTATION" >&2
        exit 2
        ;;
esac

LOG_ROOT="$ROOT/logs/sonic/python_planner_joystick"
DEPLOY_PYTHON="${G1_PYTHON:-${E_SKIN_DEPLOY_PYTHON:-$ROOT/.venv/bin/python}}"
[[ -x "$DEPLOY_PYTHON" ]] || {
    echo "ERROR: CUDA-compatible deployment Python is missing: $DEPLOY_PYTHON" >&2
    exit 1
}
export LD_LIBRARY_PATH="$TensorRT_ROOT/lib:/usr/local/cuda/lib64:/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if (( PREFLIGHT_ONLY )); then
    cd "$ROOT/g1_deploy"
    exec "$DEPLOY_PYTHON" "$ROOT/g1_deploy/sonic/deploy.py" \
        --config "$SELECTED_CONFIG" \
        --model-profile "$E_SKIN_SELECTED_MODEL_PROFILE" \
        --mode planner \
        --teleop joystick \
        --preflight-only \
        --no-camera \
        --no-tactile
fi
mkdir -p "$LOG_ROOT"
cd "$ROOT/g1_deploy"
exec "$DEPLOY_PYTHON" "$ROOT/g1_deploy/sonic/deploy.py" \
    --config "$SELECTED_CONFIG" \
    --model-profile "$E_SKIN_SELECTED_MODEL_PROFILE" \
    --mode planner \
    --teleop joystick \
    --net "$INTERFACE" \
    --out "$LOG_ROOT" \
    --auto-record \
    --no-camera \
    --no-tactile
