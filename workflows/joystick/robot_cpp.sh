#!/usr/bin/env bash
# Unified C++ planner-joystick launcher for the physical G1.
# This launcher never starts MuJoCo. Use mujoco.sh for simulation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=../../../sonic_cpp/teleop/common.sh
source "$ROOT/sonic_cpp/teleop/common.sh"

INTERFACE="enp132s0"
MODEL_PROFILE="original-release"
EMBODIMENT="g1_23dof"
ADAPTATION="official-parity"
HIP_PITCH_SCALE=""
SAGITTAL_SCALE=""
HIP_PITCH_SCALE_DEFAULT="0.85"
KNEE_GAIT_OFFSET_DEG=""
YAW_COMMAND_SCALE=""
YAW_INPLACE_SCALE=""
SLOW_GAIT=0
ARM_REAL_ROBOT=0
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: $0 [--arm-real-robot] [--interface NAME]
          [--model-profile low-latency|original-release|sonic-v1.1]
          [--embodiment g1_23dof|g1_29dof]
          [--adaptation official-parity|forward-stable-23|robust-small-gait-23]
          [--hip-pitch-scale[=SCALE]|--sagittal-scale[=SCALE]|--slow-gait]
          [--knee-gait-offset-deg DEGREES]
          [--yaw-command-scale SCALE] [--yaw-inplace-scale SCALE]
          [--probe-hosts|--clean-ports|--clean-team-ports]

Unified C++ physical joystick launcher. It never starts MuJoCo. Defaults
are Original Release, g1_23dof, and official-parity (naive 29-to-23
contract). Without --arm-real-robot this is a no-DDS preflight only.
Use run_planner_joystick_mujoco.sh for simulation.

Adaptations:
  official-parity, naive
      Official 29-to-23 contract: zero missing measured q/dq, native
      reference and action history, direct decoder/motor output.
  forward-stable-23, stable
      All-six missing-axis reference/history masks and direct output.
      Requires g1_23dof. Valid with any released model profile.
  robust-small-gait-23
      Historical backward-selected wrist-mask/hip-scale contract.

--hip-pitch-scale, --sagittal-scale, and --slow-gait are valid only with
forward-stable-23 on g1_23dof and are mutually exclusive. No value on a
scale flag selects $HIP_PITCH_SCALE_DEFAULT. Omitting every scale flag
keeps encoder-facing gait amplitude at 1.00.

--sagittal-scale shortens hip pitch, knee, and ankle pitch together
about the captured stand. Prefer it over hip-only for a first physical
trial (bare = 0.85). --slow-gait is the healthy-MuJoCo Original Release
floor: sagittal 0.28 + gait-only knee +8 deg + walk yaw 0.15 / in-place
0.35. Official Slow Walk stays 0.4. Do not start a first physical trial
at 0.28 unless the plant is known-healthy.
EOF
}

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
                echo "ERROR: --adaptation requires official-parity, forward-stable-23, or robust-small-gait-23" >&2
                exit 2
            }
            if ! ADAPTATION="$(normalize_sonic_adaptation "$2")"; then
                echo "ERROR: unsupported --adaptation: $2" >&2
                usage >&2
                exit 2
            fi
            shift 2
            ;;
        --hip-pitch-scale)
            if [[ -n "${2:-}" && "${2:-}" != --* ]]; then
                HIP_PITCH_SCALE="$2"
                shift 2
            else
                HIP_PITCH_SCALE="$HIP_PITCH_SCALE_DEFAULT"
                shift
            fi
            ;;
        --hip-pitch-scale=*)
            HIP_PITCH_SCALE="${1#*=}"
            [[ -n "$HIP_PITCH_SCALE" ]] || {
                echo "ERROR: --hip-pitch-scale= requires a value" >&2
                exit 2
            }
            shift
            ;;
        --sagittal-scale)
            if [[ -n "${2:-}" && "${2:-}" != --* ]]; then
                SAGITTAL_SCALE="$2"
                shift 2
            else
                SAGITTAL_SCALE="$HIP_PITCH_SCALE_DEFAULT"
                shift
            fi
            ;;
        --sagittal-scale=*)
            SAGITTAL_SCALE="${1#*=}"
            [[ -n "$SAGITTAL_SCALE" ]] || {
                echo "ERROR: --sagittal-scale= requires a value" >&2
                exit 2
            }
            shift
            ;;
        --knee-gait-offset-deg)
            [[ -n "${2:-}" && "${2:-}" != --* ]] || {
                echo "ERROR: --knee-gait-offset-deg requires DEGREES" >&2
                exit 2
            }
            KNEE_GAIT_OFFSET_DEG="$2"
            shift 2
            ;;
        --yaw-command-scale)
            [[ -n "${2:-}" && "${2:-}" != --* ]] || {
                echo "ERROR: --yaw-command-scale requires a value" >&2
                exit 2
            }
            YAW_COMMAND_SCALE="$2"
            shift 2
            ;;
        --yaw-inplace-scale)
            [[ -n "${2:-}" && "${2:-}" != --* ]] || {
                echo "ERROR: --yaw-inplace-scale requires a value" >&2
                exit 2
            }
            YAW_INPLACE_SCALE="$2"
            shift 2
            ;;
        --slow-gait)
            SLOW_GAIT=1
            shift
            ;;
        --arm-real-robot)
            ARM_REAL_ROBOT=1
            shift
            ;;
        --command-protocol|--unitree-remote|--ground-friction|--joint-encoder-offset-deg|--plant-gain-scale|--plant-target-delay-ms|--planner-replan-period|--idle-hold|--no-idle-hold|--record-root|--no-record)
            echo "ERROR: $1 is a MuJoCo-only flag. This is the physical C++ robot launcher." >&2
            echo "Use /home/clhsieh/e_skin/g1_deploy/scripts/run_planner_joystick_mujoco.sh for simulation." >&2
            exit 2
            ;;
        --probe-hosts|--clean-ports|--clean-team-ports)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unsupported argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ "$INTERFACE" =~ ^[A-Za-z0-9_.:-]+$ ]] || {
    echo "ERROR: invalid interface token: $INTERFACE" >&2
    exit 2
}

if [[ "$ADAPTATION" == "forward-stable-23" && "$EMBODIMENT" != "g1_23dof" ]]; then
    echo "ERROR: forward-stable-23 is defined only for --embodiment g1_23dof" >&2
    exit 2
fi
if [[ "$ADAPTATION" == "robust-small-gait-23" && "$EMBODIMENT" != "g1_23dof" ]]; then
    echo "ERROR: robust-small-gait-23 requires --embodiment g1_23dof" >&2
    exit 2
fi
SLOW_KNEE_OVERRIDE="$KNEE_GAIT_OFFSET_DEG"
SLOW_YAW_COMMAND_OVERRIDE="$YAW_COMMAND_SCALE"
SLOW_YAW_INPLACE_OVERRIDE="$YAW_INPLACE_SCALE"
if (( SLOW_GAIT )); then
    if [[ -n "$HIP_PITCH_SCALE" || -n "$SAGITTAL_SCALE" ]]; then
        echo "ERROR: --slow-gait is mutually exclusive with --hip-pitch-scale and --sagittal-scale" >&2
        exit 2
    fi
    SAGITTAL_SCALE="0.28"
    KNEE_GAIT_OFFSET_DEG="${KNEE_GAIT_OFFSET_DEG:-8}"
    YAW_COMMAND_SCALE="${YAW_COMMAND_SCALE:-0.15}"
    YAW_INPLACE_SCALE="${YAW_INPLACE_SCALE:-0.35}"
fi
if [[ -n "$HIP_PITCH_SCALE" && -n "$SAGITTAL_SCALE" ]]; then
    echo "ERROR: --hip-pitch-scale and --sagittal-scale are mutually exclusive" >&2
    exit 2
fi
if [[ -n "$HIP_PITCH_SCALE$SAGITTAL_SCALE$KNEE_GAIT_OFFSET_DEG$YAW_COMMAND_SCALE$YAW_INPLACE_SCALE" &&
      "$ADAPTATION" != "forward-stable-23" ]]; then
    echo "ERROR: gait scale / knee offset / yaw flags require --adaptation forward-stable-23" >&2
    exit 2
fi

case "$ADAPTATION" in
    official-parity)
        echo "[ROBOT STACK] C++ / $MODEL_PROFILE / official-parity / joystick / $EMBODIMENT"
        echo "[ROBOT STACK] Naive 29-to-23 contract: missing measured q/dq zero; native reference and action history; direct output."
        ;;
    forward-stable-23)
        echo "[ROBOT STACK] C++ / $MODEL_PROFILE / forward-stable-23 / joystick / $EMBODIMENT"
        if [[ -n "$SAGITTAL_SCALE" ]]; then
            echo "[ROBOT STACK] All-six reference/history masks; active-gait sagittal scale $SAGITTAL_SCALE; direct decoder targets."
        elif [[ -n "$HIP_PITCH_SCALE" ]]; then
            echo "[ROBOT STACK] All-six reference/history masks; active-gait hip-pitch scale $HIP_PITCH_SCALE; direct decoder targets."
        else
            echo "[ROBOT STACK] All-six reference/history masks; unscaled sagittal reference; direct decoder targets."
        fi
        if [[ -n "$KNEE_GAIT_OFFSET_DEG" ]]; then
            echo "[ROBOT STACK] Gait-only knee flexion offset: ${KNEE_GAIT_OFFSET_DEG} deg (inactive in IDLE)."
        fi
        if [[ -n "$YAW_COMMAND_SCALE$YAW_INPLACE_SCALE" ]]; then
            echo "[ROBOT STACK] Split yaw: walk ${YAW_COMMAND_SCALE:-1} / in-place ${YAW_INPLACE_SCALE:-$YAW_COMMAND_SCALE}."
        fi
        echo "[ROBOT STACK] Encoder copy only. Official Slow Walk stays 0.4."
        echo "[ROBOT STACK] MuJoCo-qualified recipe; first physical use is an experimental trial."
        ;;
    robust-small-gait-23)
        echo "[ROBOT STACK] C++ / $MODEL_PROFILE / robust-small-gait-23 / joystick / $EMBODIMENT"
        echo "[ROBOT STACK] Historical backward-selected wrist-mask contract; decoder targets remain direct."
        ;;
esac
echo "[ROBOT STACK] Planner recurrence, NVIDIA gains, and physical motor targets remain direct."

LAUNCHER="$ROOT/sonic_cpp/teleop/t1_planner_joystick_robot_23dof.sh"
ARGS=(
    --model-profile "$MODEL_PROFILE"
    --embodiment "$EMBODIMENT"
    --adaptation-profile "$ADAPTATION"
    --interface "$INTERFACE"
    "${EXTRA_ARGS[@]}"
)
if (( SLOW_GAIT )); then
    ARGS+=(--slow-gait)
    if [[ -n "$SLOW_KNEE_OVERRIDE" ]]; then
        ARGS+=(--knee-gait-offset-deg "$SLOW_KNEE_OVERRIDE")
    fi
    if [[ -n "$SLOW_YAW_COMMAND_OVERRIDE" ]]; then
        ARGS+=(--yaw-command-scale "$SLOW_YAW_COMMAND_OVERRIDE")
    fi
    if [[ -n "$SLOW_YAW_INPLACE_OVERRIDE" ]]; then
        ARGS+=(--yaw-inplace-scale "$SLOW_YAW_INPLACE_OVERRIDE")
    fi
else
    if [[ -n "$SAGITTAL_SCALE" ]]; then
        ARGS+=(--sagittal-scale "$SAGITTAL_SCALE")
    fi
    if [[ -n "$HIP_PITCH_SCALE" ]]; then
        ARGS+=(--hip-pitch-scale "$HIP_PITCH_SCALE")
    fi
    if [[ -n "$KNEE_GAIT_OFFSET_DEG" ]]; then
        ARGS+=(--knee-gait-offset-deg "$KNEE_GAIT_OFFSET_DEG")
    fi
    if [[ -n "$YAW_COMMAND_SCALE" ]]; then
        ARGS+=(--yaw-command-scale "$YAW_COMMAND_SCALE")
    fi
    if [[ -n "$YAW_INPLACE_SCALE" ]]; then
        ARGS+=(--yaw-inplace-scale "$YAW_INPLACE_SCALE")
    fi
fi
if (( ARM_REAL_ROBOT )); then
    ARGS+=(--arm-real-robot --non-interactive)
fi
exec env -u VIRTUAL_ENV -u CONDA_PREFIX "$LAUNCHER" "${ARGS[@]}"
