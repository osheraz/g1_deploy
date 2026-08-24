#!/usr/bin/env bash
# Headset-free Python SONIC planner test in local 23- or 29-DoF MuJoCo.
# This launcher never publishes a physical LowCmd. Use robot_cpp.sh for hardware.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck source=../../../sonic_cpp/teleop/core.common.sh
source "$ROOT/sonic_cpp/teleop/core.common.sh"

MODEL_PROFILE="low-latency"
EMBODIMENT="g1_23dof"
RECORD_ROOT="/data/$(id -un)/logs/e_skin/mujoco_joystick"
RECORD=1
COMMAND_PROTOCOL="keyboard"
ROBOT_INTERFACE="enp132s0"
GROUND_FRICTION=""
JOINT_ENCODER_OFFSETS=()
PLANT_FAULT_ARGS=()
PLANNER_REPLAN_PERIOD=""
IDLE_HOLD=1
ADAPTATION="official-parity"
HIP_PITCH_SCALE=""
SAGITTAL_SCALE=""
HIP_PITCH_SCALE_DEFAULT="0.85"
KNEE_GAIT_OFFSET_DEG=""
YAW_COMMAND_SCALE=""
YAW_INPLACE_SCALE=""
SLOW_GAIT=0
LANDING_HOLD=0
LANDING_HOLD_DVEL=0
LEGACY_REPRODUCTION=0
VIEW_MODE=""
while (( $# > 0 )); do
    case "$1" in
        --model-profile)
            [[ -n "${2:-}" ]] && validate_sonic_model_profile_name "$2" || {
                echo "ERROR: unsupported --model-profile: ${2:-<missing>}" >&2
                exit 2
            }
            MODEL_PROFILE="$2"
            shift 2
            ;;
        --embodiment)
            [[ -n "${2:-}" ]] || { echo "ERROR: --embodiment requires a value" >&2; exit 2; }
            EMBODIMENT="$2"
            shift 2
            ;;
        --record-root)
            [[ -n "${2:-}" ]] || { echo "ERROR: --record-root requires a value" >&2; exit 2; }
            RECORD_ROOT="$2"
            shift 2
            ;;
        --no-record)
            RECORD=0
            shift
            ;;
        --unitree-remote)
            # Backward-compatible spelling for the explicit protocol selector.
            COMMAND_PROTOCOL="unitree-remote"
            shift
            ;;
        --command-protocol)
            case "${2:-}" in
                keyboard|unitree-remote) COMMAND_PROTOCOL="$2" ;;
                *)
                    echo "ERROR: --command-protocol must be keyboard or unitree-remote" >&2
                    exit 2
                    ;;
            esac
            shift 2
            ;;
        --interface)
            [[ -n "${2:-}" ]] || { echo "ERROR: --interface requires a value" >&2; exit 2; }
            ROBOT_INTERFACE="$2"
            shift 2
            ;;
        --ground-friction)
            [[ -n "${2:-}" ]] || { echo "ERROR: --ground-friction requires a value" >&2; exit 2; }
            GROUND_FRICTION="$2"
            shift 2
            ;;
        --joint-encoder-offset-deg)
            [[ -n "${2:-}" ]] || {
                echo "ERROR: --joint-encoder-offset-deg requires JOINT=DEGREES" >&2
                exit 2
            }
            JOINT_ENCODER_OFFSETS+=(--joint-encoder-offset-deg "$2")
            shift 2
            ;;
        --plant-gain-scale)
            [[ -n "${2:-}" ]] || {
                echo "ERROR: --plant-gain-scale requires JOINT_OR_GROUP=SCALE" >&2
                exit 2
            }
            PLANT_FAULT_ARGS+=(--plant-gain-scale "$2")
            shift 2
            ;;
        --plant-target-delay-ms)
            [[ -n "${2:-}" ]] || {
                echo "ERROR: --plant-target-delay-ms requires JOINT_OR_GROUP=MILLISECONDS" >&2
                exit 2
            }
            PLANT_FAULT_ARGS+=(--plant-target-delay-ms "$2")
            shift 2
            ;;
        --planner-replan-period)
            [[ -n "${2:-}" ]] || { echo "ERROR: --planner-replan-period requires seconds" >&2; exit 2; }
            PLANNER_REPLAN_PERIOD="$2"
            shift 2
            ;;
        --viser)
            # Browser viewer for SSH sessions: no GLFW window, drive buttons
            # and the robot mirror on http://<host>:8080.
            VIEW_MODE="viser"
            shift
            ;;
        --idle-hold)
            IDLE_HOLD=1
            shift
            ;;
        --no-idle-hold)
            IDLE_HOLD=0
            shift
            ;;
        --adaptation|--adaptation-profile)
            [[ -n "${2:-}" ]] || { echo "ERROR: --adaptation requires a value" >&2; exit 2; }
            if ! ADAPTATION="$(normalize_sonic_adaptation "$2")"; then
                echo "ERROR: unsupported --adaptation: $2" >&2
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
            [[ -n "${2:-}" ]] || {
                echo "ERROR: --knee-gait-offset-deg requires DEGREES" >&2
                exit 2
            }
            KNEE_GAIT_OFFSET_DEG="$2"
            shift 2
            ;;
        --yaw-command-scale)
            [[ -n "${2:-}" ]] || {
                echo "ERROR: --yaw-command-scale requires a value" >&2
                exit 2
            }
            YAW_COMMAND_SCALE="$2"
            shift 2
            ;;
        --yaw-inplace-scale)
            [[ -n "${2:-}" ]] || {
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
        --landing-hold)
            LANDING_HOLD=1
            shift
            ;;
        --landing-hold-dvel)
            LANDING_HOLD_DVEL=1
            shift
            ;;
        --legacy-reproduction)
            LEGACY_REPRODUCTION=1
            shift
            ;;
        --arm-real-robot|--non-interactive|--probe-hosts|--clean-ports|--clean-team-ports)
            echo "ERROR: $1 is a physical-robot flag. This is the MuJoCo launcher and never sends a LowCmd." >&2
            echo "Use /home/clhsieh/e_skin/g1_deploy/scripts/run_planner_joystick_robot_cpp.sh for the real robot." >&2
            exit 2
            ;;
        -h|--help)
            echo "Usage: $0 [--model-profile low-latency|original-release|sonic-v1.1] [--embodiment g1_23dof|g1_29dof] [--command-protocol keyboard|unitree-remote] [--adaptation official-parity|forward-stable-23] [--hip-pitch-scale[=SCALE]|--sagittal-scale[=SCALE]|--slow-gait] [--viser] [--knee-gait-offset-deg DEGREES] [--yaw-command-scale SCALE] [--yaw-inplace-scale SCALE] [--interface IFACE] [--ground-friction MU] [--joint-encoder-offset-deg JOINT=DEGREES] [--plant-gain-scale JOINT_OR_GROUP=SCALE] [--plant-target-delay-ms JOINT_OR_GROUP=MILLISECONDS] [--planner-replan-period SECONDS] [--idle-hold|--no-idle-hold] [--record-root DIR] [--no-record]"
            echo "This is the MuJoCo launcher only. It never publishes a physical LowCmd."
            echo "--adaptation naive=official-parity; stable=forward-stable-23."
            echo "--hip-pitch-scale, --sagittal-scale, and --slow-gait are valid only with forward-stable-23 on g1_23dof and are mutually exclusive."
            echo "--slow-gait is sagittal 0.28 + knee +8 deg + walk yaw 0.15 / in-place yaw 0.35."
            echo "robust-small-gait-23, stable-slow-23, --landing-hold, and --landing-hold-dvel are withdrawn legacy attempts. Blocked unless --legacy-reproduction."
            echo "--viser opens a browser viewer on :8080 instead of the GLFW window (SSH)."
            exit 0
            ;;
        *)
            echo "ERROR: unsupported argument: $1" >&2
            exit 2
            ;;
    esac
done

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
        echo "Adaptation: forward-stable-23 (all-six reference/history masks; direct reference and motor output)"
        ;;
    stable-slow-23)
        if (( ! LEGACY_REPRODUCTION )); then
            legacy_sonic_attempt_error "stable-slow-23"
            echo "MuJoCo reproduction only: add --legacy-reproduction." >&2
            exit 2
        fi
        if [[ "$MODEL_PROFILE" != "original-release" || "$EMBODIMENT" != "g1_23dof" ]]; then
            echo "ERROR: stable-slow-23 requires --model-profile original-release --embodiment g1_23dof" >&2
            exit 2
        fi
        SELECTED_CONFIG="$ROOT/g1_deploy/configs/adaptations/original_release_23dof_stable_slow.yaml"
        echo "LEGACY reproduction: stable-slow-23 (withdrawn ankle-shaping profile)"
        ;;
    robust-small-gait-23)
        if (( ! LEGACY_REPRODUCTION )); then
            legacy_sonic_attempt_error "robust-small-gait-23"
            echo "MuJoCo reproduction only: add --legacy-reproduction." >&2
            exit 2
        fi
        if [[ "$EMBODIMENT" != "g1_23dof" ]]; then
            echo "ERROR: robust-small-gait-23 requires --embodiment g1_23dof" >&2
            exit 2
        fi
        SELECTED_CONFIG="$E_SKIN_SELECTED_ROBUST_23DOF_CONFIG"
        echo "LEGACY reproduction: robust-small-gait-23 (withdrawn backward-selected wrist-mask profile)"
        ;;
    *)
        echo "ERROR: unsupported --adaptation: $ADAPTATION" >&2
        exit 2
        ;;
esac

if (( SLOW_GAIT )); then
    if [[ -n "$HIP_PITCH_SCALE" || -n "$SAGITTAL_SCALE" ]]; then
        echo "ERROR: --slow-gait is mutually exclusive with --hip-pitch-scale and --sagittal-scale" >&2
        exit 2
    fi
    if [[ "$ADAPTATION" != "forward-stable-23" ]]; then
        echo "ERROR: --slow-gait is supported only on top of --adaptation forward-stable-23" >&2
        exit 2
    fi
fi
if [[ -n "$HIP_PITCH_SCALE" && -n "$SAGITTAL_SCALE" ]]; then
    echo "ERROR: --hip-pitch-scale and --sagittal-scale are mutually exclusive" >&2
    exit 2
fi
if [[ -n "$HIP_PITCH_SCALE" && "$ADAPTATION" != "forward-stable-23" ]]; then
    echo "ERROR: --hip-pitch-scale is supported only on top of --adaptation forward-stable-23" >&2
    exit 2
fi
if [[ -n "$SAGITTAL_SCALE" && "$ADAPTATION" != "forward-stable-23" ]]; then
    echo "ERROR: --sagittal-scale is supported only on top of --adaptation forward-stable-23" >&2
    exit 2
fi
if [[ -n "$KNEE_GAIT_OFFSET_DEG$YAW_COMMAND_SCALE$YAW_INPLACE_SCALE" && "$ADAPTATION" != "forward-stable-23" ]]; then
    echo "ERROR: --knee-gait-offset-deg / yaw scales require --adaptation forward-stable-23" >&2
    exit 2
fi
if (( LANDING_HOLD + LANDING_HOLD_DVEL > 1 )); then
    echo "ERROR: --landing-hold and --landing-hold-dvel are mutually exclusive" >&2
    exit 2
fi
if (( LANDING_HOLD + LANDING_HOLD_DVEL > 0 )) && (( ! LEGACY_REPRODUCTION )); then
    legacy_sonic_attempt_error "--landing-hold / --landing-hold-dvel"
    echo "MuJoCo reproduction only: add --legacy-reproduction." >&2
    exit 2
fi
if (( LANDING_HOLD + LANDING_HOLD_DVEL > 0 )) && [[ "$ADAPTATION" != "forward-stable-23" ]]; then
    echo "ERROR: --landing-hold / --landing-hold-dvel require --adaptation forward-stable-23" >&2
    exit 2
fi

DEPLOY_PYTHON="${E_SKIN_DEPLOY_PYTHON:-/data/$(id -un)/environments/python/venvs/e-skin-4090-verify/bin/python}"
[[ -x "$DEPLOY_PYTHON" ]] || {
    echo "ERROR: CUDA-compatible deployment Python is missing: $DEPLOY_PYTHON" >&2
    exit 1
}
export LD_LIBRARY_PATH="$TensorRT_ROOT/lib:/usr/local/cuda/lib64:/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$ROOT/g1_deploy"
record_args=()
if (( RECORD )); then
    session_id="$(date -u +%Y%m%dT%H%M%SZ)-$(hostname -s)-$$-${EMBODIMENT}-${MODEL_PROFILE}"
    record_dir="$RECORD_ROOT/$session_id"
    mkdir -p "$record_dir"
    record_args=(--record-dir "$record_dir")
    echo "MuJoCo recording: $record_dir"
fi
remote_args=()
if [[ "$COMMAND_PROTOCOL" == "unitree-remote" ]]; then
    remote_args=(--remote --interface "$ROBOT_INTERFACE")
else
    # The keyboard is only an input device; command semantics must still match
    # NVIDIA's native gamepad_manager. In particular SLOW_WALK is fixed at
    # 0.4 m/s and RUN at 1.5 m/s rather than analog-speed extrapolations.
    remote_args=(--native-gamepad)
    echo "Keyboard planner semantics: native gamepad_manager (fixed gait speeds)"
fi
friction_args=()
if [[ -n "$GROUND_FRICTION" ]]; then
    friction_args=(--ground-friction "$GROUND_FRICTION")
fi
replan_args=()
if [[ -n "$PLANNER_REPLAN_PERIOD" ]]; then
    replan_args=(--planner-replan-period "$PLANNER_REPLAN_PERIOD")
fi
reference_args=()
if (( SLOW_GAIT )); then
    reference_args=(--slow-gait)
    echo "Slow-gait recipe: sagittal 0.28, gait-only knee +8 deg, walk yaw 0.15 / in-place 0.35"
elif [[ -n "$SAGITTAL_SCALE" ]]; then
    reference_args=(--reference-joint-scale "sagittal=$SAGITTAL_SCALE")
    echo "Forward-stable sagittal reference scale (hip-pitch/knee/ankle-pitch): $SAGITTAL_SCALE"
    echo "Command yaw rate is scaled by ${YAW_COMMAND_SCALE:-$SAGITTAL_SCALE} so walk-plus-turn stays a slower heading change."
elif [[ -n "$HIP_PITCH_SCALE" ]]; then
    reference_args=(--reference-joint-scale "hip_pitch=$HIP_PITCH_SCALE")
    echo "Forward-stable bilateral hip-pitch reference scale: $HIP_PITCH_SCALE"
    echo "Command yaw rate is scaled by $HIP_PITCH_SCALE so walk-plus-turn stays a slower heading change."
fi
if [[ -n "$KNEE_GAIT_OFFSET_DEG" ]]; then
    reference_args+=(--reference-joint-offset-deg "knee=$KNEE_GAIT_OFFSET_DEG")
    echo "Gait-only knee flexion offset: ${KNEE_GAIT_OFFSET_DEG} deg (inactive in IDLE)"
fi
if [[ -n "$YAW_COMMAND_SCALE" ]]; then
    reference_args+=(--yaw-command-scale "$YAW_COMMAND_SCALE")
fi
if [[ -n "$YAW_INPLACE_SCALE" ]]; then
    reference_args+=(--yaw-inplace-scale "$YAW_INPLACE_SCALE")
    echo "In-place turn yaw scale: $YAW_INPLACE_SCALE (walk-plus-turn keeps ${YAW_COMMAND_SCALE:-derived})"
fi
if (( LEGACY_REPRODUCTION )); then
    reference_args+=(--legacy-reproduction)
fi
if (( LANDING_HOLD )); then
    reference_args+=(--landing-hold)
    echo "LEGACY landing hold (pose freeze). Withdrawn; destroys swing. Prefer --slow-gait alone."
fi
if (( LANDING_HOLD_DVEL )); then
    reference_args+=(--landing-hold-dvel)
    echo "LEGACY landing hold (dvel-only). Withdrawn; does not improve --slow-gait."
fi
idle_args=()
if (( ! IDLE_HOLD )); then
    idle_args=(--no-idle-hold)
    echo "IDLE reference: learned planner trajectory (local IDLE hold disabled)"
else
    echo "IDLE reference: stable-stance hold (default)"
fi
exec "$DEPLOY_PYTHON" "$ROOT/g1_deploy/sonic/scripts/sim2sim_sonic.py" \
    --config "$SELECTED_CONFIG" \
    --model-profile "$MODEL_PROFILE" \
    --mjcf "$E_SKIN_SELECTED_MJCF" \
    --pico joystick \
    --planner \
    "--${VIEW_MODE:-view}" \
    --realtime \
    "${remote_args[@]}" \
    "${friction_args[@]}" \
    "${JOINT_ENCODER_OFFSETS[@]}" \
    "${PLANT_FAULT_ARGS[@]}" \
    "${reference_args[@]}" \
    "${replan_args[@]}" \
    "${idle_args[@]}" \
    "${record_args[@]}"
