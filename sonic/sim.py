"""Closed-loop sim2sim gate for the SONIC tracker in MuJoCo.

Runs the real deploy objects -- SonicRunner and MotionReferenceWindow, with the
gains, default pose and per-joint action_scale from the config -- against a
different physics engine than the one they were trained in. A pass here is the
prerequisite for touching hardware; a fall here means do not.

  python g1_deploy/sonic/sim.py \
      --mjcf ~/GR00T-WholeBodyControl/gear_sonic_deploy/g1/scene_23dof.xml

Reports height, tilt and per-joint tracking error against the clip.
"""
import argparse
from collections import deque
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import mujoco
from mujoco.glfw import glfw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import Config
from sonic.sonic_reference import MotionReferenceWindow
from sonic.model.robot_profile import MUJOCO_JOINT_NAMES, get_robot_profile
from sonic.inputs.scripted_teleop import ScriptedTeleop
from core.robot.unitree_dds import configure_process_local_cyclonedds_log
from sonic.sonic_runner import SonicRunner
from sonic.model.deployment_profile import model_profile_names


# MuJoCo's Simulate viewer owns the ordinary digits, most letter keys,
# left/right arrows, and square brackets. Keep policy control entirely on the
# numeric keypad, whose GLFW key codes are distinct from those viewer bindings.
_KEYPAD_TELEOP_KEYS = frozenset(
    {
        glfw.KEY_KP_0,
        glfw.KEY_KP_1,
        glfw.KEY_KP_2,
        glfw.KEY_KP_3,
        glfw.KEY_KP_4,
        glfw.KEY_KP_5,
        glfw.KEY_KP_6,
        glfw.KEY_KP_7,
        glfw.KEY_KP_8,
        glfw.KEY_KP_9,
        glfw.KEY_KP_ENTER,
    }
)
_MUJOCO_DOCUMENTED_VIEWER_KEYS = frozenset(
    {
        glfw.KEY_SPACE,
        glfw.KEY_EQUAL,
        glfw.KEY_MINUS,
        glfw.KEY_BACKSPACE,
        glfw.KEY_ESCAPE,
        glfw.KEY_LEFT,
        glfw.KEY_RIGHT,
        glfw.KEY_LEFT_BRACKET,
        glfw.KEY_RIGHT_BRACKET,
        glfw.KEY_PAGE_UP,
        glfw.KEY_TAB,
        *range(glfw.KEY_F1, glfw.KEY_F7 + 1),
        *range(glfw.KEY_0, glfw.KEY_5 + 1),
        *(getattr(glfw, f"KEY_{letter}") for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        glfw.KEY_COMMA,
        glfw.KEY_APOSTROPHE,
        glfw.KEY_SEMICOLON,
        glfw.KEY_GRAVE_ACCENT,
        glfw.KEY_BACKSLASH,
        glfw.KEY_SLASH,
    }
)
if not _KEYPAD_TELEOP_KEYS.isdisjoint(_MUJOCO_DOCUMENTED_VIEWER_KEYS):
    raise RuntimeError("SONIC keypad bindings overlap MuJoCo viewer shortcuts")


class _MujocoJointState:
    """Expose the simulator's current physical joints to IDLE-hold capture."""

    def __init__(
        self,
        data,
        qpos_addresses,
        qvel_addresses,
        encoder_position_offset=None,
        support_diagnostics=None,
    ):
        self._data = data
        self._qpos_addresses = np.asarray(qpos_addresses, dtype=np.int64)
        self._qvel_addresses = np.asarray(qvel_addresses, dtype=np.int64)
        self._encoder_position_offset = np.asarray(
            np.zeros(len(self._qpos_addresses), dtype=np.float64)
            if encoder_position_offset is None
            else encoder_position_offset,
            dtype=np.float64,
        )
        if self._encoder_position_offset.shape != self._qpos_addresses.shape:
            raise ValueError("encoder-position offset must match physical joints")
        self._support_diagnostics = support_diagnostics

    @property
    def qj(self) -> np.ndarray:
        true_q = np.asarray(
            self._data.qpos[self._qpos_addresses], np.float64
        ).copy()
        return true_q + self._encoder_position_offset

    @property
    def dqj(self) -> np.ndarray:
        return np.asarray(self._data.qvel[self._qvel_addresses], np.float64).copy()

    @property
    def base_quat_wxyz(self) -> np.ndarray:
        return np.asarray(self._data.qpos[3:7], np.float64).copy()

    @property
    def root_z(self) -> float:
        return float(self._data.qpos[2])

    def stand_support_diagnostics(self):
        if self._support_diagnostics is None:
            return None
        return self._support_diagnostics()


def _convex_hull_xy(points: np.ndarray) -> np.ndarray:
    """Return the counter-clockwise 2-D convex hull without SciPy."""
    unique = sorted(set(map(tuple, np.asarray(points, np.float64).reshape(-1, 2))))
    if len(unique) <= 1:
        return np.asarray(unique, np.float64)

    def cross(origin, a, b):
        return ((a[0] - origin[0]) * (b[1] - origin[1])
                - (a[1] - origin[1]) * (b[0] - origin[0]))

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], np.float64)


def _support_margin_xy(points: np.ndarray, point: np.ndarray) -> float:
    """Signed distance from a point to a CCW convex support polygon."""
    hull = _convex_hull_xy(points)
    if len(hull) < 3:
        return float("-inf")
    edge = np.roll(hull, -1, axis=0) - hull
    relative = np.asarray(point, np.float64).reshape(1, 2) - hull
    length = np.linalg.norm(edge, axis=1)
    valid = length > 1e-9
    if np.count_nonzero(valid) < 3:
        return float("-inf")
    signed = (
        edge[valid, 0] * relative[valid, 1]
        - edge[valid, 1] * relative[valid, 0]
    ) / length[valid]
    return float(np.min(signed))


def _configure_process_local_dds_log(channel_module) -> Path:
    """Compatibility wrapper for the shared Unitree DDS configuration."""
    return configure_process_local_cyclonedds_log(channel_module)


def _parse_joint_encoder_offsets_deg(
    specs: list[str], joint_names: list[str]
) -> np.ndarray:
    """Parse repeatable JOINT=DEGREES measurement offsets in physical order."""
    offsets = np.zeros(len(joint_names), dtype=np.float64)
    aliases = {}
    for index, name in enumerate(joint_names):
        aliases[name] = index
        aliases[name.removesuffix("_joint")] = index
    seen = set()
    for spec in specs:
        try:
            joint, value_text = spec.split("=", 1)
            degrees = float(value_text)
        except ValueError as exc:
            raise SystemExit(
                f"--joint-encoder-offset-deg expects JOINT=DEGREES, got {spec!r}"
            ) from exc
        joint = joint.strip()
        if joint not in aliases:
            raise SystemExit(
                f"unknown encoder-offset joint {joint!r}; choose "
                + ", ".join(name.removesuffix("_joint") for name in joint_names)
            )
        index = aliases[joint]
        if index in seen:
            raise SystemExit(f"encoder offset for {joint!r} was specified twice")
        if not np.isfinite(degrees) or abs(degrees) > 90.0:
            raise SystemExit(
                "--joint-encoder-offset-deg must be finite and within [-90, 90]"
            )
        offsets[index] = np.deg2rad(degrees)
        seen.add(index)
    return offsets

def gravity_orientation(q_wxyz):
    qw, qx, qy, qz = q_wxyz
    return np.array([2 * (-qz * qx + qw * qy),
                     -2 * (qz * qy + qw * qx),
                     1 - 2 * (qw * qw + qz * qz)], np.float32)


def draw_command(viewer, d, teleop):
    """Spheres at the VR targets, a line along the movement direction."""
    scn = viewer.user_scn
    scn.ngeom = 0
    root, quat = d.qpos[:3], d.qpos[3:7]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    R = R.reshape(3, 3)

    vr = teleop.vr_3point()
    if vr is not None:
        colors = [[0.2, 1, 0.4, 1], [1, 0.4, 0.2, 1], [0.2, 0.6, 1, 1]]  # L, R, torso
        for p, c in zip(vr[0], colors):
            w = root + R @ p
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.04, 0, 0]), w, np.eye(3).flatten(),
                                np.array(c, np.float32))
            scn.ngeom += 1

    nav = teleop.nav_command()
    if nav["speed"] > 0:
        tip = root + R @ (np.asarray(nav["movement"]) * 0.6)
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE,
                            np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                            np.array([1, 1, 0.2, 1], np.float32))
        mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE, 5,
                             root.astype(np.float64), tip.astype(np.float64))
        scn.ngeom += 1


def _write_recording(
    record_dir: Path,
    *,
    args,
    cfg,
    profile,
    dof_names: list[str],
    kp: np.ndarray,
    kd: np.ndarray,
    action_scale: np.ndarray,
    runner,
    rows: dict[str, list],
    fell: bool,
    interrupted: bool,
) -> None:
    """Persist an interactive run after simulation, never in the control loop."""
    record_dir.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(values) for name, values in rows.items()}
    tmp_state = record_dir / "state.npz.tmp"
    with tmp_state.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    tmp_state.replace(record_dir / "state.npz")

    meta = {
        "schema": "e_skin.mujoco_joystick.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_profile": args.model_profile,
        "embodiment": profile.name,
        "trial_label": args.trial_label,
        "command_source": (
            "unitree-remote" if args.remote
            else "keyboard-native-gamepad" if args.native_gamepad and args.view
            else "synthetic-native-gamepad" if args.native_gamepad
            else "keyboard"
        ),
        "remote_protocol": (
            "native-gamepad-manager" if (args.remote or args.native_gamepad) else None
        ),
        "remote_interface": args.interface if args.remote else None,
        "ground_friction": float(args.effective_ground_friction),
        "ground_friction_overridden": args.ground_friction is not None,
        "joint_encoder_offset_convention": (
            "measured_q = true_mujoco_q + offset; low-level PD uses measured_q"
        ),
        "joint_encoder_offset_rad": np.asarray(
            args.joint_encoder_offset_rad, np.float64
        ).tolist(),
        "joint_encoder_offset_deg": np.rad2deg(
            np.asarray(args.joint_encoder_offset_rad, np.float64)
        ).tolist(),
        "plant_gain_scale": np.asarray(
            args.plant_gain_scale_values, np.float64
        ).tolist(),
        "plant_target_delay_ms": np.asarray(
            args.plant_target_delay_ms_values, np.float64
        ).tolist(),
        "plant_target_delay_steps": np.asarray(
            args.plant_target_delay_steps, np.int64
        ).tolist(),
        "planner_replan_period_s": float(args.effective_replan_period_s),
        "planner_random_seed": int(cfg.sonic_planner_random_seed),
        "config": str(Path(args.config).resolve()),
        "mjcf": str(Path(args.mjcf).expanduser().resolve()),
        "control_dt": float(args.control_dt),
        "joint_names": dof_names,
        "leg_joint_names": dof_names[:12],
        "kp": np.asarray(kp, np.float64).tolist(),
        "kd": np.asarray(kd, np.float64).tolist(),
        "action_scale": np.asarray(action_scale, np.float64).tolist(),
        "dof_lower": np.asarray(cfg.dof_lower, np.float64).tolist(),
        "dof_upper": np.asarray(cfg.dof_upper, np.float64).tolist(),
        "blend": bool(args.blend),
        "planner": bool(args.planner),
        "idle_hold": bool(args.idle_hold),
        "planner_speed_override": args.planner_speed,
        "axes_from_step": args.axes_from_step,
        "axes_until_step": args.axes_until,
        "action_ema": float(cfg.action_ema),
        "adaptation_profile": str(cfg.sonic_adaptation_profile),
        "reduced_proprioception": str(cfg.sonic_reduced_proprioception),
        "shadow_settling_time_s": float(cfg.sonic_shadow_settling_time_s),
        "mask_missing_reference": bool(cfg.sonic_mask_missing_reference),
        "mask_missing_action_history": bool(
            cfg.sonic_mask_missing_action_history
        ),
        "missing_adaptation_scope": str(cfg.sonic_missing_adaptation_scope),
        "motor_action_scale": np.asarray(
            cfg.sonic_motor_action_scale, np.float64
        ).tolist(),
        "motor_action_abs_limit": np.asarray(
            cfg.sonic_motor_action_abs_limit, np.float64
        ).tolist(),
        "motor_action_ema": np.asarray(
            cfg.sonic_motor_action_ema, np.float64
        ).tolist(),
        "motor_target_rate_limit": np.asarray(
            cfg.sonic_motor_target_rate_limit, np.float64
        ).tolist(),
        "action_history_source": str(cfg.sonic_action_history_source),
        "reference_leg_scale": float(cfg.sonic_reference_leg_scale),
        "reference_root_height_scale": float(
            cfg.sonic_reference_root_height_scale
        ),
        "reference_joint_scale": np.asarray(
            cfg.sonic_reference_joint_scale, np.float64
        ).tolist(),
        "reference_joint_dvel_scale": (
            None if getattr(cfg, "sonic_reference_joint_dvel_scale", None) is None
            else np.asarray(cfg.sonic_reference_joint_dvel_scale, np.float64).tolist()
        ),
        "reference_joint_offset_rad": (
            None if getattr(cfg, "sonic_reference_joint_offset_rad", None) is None
            else np.asarray(cfg.sonic_reference_joint_offset_rad, np.float64).tolist()
        ),
        "reference_joint_flex_scale": (
            None if getattr(cfg, "sonic_reference_joint_flex_scale", None) is None
            else np.asarray(cfg.sonic_reference_joint_flex_scale, np.float64).tolist()
        ),
        "derived_yaw_command_scale": float(
            getattr(args, "derived_yaw_command_scale", 1.0)
        ),
        "yaw_command_scale": float(
            getattr(args, "applied_yaw_command_scale", 1.0)
        ),
        "yaw_inplace_scale": float(
            getattr(args, "applied_yaw_inplace_scale", 1.0)
        ),
        "yaw_command_scale_overridden": bool(
            getattr(args, "yaw_command_scale", None) is not None
        ),
        "yaw_inplace_scale_overridden": bool(
            getattr(args, "yaw_inplace_scale", None) is not None
        ),
        "realtime": bool(args.realtime),
        "encoder_providers": list(runner.encoder_providers),
        "decoder_providers": list(runner.decoder_providers),
        "fell": bool(fell),
        "interrupted": bool(interrupted),
        "rows": int(len(arrays.get("time_s", []))),
    }
    tmp_meta = record_dir / "meta.json.tmp"
    tmp_meta.write_text(json.dumps(meta, indent=2) + "\n")
    tmp_meta.replace(record_dir / "meta.json")
    print(f"recording: {record_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="g1_23dof_sonic.yaml")
    ap.add_argument(
        "--model-profile",
        choices=model_profile_names(),
        default="low-latency",
        help="atomic SONIC encoder/decoder contract",
    )
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--clip", default=None, help="default: sonic_motion_clip from the config")
    ap.add_argument("--pico", choices=["nav", "vr3", "both", "idle", "carry", "joystick"],
                    default=None,
                    help="drive the planner with a scripted teleop command")
    ap.add_argument("--planner", action="store_true",
                    help="force the G1 encoder, exactly deploy --mode planner")
    ap.add_argument(
        "--idle-hold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="hold measured joint pose in IDLE without changing recurrent planner "
             "context (default: enabled; --no-idle-hold is diagnostic only)",
    )
    ap.add_argument("--slow", action="store_true",
                    help="joystick SLOW_WALK instead of WALK, as deploy --slow")
    ap.add_argument("--speed", type=float, default=0.2, help="--pico nav: m/s")
    ap.add_argument("--axes", default=None,
                    help="--pico joystick: fixed 'lx,ly,rx' instead of the keyboard")
    ap.add_argument(
        "--axes-from-step",
        type=int,
        default=None,
        help="keep joystick neutral until this control step, then apply --axes",
    )
    ap.add_argument("--axes-until", type=int, default=None,
                    help="release the stick (axes to 0) after N steps")
    ap.add_argument("--remote", action="store_true",
                    help="--pico joystick: real Unitree remote via DDS "
                         "(robot on and reachable; nothing publishes lowcmd)")
    ap.add_argument(
        "--native-gamepad",
        action="store_true",
        help="use gamepad_manager semantics with --axes, but without DDS",
    )
    ap.add_argument(
        "--joystick-mode",
        choices=["slow_walk", "walk", "run", "stealth_walk", "injured_walk"],
        default=None,
        help="deterministic joystick mode override for headless diagnostics",
    )
    ap.add_argument(
        "--interface",
        default=None,
        help="network interface used only by --remote LowState subscription",
    )
    ap.add_argument(
        "--ground-friction",
        type=float,
        default=None,
        metavar="MU",
        help="override floor sliding friction (default MJCF value is normally 1.0)",
    )
    ap.add_argument(
        "--joint-encoder-offset-deg",
        action="append",
        default=[],
        metavar="JOINT=DEGREES",
        help=(
            "repeatable signed encoder bias; measured_q=true_q+offset and the "
            "same biased measurement drives observations and low-level PD"
        ),
    )
    ap.add_argument(
        "--plant-gain-scale",
        action="append",
        default=[],
        metavar="GROUP=VALUE",
        help=(
            "repeatable simulated plant fault: scale effective PD stiffness "
            "and damping for a joint/group without changing SONIC"
        ),
    )
    ap.add_argument(
        "--plant-target-delay-ms",
        action="append",
        default=[],
        metavar="GROUP=MILLISECONDS",
        help=(
            "repeatable simulated plant fault: delay the target reaching a "
            "joint/group; SONIC still receives current measured state"
        ),
    )
    ap.add_argument("--blend", action=argparse.BooleanOptionalAction, default=True,
                    help="same one-second raised-cosine handover as deployment "
                         "(use --no-blend only for official parity diagnostics)")
    ap.add_argument("--replan-ungated", action="store_true",
                    help="replan on every command instead of official_gated, as deploy")
    ap.add_argument(
        "--planner-replan-period",
        type=float,
        default=None,
        metavar="SECONDS",
        help="override the periodic replan interval for ordinary locomotion",
    )
    ap.add_argument(
        "--planner-seed",
        type=int,
        default=None,
        metavar="INTEGER",
        help="override the deterministic planner random seed for robustness studies",
    )
    ap.add_argument(
        "--planner-speed",
        type=float,
        default=None,
        help="diagnostic fixed target_vel override for native SLOW_WALK",
    )
    ap.add_argument(
        "--action-ema",
        type=float,
        default=None,
        help="diagnostic physical-action EMA coefficient in (0,1]",
    )
    ap.add_argument(
        "--reduced-proprioception",
        choices=["zero", "reference", "shadow"],
        default=None,
        help="23-DoF missing-axis decoder observation strategy",
    )
    ap.add_argument(
        "--shadow-settling-time",
        type=float,
        default=None,
        metavar="SECONDS",
        help="2%% settling time for --reduced-proprioception shadow",
    )
    ap.add_argument(
        "--mask-missing-reference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="diagnostic encoder-reference mask on absent 23-DoF axes",
    )
    ap.add_argument(
        "--mask-missing-action-history",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="diagnostic decoder action-history mask on absent axes",
    )
    ap.add_argument(
        "--missing-adaptation-scope",
        choices=["all", "waist", "wrist"],
        default=None,
        help="limit missing reference/history masks to waist or wrists",
    )
    for flag, description in (
        ("motor-scale", "multiply normalized action deltas"),
        ("motor-clip", "clip absolute normalized action deltas"),
        ("motor-ema", "EMA coefficient for normalized action deltas"),
        ("motor-rate", "target-angle slew limit in rad/s"),
    ):
        ap.add_argument(
            f"--{flag}",
            action="append",
            default=[],
            metavar="GROUP=VALUE",
            help=(
                f"diagnostic physical motor shaping: {description}; repeat for "
                "legs, hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, "
                "ankle_roll, upper, or all"
            ),
        )
    ap.add_argument(
        "--action-history-source",
        choices=["raw", "executed"],
        default=None,
        help="decoder last_action feedback after output shaping (default: raw official)",
    )
    ap.add_argument(
        "--reference-leg-scale",
        type=float,
        default=None,
        metavar="SCALE",
        help=(
            "scale locomotion leg-reference deviations around the captured "
            "stand before SONIC's encoder; planner recurrence is unchanged"
        ),
    )
    ap.add_argument(
        "--reference-root-height-scale",
        type=float,
        default=None,
        metavar="SCALE",
        help="scale locomotion root-height deviations around captured stand",
    )
    ap.add_argument(
        "--reference-joint-scale",
        action="append",
        default=[],
        metavar="GROUP=VALUE",
        help=(
            "repeatable encoder-reference amplitude scale for physical joint "
            "groups; unlike motor shaping, tracker recovery output is untouched"
        ),
    )
    ap.add_argument(
        "--reference-joint-dvel-scale",
        action="append",
        default=[],
        metavar="GROUP=VALUE",
        help=(
            "repeatable encoder-reference velocity scale; default matches "
            "--reference-joint-scale so a smaller stride also slows dvel_fut"
        ),
    )
    ap.add_argument(
        "--reference-joint-offset-deg",
        action="append",
        default=[],
        metavar="GROUP=DEGREES",
        help=(
            "repeatable gait-only encoder-reference pose offset in degrees; "
            "inactive in captured IDLE / standing"
        ),
    )
    ap.add_argument(
        "--reference-joint-flex-scale",
        action="append",
        default=[],
        metavar="GROUP=VALUE",
        help=(
            "preserve flexion-side reference amplitude after a smaller stride "
            "scale; unspecified groups keep --reference-joint-scale"
        ),
    )
    ap.add_argument(
        "--slow-gait",
        action="store_true",
        help=(
            "healthy-plant Original Release recipe: sagittal 0.28, gait-only "
            "knee +8 deg, walk yaw 0.15 / in-place 0.35. Exclusive with "
            "explicit sagittal or hip-pitch --reference-joint-scale"
        ),
    )
    ap.add_argument(
        "--yaw-command-scale",
        type=float,
        default=None,
        metavar="SCALE",
        help=(
            "override stick yaw-rate scale; default is the smallest sagittal "
            "reference scale so walk-plus-turn matches the shortened stride"
        ),
    )
    ap.add_argument(
        "--yaw-inplace-scale",
        type=float,
        default=None,
        metavar="SCALE",
        help=(
            "override turn-only yaw-rate scale; walk-plus-turn keeps "
            "--yaw-command-scale so a short stride can still step in place"
        ),
    )
    ap.add_argument("--steps", type=int, default=None, help="default: whole clip")
    ap.add_argument("--control-dt", type=float, default=None)
    ap.add_argument("--init-height", type=float, default=0.78)
    ap.add_argument("--zero-action", action="store_true",
                    help="diagnostic: hold default pose, ignore policy")
    ap.add_argument("--trace", action="store_true", help="print per-step height/tilt/dof-err")
    ap.add_argument("--view", action="store_true", help="open the MuJoCo viewer")
    ap.add_argument("--viser", action="store_true",
                    help="browser viewer + drive buttons on :8080 (for SSH; "
                         "replaces the GLFW window)")
    ap.add_argument("--realtime", action="store_true",
                    help="with --view, pace the sim to wall clock")
    ap.add_argument(
        "--sim-plugin",
        default=None,
        help="dotted path module:Class of a per-step observer plugin; "
             "instantiated with (model, data, args), on_control_step(step) "
             "runs after each control step, finalize(record_dir) after the run",
    )
    ap.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help="write state.npz and meta.json here after the run",
    )
    ap.add_argument(
        "--trial-label",
        default="interactive",
        help="descriptive label stored with a recording",
    )
    ap.add_argument(
        "--capture-at-step",
        type=int,
        default=None,
        help="diagnostic: request stable-IDLE qualification at this control step",
    )
    args = ap.parse_args()
    args.derived_yaw_command_scale = 1.0
    args.applied_yaw_command_scale = 1.0
    args.applied_yaw_inplace_scale = 1.0

    cfg = Config(args.config)
    cfg.select_sonic_model_profile(args.model_profile)
    if args.action_ema is not None:
        if not np.isfinite(args.action_ema) or not 0.0 < args.action_ema <= 1.0:
            raise SystemExit("--action-ema must be finite and in (0, 1]")
        cfg.action_ema = float(args.action_ema)
    if args.reduced_proprioception is not None:
        cfg.sonic_reduced_proprioception = args.reduced_proprioception
    if args.shadow_settling_time is not None:
        if not np.isfinite(args.shadow_settling_time) or args.shadow_settling_time <= 0:
            raise SystemExit("--shadow-settling-time must be finite and positive")
        cfg.sonic_shadow_settling_time_s = float(args.shadow_settling_time)
    if args.mask_missing_reference is not None:
        cfg.sonic_mask_missing_reference = bool(args.mask_missing_reference)
    if args.mask_missing_action_history is not None:
        cfg.sonic_mask_missing_action_history = bool(
            args.mask_missing_action_history
        )
    if args.missing_adaptation_scope is not None:
        cfg.sonic_missing_adaptation_scope = args.missing_adaptation_scope
    profile = get_robot_profile(cfg.sonic_robot_profile)
    group_indices = {
        "all": np.arange(profile.physical_dof, dtype=np.int64),
        "legs": np.arange(min(12, profile.physical_dof), dtype=np.int64),
        "upper": np.arange(min(12, profile.physical_dof), profile.physical_dof),
    }
    physical_names = np.asarray(profile.physical_joint_names)
    for group in (
        "hip_pitch", "hip_roll", "hip_yaw", "knee",
        "ankle_pitch", "ankle_roll",
    ):
        group_indices[group] = np.flatnonzero(
            np.char.find(physical_names.astype(str), group) >= 0
        )
    for index, name in enumerate(physical_names):
        short = str(name).removesuffix("_joint")
        group_indices.setdefault(short, np.array([index], dtype=np.int64))
    group_indices["sagittal"] = np.unique(np.concatenate([
        group_indices["hip_pitch"],
        group_indices["knee"],
        group_indices["ankle_pitch"],
    ]))
    group_indices["lateral"] = np.unique(np.concatenate([
        group_indices["hip_roll"],
        group_indices["hip_yaw"],
        group_indices["ankle_roll"],
    ]))
    for joint_index, joint_name in enumerate(profile.physical_joint_names):
        group_indices[joint_name] = np.asarray([joint_index], dtype=np.int64)
        group_indices[joint_name.removesuffix("_joint")] = np.asarray(
            [joint_index], dtype=np.int64
        )

    def apply_motor_overrides(specs, attribute: str, kind: str) -> None:
        values = np.asarray(getattr(cfg, attribute), np.float64).copy()
        if values.shape != (profile.physical_dof,):
            raise SystemExit(
                f"{attribute} must have {profile.physical_dof} physical values, "
                f"got {values.shape}"
            )
        for spec in specs:
            try:
                group, text = spec.split("=", 1)
                value = float(text)
            except ValueError as exc:
                raise SystemExit(
                    f"--motor-{kind} expects GROUP=VALUE, got {spec!r}"
                ) from exc
            group = group.strip().lower()
            if group not in group_indices:
                raise SystemExit(
                    f"unknown --motor-{kind} group {group!r}; choose "
                    + ", ".join(group_indices)
                )
            if not np.isfinite(value) or value <= 0:
                raise SystemExit(f"--motor-{kind} value must be finite and positive")
            if kind == "ema" and value > 1:
                raise SystemExit("--motor-ema value must be in (0, 1]")
            values[group_indices[group]] = value
        setattr(cfg, attribute, values)

    apply_motor_overrides(args.motor_scale, "sonic_motor_action_scale", "scale")
    apply_motor_overrides(args.motor_clip, "sonic_motor_action_abs_limit", "clip")
    apply_motor_overrides(args.motor_ema, "sonic_motor_action_ema", "ema")
    apply_motor_overrides(
        args.motor_rate, "sonic_motor_target_rate_limit", "rate"
    )
    if getattr(args, "slow_gait", False):
        from sonic.model.reference_adaptation import expand_slow_gait_recipe

        if str(cfg.sonic_adaptation_profile).strip().lower() != "forward-stable-23":
            raise SystemExit(
                "--slow-gait requires a forward-stable-23 adaptation config"
            )
        if profile.name != "g1_23dof":
            raise SystemExit("--slow-gait requires embodiment g1_23dof")
        try:
            (
                args.reference_joint_scale,
                args.reference_joint_offset_deg,
                args.yaw_command_scale,
                args.yaw_inplace_scale,
            ) = expand_slow_gait_recipe(
                joint_scale_specs=args.reference_joint_scale,
                joint_offset_specs=args.reference_joint_offset_deg,
                yaw_command_scale=args.yaw_command_scale,
                yaw_inplace_scale=args.yaw_inplace_scale,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(
            "Slow-gait recipe: sagittal 0.28, gait-only knee +8 deg, "
            f"walk yaw {args.yaw_command_scale:g} / "
            f"in-place yaw {args.yaw_inplace_scale:g}",
            flush=True,
        )
    if np.asarray(cfg.sonic_reference_joint_scale).shape != (
        profile.physical_dof,
    ):
        raise SystemExit(
            "sonic_reference_joint_scale must contain one value per physical joint"
        )
    reference_joint_scale = np.asarray(
        cfg.sonic_reference_joint_scale, np.float64
    ).copy()
    for spec in args.reference_joint_scale:
        try:
            group, text = spec.split("=", 1)
            value = float(text)
        except ValueError as exc:
            raise SystemExit(
                f"--reference-joint-scale expects GROUP=VALUE, got {spec!r}"
            ) from exc
        group = group.strip().lower()
        if group not in group_indices:
            raise SystemExit(
                f"unknown --reference-joint-scale group {group!r}; choose "
                + ", ".join(sorted(group_indices))
            )
        if not np.isfinite(value) or not 0.0 < value <= 2.0:
            raise SystemExit(
                "--reference-joint-scale value must be finite and in (0, 2]"
            )
        reference_joint_scale[group_indices[group]] = value
    cfg.sonic_reference_joint_scale = reference_joint_scale
    if args.reference_joint_dvel_scale:
        reference_dvel_scale = np.ones(profile.physical_dof, dtype=np.float64)
        for spec in args.reference_joint_dvel_scale:
            try:
                group, text = spec.split("=", 1)
                value = float(text)
            except ValueError as exc:
                raise SystemExit(
                    f"--reference-joint-dvel-scale expects GROUP=VALUE, got {spec!r}"
                ) from exc
            group = group.strip().lower()
            if group not in group_indices:
                raise SystemExit(
                    f"unknown --reference-joint-dvel-scale group {group!r}; choose "
                    + ", ".join(sorted(group_indices))
                )
            if not np.isfinite(value) or not 0.0 < value <= 2.0:
                raise SystemExit(
                    "--reference-joint-dvel-scale value must be finite and in (0, 2]"
                )
            reference_dvel_scale[group_indices[group]] = value
        cfg.sonic_reference_joint_dvel_scale = reference_dvel_scale
    if args.reference_joint_offset_deg:
        reference_offset = np.zeros(profile.physical_dof, dtype=np.float64)
        for spec in args.reference_joint_offset_deg:
            try:
                group, text = spec.split("=", 1)
                value = float(text)
            except ValueError as exc:
                raise SystemExit(
                    f"--reference-joint-offset-deg expects GROUP=DEGREES, got {spec!r}"
                ) from exc
            group = group.strip().lower()
            if group not in group_indices:
                raise SystemExit(
                    f"unknown --reference-joint-offset-deg group {group!r}; choose "
                    + ", ".join(sorted(group_indices))
                )
            if not np.isfinite(value) or abs(value) > 45.0:
                raise SystemExit(
                    "--reference-joint-offset-deg value must be finite and within ±45"
                )
            reference_offset[group_indices[group]] = np.radians(value)
        cfg.sonic_reference_joint_offset_rad = reference_offset
    if args.reference_joint_flex_scale:
        reference_flex_scale = np.asarray(
            cfg.sonic_reference_joint_scale, np.float64
        ).copy()
        for spec in args.reference_joint_flex_scale:
            try:
                group, text = spec.split("=", 1)
                value = float(text)
            except ValueError as exc:
                raise SystemExit(
                    f"--reference-joint-flex-scale expects GROUP=VALUE, got {spec!r}"
                ) from exc
            group = group.strip().lower()
            if group not in group_indices:
                raise SystemExit(
                    f"unknown --reference-joint-flex-scale group {group!r}; choose "
                    + ", ".join(sorted(group_indices))
                )
            if not np.isfinite(value) or not 0.0 < value <= 2.0:
                raise SystemExit(
                    "--reference-joint-flex-scale value must be finite and in (0, 2]"
                )
            reference_flex_scale[group_indices[group]] = value
        cfg.sonic_reference_joint_flex_scale = reference_flex_scale
    def parse_plant_overrides(
        specs, *, neutral: float, kind: str, lower: float, upper: float
    ) -> np.ndarray:
        values = np.full(profile.physical_dof, neutral, dtype=np.float64)
        for spec in specs:
            try:
                group, text = spec.split("=", 1)
                value = float(text)
            except ValueError as exc:
                raise SystemExit(
                    f"--plant-{kind} expects GROUP=VALUE, got {spec!r}"
                ) from exc
            group = group.strip().lower()
            if group not in group_indices:
                raise SystemExit(
                    f"unknown --plant-{kind} group {group!r}; choose a physical "
                    "joint name or " + ", ".join(sorted(group_indices))
                )
            if not np.isfinite(value) or not lower <= value <= upper:
                raise SystemExit(
                    f"--plant-{kind} value must be finite and in [{lower:g}, {upper:g}]"
                )
            values[group_indices[group]] = value
        return values

    args.plant_gain_scale_values = parse_plant_overrides(
        args.plant_gain_scale,
        neutral=1.0,
        kind="gain-scale",
        lower=0.05,
        upper=2.0,
    )
    args.plant_target_delay_ms_values = parse_plant_overrides(
        args.plant_target_delay_ms,
        neutral=0.0,
        kind="target-delay-ms",
        lower=0.0,
        upper=200.0,
    )
    if args.action_history_source is not None:
        cfg.sonic_action_history_source = args.action_history_source
    for argument, attribute in (
        (args.reference_leg_scale, "sonic_reference_leg_scale"),
        (
            args.reference_root_height_scale,
            "sonic_reference_root_height_scale",
        ),
    ):
        if argument is not None:
            if not np.isfinite(argument) or not 0.0 < argument <= 2.0:
                raise SystemExit(
                    f"--{attribute.removeprefix('sonic_').replace('_', '-')} "
                    "must be finite and in (0, 2]"
                )
            setattr(cfg, attribute, float(argument))
    args.effective_replan_period_s = float(
        cfg.sonic_planner_replan_interval_s
        if args.planner_replan_period is None
        else args.planner_replan_period
    )
    if not 0.1 <= args.effective_replan_period_s <= 10.0:
        raise SystemExit("--planner-replan-period must be in [0.1, 10.0] seconds")
    if args.planner_seed is not None:
        if args.planner_seed < 0:
            raise SystemExit("--planner-seed must be non-negative")
        cfg.sonic_planner_random_seed = int(args.planner_seed)
    if (args.remote or args.native_gamepad) and args.pico != "joystick":
        raise SystemExit("--remote/--native-gamepad require --pico joystick")
    if args.joystick_mode is not None and args.pico != "joystick":
        raise SystemExit("--joystick-mode requires --pico joystick")
    if args.axes_from_step is not None and not args.axes:
        raise SystemExit("--axes-from-step requires --axes")
    if args.planner_speed is not None:
        if not np.isfinite(args.planner_speed) or args.planner_speed <= 0:
            raise SystemExit("--planner-speed must be finite and positive")
        if args.joystick_mode not in (None, "slow_walk"):
            raise SystemExit("--planner-speed is currently scoped to SLOW_WALK")
    dof_names = list(profile.physical_joint_names)
    args.joint_encoder_offset_rad = _parse_joint_encoder_offsets_deg(
        args.joint_encoder_offset_deg, dof_names
    )
    encoder_position_offset = args.joint_encoder_offset_rad.astype(np.float32)
    nonzero_encoder_offsets = np.flatnonzero(args.joint_encoder_offset_rad)
    if len(nonzero_encoder_offsets):
        print("MuJoCo joint-encoder measurement offsets:")
        for joint_index in nonzero_encoder_offsets:
            print(
                f"  {dof_names[joint_index]}: "
                f"{np.rad2deg(args.joint_encoder_offset_rad[joint_index]):+.1f} deg "
                "(measured=true+offset; PD equilibrium shifts by -offset)"
            )
    default = np.asarray(cfg.physical_default_angles, np.float32)
    idx = np.asarray(cfg.action_joint2motor_idx)
    kp = np.asarray(cfg.kps, np.float32)[idx]
    kd = np.asarray(cfg.kds, np.float32)[idx]
    action_scale = np.asarray(cfg.action_scale, np.float32)
    if args.control_dt is None:
        args.control_dt = float(cfg.control_dt)
    args.plant_target_delay_steps = np.rint(
        args.plant_target_delay_ms_values / (1000.0 * args.control_dt)
    ).astype(np.int64)
    effective_delay_ms = 1000.0 * args.control_dt * args.plant_target_delay_steps
    nonneutral_plant = np.flatnonzero(
        (np.abs(args.plant_gain_scale_values - 1.0) > 1e-12)
        | (args.plant_target_delay_steps != 0)
    )
    if len(nonneutral_plant):
        print("MuJoCo simulated plant faults (SONIC remains unchanged):")
        for joint_index in nonneutral_plant:
            print(
                f"  {dof_names[joint_index]}: gain="
                f"{args.plant_gain_scale_values[joint_index]:.2f}, target delay="
                f"{effective_delay_ms[joint_index]:.0f} ms"
            )

    # MJCF as authored: its own timestep, damping, friction, foot geoms
    m = mujoco.MjModel.from_xml_path(str(Path(args.mjcf).expanduser()))
    d = mujoco.MjData(m)
    plugin = None
    if args.sim_plugin:
        import importlib
        plugin_module, _, plugin_class = args.sim_plugin.partition(":")
        plugin = getattr(importlib.import_module(plugin_module), plugin_class)(m, d, args)
    decim = max(1, round(args.control_dt / m.opt.timestep))

    floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise SystemExit(f"{args.mjcf} does not contain a named floor geom")
    if args.ground_friction is not None:
        if not np.isfinite(args.ground_friction) or args.ground_friction < 0.0:
            raise SystemExit("--ground-friction must be a finite value >= 0")
        # Dynamic contacts otherwise use max(floor friction, foot friction).
        # A higher floor priority makes this coefficient authoritative for all
        # floor contacts while retaining the MJCF torsional/rolling values.
        m.geom_friction[floor_id, 0] = args.ground_friction
        m.geom_priority[floor_id] = int(np.max(m.geom_priority)) + 1
    args.effective_ground_friction = float(m.geom_friction[floor_id, 0])
    print(
        f"MuJoCo floor sliding friction: {args.effective_ground_friction:g}"
        + (" (priority override)" if args.ground_friction is not None else " (MJCF default)")
    )
    foot_body_ids = {
        side: mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link"
        )
        for side in ("left", "right")
    }
    if any(body_id < 0 for body_id in foot_body_ids.values()):
        raise SystemExit("MJCF is missing left/right ankle roll bodies")
    foot_geom_ids = {
        side: np.flatnonzero(m.geom_bodyid == body_id).astype(np.int32)
        for side, body_id in foot_body_ids.items()
    }
    foot_corner_geom_ids = {
        side: ids[m.geom_type[ids] == int(mujoco.mjtGeom.mjGEOM_SPHERE)]
        for side, ids in foot_geom_ids.items()
    }

    def _foot_diagnostics(side: str):
        ids = set(int(value) for value in foot_geom_ids[side])
        contact_count = 0
        for contact_index in range(d.ncon):
            contact = d.contact[contact_index]
            pair = (int(contact.geom1), int(contact.geom2))
            if (pair[0] == floor_id and pair[1] in ids) or (
                pair[1] == floor_id and pair[0] in ids
            ):
                contact_count += 1
        corners = foot_corner_geom_ids[side]
        corner_z = (
            np.asarray(d.geom_xpos[corners, 2], np.float32).copy()
            if len(corners)
            else np.full(4, np.nan, np.float32)
        )
        return (
            np.bool_(contact_count > 0),
            np.int8(contact_count),
            np.asarray(d.xpos[foot_body_ids[side]], np.float32).copy(),
            corner_z,
        )

    def _stand_support_diagnostics():
        left = _foot_diagnostics("left")
        right = _foot_diagnostics("right")
        corner_ids = np.concatenate(
            (foot_corner_geom_ids["left"], foot_corner_geom_ids["right"])
        )
        if len(corner_ids) < 6:
            return None
        corner_xyz = np.asarray(d.geom_xpos[corner_ids], np.float64)
        # Body 1 is the free-root pelvis, so its subtree COM is the whole robot.
        com_xy = np.asarray(d.subtree_com[1, :2], np.float64)
        return {
            # Two or more contacts per foot establish a finite support patch.
            # All four sphere centers are checked separately for near-ground
            # height and coplanarity; requiring four solver contacts exactly
            # rejects a geometrically flat foot because of contact tolerance.
            "left_supported": bool(left[1] >= 2),
            "right_supported": bool(right[1] >= 2),
            "max_corner_z_m": float(np.max(corner_xyz[:, 2])),
            "max_foot_corner_range_m": float(max(
                np.ptp(d.geom_xpos[foot_corner_geom_ids["left"], 2]),
                np.ptp(d.geom_xpos[foot_corner_geom_ids["right"], 2]),
            )),
            "com_support_margin_m": _support_margin_xy(corner_xyz[:, :2], com_xy),
            "base_xy": np.asarray(d.qpos[:2], np.float64).copy(),
            "base_planar_speed_mps": float(np.linalg.norm(d.qvel[:2])),
        }

    jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in dof_names}
    missing_model_joints = [name for name, joint_id in jid.items() if joint_id < 0]
    if missing_model_joints:
        raise SystemExit(
            f"{args.mjcf} does not contain the {profile.name} physical joints: "
            f"{missing_model_joints}"
        )
    qadr = np.array([m.jnt_qposadr[jid[n]] for n in dof_names])
    vadr = np.array([m.jnt_dofadr[jid[n]] for n in dof_names])

    # affine position-actuator PD (ProtoMotions): force = kp*(ctrl-q) - kd*qd, integrated implicitly
    act_for_joint = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}

    def _set_pd(a, kp, kd, eff):
        m.actuator_gaintype[a] = 0
        m.actuator_gainprm[a, 0] = kp
        m.actuator_biastype[a] = 1
        m.actuator_biasprm[a, 0] = 0.0
        m.actuator_biasprm[a, 1] = -kp
        m.actuator_biasprm[a, 2] = -kd
        m.actuator_ctrllimited[a] = 0
        if eff > 0:
            m.actuator_forcerange[a] = [-eff, eff]
            m.actuator_forcelimited[a] = 1

    try:
        pd_act = np.array([act_for_joint[jid[n]] for n in dof_names])
    except KeyError as exc:
        raise SystemExit(f"physical joint has no MuJoCo actuator: joint id {exc.args[0]}") from exc
    for k, n in enumerate(dof_names):
        j = jid[n]
        eff = float(m.jnt_actfrcrange[j, 1]) if m.jnt_actfrclimited[j] else 0.0
        plant_scale = float(args.plant_gain_scale_values[k])
        _set_pd(
            pd_act[k],
            float(kp[k]) * plant_scale,
            float(kd[k]) * plant_scale,
            eff,
        )
    # Undriven joints get the config's fixed_kps/fixed_kds, as on the robot.
    fkp = np.asarray(cfg.fixed_kps, np.float32)
    fkd = np.asarray(cfg.fixed_kds, np.float32)
    hold_act, hold_tgt = [], []
    fixed_names = [MUJOCO_JOINT_NAMES[i] for i in cfg.fixed_joint2motor_idx]
    for k, n in enumerate(fixed_names):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j >= 0 and j in act_for_joint:
            _set_pd(act_for_joint[j], float(fkp[k]), float(fkd[k]), 0.0)
            hold_act.append(act_for_joint[j])
            hold_tgt.append(float(cfg.fixed_target[k]))
    hold_act = np.array(hold_act, dtype=int)
    hold_tgt = np.array(hold_tgt, np.float32)

    teleop = None
    if args.pico:
        from sonic.pico_reference import PicoReferenceWindow

        if args.pico == "joystick":
            from sonic.inputs.pico_client import LocomotionMode
            if args.remote or args.native_gamepad:
                from sonic.inputs.joystick_teleop import GamepadManagerTeleop

                teleop = GamepadManagerTeleop()
                if not args.remote:
                    teleop.control_started = True
            else:
                from sonic.inputs.joystick_teleop import JoystickTeleop

                teleop = JoystickTeleop(
                    locomotion_mode=LocomotionMode.SLOW_WALK if args.slow
                    else LocomotionMode.WALK
                )
            scheduled_axes = (
                tuple(map(float, args.axes.split(","))) if args.axes else None
            )
            if scheduled_axes is not None and (
                args.axes_from_step is None or args.axes_from_step <= 0
            ):
                teleop.set_axes(*scheduled_axes)
            if args.joystick_mode is not None:
                teleop.set_locomotion_mode(
                    {
                        "slow_walk": LocomotionMode.SLOW_WALK,
                        "walk": LocomotionMode.WALK,
                        "run": LocomotionMode.RUN,
                        "stealth_walk": LocomotionMode.STEALTH_WALK,
                        "injured_walk": LocomotionMode.INJURED_WALK,
                    }[args.joystick_mode]
                )
            if args.planner_speed is not None:
                if not (args.remote or args.native_gamepad):
                    raise SystemExit(
                        "--planner-speed requires --native-gamepad or --remote"
                    )
                teleop.movement_speed = float(args.planner_speed)
            from sonic.model.reference_adaptation import gait_yaw_command_scale

            derived_yaw = gait_yaw_command_scale(cfg.sonic_reference_joint_scale)
            if args.yaw_command_scale is not None:
                if (
                    not np.isfinite(args.yaw_command_scale)
                    or not 0.0 < args.yaw_command_scale <= 2.0
                ):
                    raise SystemExit(
                        "--yaw-command-scale must be finite and in (0, 2]"
                    )
                teleop.yaw_command_scale = float(args.yaw_command_scale)
            else:
                teleop.yaw_command_scale = derived_yaw
            if args.yaw_inplace_scale is not None:
                if (
                    not np.isfinite(args.yaw_inplace_scale)
                    or not 0.0 < args.yaw_inplace_scale <= 2.0
                ):
                    raise SystemExit(
                        "--yaw-inplace-scale must be finite and in (0, 2]"
                    )
                teleop.yaw_inplace_scale = float(args.yaw_inplace_scale)
            args.derived_yaw_command_scale = derived_yaw
            args.applied_yaw_command_scale = float(teleop.yaw_command_scale)
            args.applied_yaw_inplace_scale = float(teleop.yaw_inplace_scale)
            print(
                "Command yaw rate scale "
                f"{teleop.yaw_command_scale:g} "
                f"(in-place {teleop.yaw_inplace_scale:g}; "
                f"derived {derived_yaw:g} from sagittal reference; "
                "hip roll/yaw stay unscaled)"
            )
            if args.remote:
                from unitree_sdk2py.core import channel as _unitree_channel
                from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
                from core.robot.remote import RemoteController

                remote = RemoteController()
                dds_log = _configure_process_local_dds_log(_unitree_channel)
                if args.interface:
                    _unitree_channel.ChannelFactoryInitialize(0, args.interface)
                else:
                    _unitree_channel.ChannelFactoryInitialize(0)
                sub = _unitree_channel.ChannelSubscriber(cfg.lowstate_topic, LowState_)
                sub.Init(lambda msg: remote.set(msg.wireless_remote), 10)
                print(
                    "remote: read-only Unitree sticks from rt/lowstate"
                    + (f" on {args.interface}" if args.interface else "")
                    + "; no LowCmd publisher is created"
                )
                print(f"CycloneDDS process-local log: {dds_log}")
                print(
                    "WAIT_FOR_CONTROL: keep sticks neutral, then press Start; "
                    "Select stops and saves"
                )
                print(
                    "gamepad: D-pad selects set; X/Y next/previous; B resets; "
                    "A sends an immediate planner halt; F1+F3 captures a "
                    "verified steady IDLE pose"
                )
        else:
            teleop = ScriptedTeleop(args.pico, speed=args.speed)
        # wall-paced viewing uses the deploy's planner thread; headless sim
        # outruns the wall clock, so it replans synchronously instead
        ref_kwargs = dict(threaded=args.realtime,
                          gated=not args.replan_ungated,
                          force_g1=args.planner,
                          idle_hold=args.idle_hold,
                          replan_period_s=args.effective_replan_period_s)
        ref = PicoReferenceWindow(cfg, teleop, **ref_kwargs)
        ref.robot = _MujocoJointState(
            d,
            qadr,
            vadr,
            encoder_position_offset=encoder_position_offset,
            support_diagnostics=_stand_support_diagnostics,
        )
        runner = SonicRunner(cfg, ref)
        ref_dof = None
        d.qpos[:3] = [0.0, 0.0, 0.793]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[qadr] = default
        mujoco.mj_forward(m, d)
        ref.initialize_from_robot(d.qpos[3:7].copy(), ref.robot.qj)
        # pico: run until the viewer closes; headless still needs a bound
        n_steps = args.steps or (10**9 if (args.view or args.viser) else 500)
    else:
        clip_path = args.clip or cfg.sonic_motion_clip
        ref = MotionReferenceWindow(
            clip_path,
            loop=False,
            model_profile=cfg.sonic_model_variant,
            robot_profile=cfg.sonic_robot_profile,
        )
        runner = SonicRunner(cfg, ref)
        clip = np.load(Path(clip_path).expanduser())
        ref_dof = profile.isaaclab_to_physical(clip["dof_il"])
        ref_vel = profile.isaaclab_to_physical(clip["dof_vel_il"])
        root_pos = clip["body_pos_il"][:, 0]
        root_quat = clip["body_quat_wxyz_il"][:, 0]

        # start on the clip's frame 0, as the Isaac eval does
        d.qpos[:3] = [0.0, 0.0, float(root_pos[0, 2])]
        d.qpos[3:7] = root_quat[0]
        d.qpos[qadr] = ref_dof[0]
        d.qvel[vadr] = ref_vel[0]
        mujoco.mj_forward(m, d)
        ref.initialize_from_robot(
            d.qpos[3:7].copy(),
            d.qpos[qadr].copy() + encoder_position_offset,
        )
        n_steps = args.steps or ref.T

    viewer = None
    key_cb = None
    if args.pico == "joystick" and not args.remote:
        # Tap to set latched virtual axes. These numeric-keypad bindings are
        # disjoint from the complete documented MuJoCo Simulate shortcut set.
        axis_keys = {
            glfw.KEY_KP_8: (0, -1, 0),   # forward
            glfw.KEY_KP_2: (0, 1, 0),    # backward
            glfw.KEY_KP_4: (-1, 0, 0),   # strafe left
            glfw.KEY_KP_6: (1, 0, 0),    # strafe right
            glfw.KEY_KP_7: (0, 0, -1),   # turn left
            glfw.KEY_KP_9: (0, 0, 1),    # turn right
        }
        mode_keys = {
            glfw.KEY_KP_1: "slow_walk",
            glfw.KEY_KP_0: "walk",
            glfw.KEY_KP_3: "run",
        }

        def key_cb(key):
            handled = True
            if key == glfw.KEY_KP_5:
                teleop.set_axes(0, 0, 0)
            elif key == glfw.KEY_KP_ENTER:
                ref.request_stable_stand_capture("MuJoCo Numpad Enter")
            elif key in mode_keys:
                from sonic.inputs.pico_client import LocomotionMode

                teleop.set_locomotion_mode(
                    {
                        "slow_walk": LocomotionMode.SLOW_WALK,
                        "walk": LocomotionMode.WALK,
                        "run": LocomotionMode.RUN,
                    }[mode_keys[key]]
                )
            elif key in axis_keys:
                dlx, dly, drx = axis_keys[key]
                teleop.set_axes(np.clip(teleop.lx + 0.5 * dlx, -1, 1),
                                np.clip(teleop.ly - 0.5 * dly, -1, 1),
                                np.clip(teleop.rx + 0.5 * drx, -1, 1))
            else:
                handled = False
            if not handled:
                return
            print(
                f"\rmode={teleop.locomotion_mode.name:<9} "
                f"axes lx={teleop.lx:+.1f} ly={teleop.ly:+.1f} "
                f"rx={teleop.rx:+.1f}   ",
                end="",
                flush=True,
            )

    if args.view:
        # not `import mujoco.viewer`: that shadows the module-level import
        from mujoco import viewer as _mj_viewer

        viewer = _mj_viewer.launch_passive(m, d, key_callback=key_cb)
        if key_cb is not None:
            print(
                "SONIC controls use only the numeric keypad; MuJoCo's native "
                "shortcuts remain available:\n"
                "  Num7 turn-L   Num8 forward   Num9 turn-R\n"
                "  Num4 left     Num5 IDLE      Num6 right\n"
                "  Num1 slow     Num2 backward  Num3 run\n"
                "  Num0 walk     Numpad Enter capture steady IDLE pose"
            )
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -15
        print("viewer open -- close the window to stop")

    if plugin is not None and hasattr(plugin, "attach_teleop"):
        plugin.attach_teleop(teleop)
    viser_view = None
    if args.viser:
        from sonic.visualization import ViserSimView

        on_capture = None
        if args.pico and hasattr(ref, "request_stable_stand_capture"):
            on_capture = lambda: ref.request_stable_stand_capture("viser button")
        viser_view = ViserSimView(cfg.urdf_path,
                                  teleop=teleop if args.pico == "joystick" else None,
                                  on_capture=on_capture)
        print(f"viser: http://localhost:{viser_view.port}  (drive sliders in the sidebar)")
        if plugin is not None and hasattr(plugin, "attach_viser"):
            plugin.attach_viser(viser_view.server, teleop)

    import time as _time
    t_wall = _time.perf_counter()

    qlog = []
    errs, heights, tilts, fell = [], [], [], False
    interrupted = False
    rows = {
        "time_s": [],
        "mode": [],
        "axes": [],
        "nav_movement": [],
        "nav_facing": [],
        "nav_speed": [],
        "q": [],
        "q_true": [],
        "q_measured": [],
        "qd": [],
        "target_q": [],
        "pd_equilibrium_q_true": [],
        "plant_applied_target_q_true": [],
        "action": [],
        "actuator_force": [],
        "base_position": [],
        "base_quat_wxyz": [],
        "projected_gravity": [],
        "encoder_inference_ms": [],
        "decoder_inference_ms": [],
        "policy_total_ms": [],
        "planner_inference_ms": [],
        "planner_total_ms": [],
        "control_work_ms": [],
        "control_started": [],
        "motion_set": [],
        "planner_replan_count": [],
        "reference_q_il": [],
        "reference_dq_il": [],
        "reference_root_position": [],
        "reference_root_z": [],
        "idle_hold_phase": [],
        "stand_capture_pending": [],
        "stand_capture_count": [],
        "left_foot_contact": [],
        "right_foot_contact": [],
        "left_foot_contact_count": [],
        "right_foot_contact_count": [],
        "left_foot_position": [],
        "right_foot_position": [],
        "left_foot_corner_z": [],
        "right_foot_corner_z": [],
        "sonic_action_il_raw": [],
        "sonic_missing_action_raw": [],
        "sonic_missing_proprio_q": [],
        "sonic_missing_proprio_dq": [],
        "sonic_reference_q_il_adapted": [],
        "sonic_reference_dq_il_adapted": [],
    }
    recording_enabled = args.record_dir is not None
    handover_origin = (
        d.qpos[qadr].astype(np.float32)
        + encoder_position_offset
    )
    remote_activation_step = None
    max_plant_delay_steps = int(np.max(args.plant_target_delay_steps))
    initial_plant_target = (
        d.qpos[qadr].astype(np.float32).copy()
    )
    plant_target_history = deque(
        (initial_plant_target.copy() for _ in range(max_plant_delay_steps + 1)),
        maxlen=max_plant_delay_steps + 1,
    )
    try:
        for step in range(n_steps):
            work_start = _time.perf_counter()
            if viewer is not None and not viewer.is_running():
                print("viewer closed")
                break
            if (
                qlog
                and teleop is not None
                and getattr(teleop, "stop_requested", False)
            ):
                print("Unitree Select received; stopping MuJoCo and saving")
                break
            if (
                teleop is not None
                and args.pico == "joystick"
                and args.axes_from_step is not None
                and step == args.axes_from_step
            ):
                teleop.set_axes(*scheduled_axes)
            if (
                teleop is not None
                and args.pico == "joystick"
                and args.axes_until is not None
                and step == args.axes_until
            ):
                teleop.set_axes(0, 0, 0)
            if args.capture_at_step is not None and step == args.capture_at_step:
                ref.request_stable_stand_capture(
                    f"diagnostic step {args.capture_at_step}"
                )
            q_true = d.qpos[qadr].astype(np.float32)
            q = q_true + encoder_position_offset
            qd = d.qvel[vadr].astype(np.float32)
            base_q = d.qpos[3:7].astype(np.float32)              # pelvis wxyz
            ang_vel = d.qvel[3:6].astype(np.float32)             # base-local
            proj_g = gravity_orientation(base_q)

            # SONIC builds its own observation from RAW state; this is exactly the
            # call sonic/deploy.py makes.
            left_foot = _foot_diagnostics("left")
            right_foot = _foot_diagnostics("right")
            action = runner.compute_action(
                dof_pos=q.astype(np.float64),
                dof_vel=qd.astype(np.float64),
                base_quat_wxyz=base_q.astype(np.float64),
                base_ang_vel_body=ang_vel.astype(np.float64),  # qvel[3:6] is body frame
                foot_contact=(bool(left_foot[0]), bool(right_foot[0])),
                foot_z=(
                    float(np.nanmin(left_foot[3])),
                    float(np.nanmin(right_foot[3])),
                ),
            )

            if args.trace and step < 40 and ref_dof is not None:
                de = float(np.abs(q - ref_dof[min(ref.frame, len(ref_dof) - 1)]).mean())
                print(f"  t{step:3d} h={d.qpos[2]:.3f} gz={proj_g[2]:+.2f} "
                      f"gx={proj_g[0]:+.2f} gy={proj_g[1]:+.2f} dof_err={de:.3f}", flush=True)

            waiting_for_remote_start = bool(
                args.remote and not getattr(teleop, "control_started", False)
            )
            target = default if args.zero_action else action * action_scale + default
            if waiting_for_remote_start:
                target = default.copy()
            elif args.blend:
                if args.remote:
                    if remote_activation_step is None:
                        remote_activation_step = step
                        handover_origin = q.copy()
                    blend_step = step - remote_activation_step
                else:
                    blend_step = step
                if blend_step < 50:
                    phase = blend_step / 49.0
                    alpha = 0.5 - 0.5 * np.cos(np.pi * phase)
                    target = alpha * target + (1.0 - alpha) * handover_origin
            # Match the released controller: apply the finite policy target
            # directly. Joint limits remain diagnostics, not command shaping.
            # Unitree's low-level servo compares the target with its encoder
            # reading. If measured=true+offset, the equivalent physical torque
            # is produced in MuJoCo by shifting the actuator equilibrium by
            # -offset, since MuJoCo itself sees the true joint angle.
            pd_equilibrium_q_true = target - encoder_position_offset
            plant_target_history.append(
                np.asarray(pd_equilibrium_q_true, np.float32).copy()
            )
            plant_applied_target_q_true = np.empty_like(pd_equilibrium_q_true)
            history = list(plant_target_history)
            for joint_index, delay_steps in enumerate(
                args.plant_target_delay_steps
            ):
                plant_applied_target_q_true[joint_index] = history[
                    -1 - int(delay_steps)
                ][joint_index]
            d.ctrl[pd_act] = plant_applied_target_q_true
            if len(hold_act):
                d.ctrl[hold_act] = hold_tgt
            if plugin is not None and hasattr(plugin, "pre_physics"):
                plugin.pre_physics(step)
            for _ in range(decim):
                mujoco.mj_step(m, d)

            if teleop is not None:
                if args.remote:
                    teleop.update_remote(remote)
                teleop.tick(args.control_dt)

            q_true_after = d.qpos[qadr].astype(np.float32).copy()
            q_after = (
                q_true_after
                + encoder_position_offset
            )
            qd_after = d.qvel[vadr].astype(np.float32).copy()
            base_after = d.qpos[3:7].astype(np.float32).copy()
            projected_after = gravity_orientation(base_after)
            if recording_enabled:
                if teleop is not None:
                    nav = teleop.nav_command()
                    mode = int(teleop.mode())
                    axes = np.array([teleop.lx, teleop.ly, teleop.rx], np.float32) \
                        if args.pico == "joystick" else np.full(3, np.nan, np.float32)
                else:
                    nav = {
                        "movement": np.full(3, np.nan, np.float32),
                        "facing": np.full(3, np.nan, np.float32),
                        "speed": np.nan,
                    }
                    mode = -1
                    axes = np.full(3, np.nan, np.float32)
                rows["time_s"].append((step + 1) * args.control_dt)
                rows["mode"].append(mode)
                rows["axes"].append(axes)
                rows["nav_movement"].append(np.asarray(nav["movement"], np.float32))
                rows["nav_facing"].append(np.asarray(nav["facing"], np.float32))
                rows["nav_speed"].append(float(nav["speed"]))
                rows["q"].append(q_after)
                rows["q_true"].append(q_true_after)
                rows["q_measured"].append(q_after)
                rows["qd"].append(qd_after)
                rows["target_q"].append(np.asarray(target, np.float32).copy())
                rows["pd_equilibrium_q_true"].append(
                    np.asarray(pd_equilibrium_q_true, np.float32).copy()
                )
                rows["plant_applied_target_q_true"].append(
                    np.asarray(
                        plant_applied_target_q_true, np.float32
                    ).copy()
                )
                rows["action"].append(np.asarray(action, np.float32).copy())
                rows["actuator_force"].append(
                    np.asarray(d.actuator_force[pd_act], np.float32).copy()
                )
                rows["base_position"].append(d.qpos[:3].astype(np.float32).copy())
                rows["base_quat_wxyz"].append(base_after)
                rows["projected_gravity"].append(projected_after)
                rows["encoder_inference_ms"].append(float(runner.last_encoder_ms))
                rows["decoder_inference_ms"].append(float(runner.last_decoder_ms))
                rows["policy_total_ms"].append(float(runner.last_policy_ms))
                planner = getattr(ref, "planner", None)
                rows["planner_inference_ms"].append(
                    float(getattr(planner, "last_inference_ms", np.nan))
                )
                rows["planner_total_ms"].append(
                    float(getattr(planner, "last_total_ms", np.nan))
                )
                rows["control_work_ms"].append(
                    (_time.perf_counter() - work_start) * 1e3
                )
                runner_diagnostics = runner.diagnostic_fields()
                for name in (
                    "sonic_action_il_raw",
                    "sonic_missing_action_raw",
                    "sonic_missing_proprio_q",
                    "sonic_missing_proprio_dq",
                    "sonic_reference_q_il_adapted",
                    "sonic_reference_dq_il_adapted",
                ):
                    rows[name].append(
                        np.asarray(runner_diagnostics[name]).copy()
                    )
                rows["control_started"].append(
                    bool(getattr(teleop, "control_started", True))
                )
                rows["motion_set"].append(
                    int(getattr(teleop, "motion_set_index", -1))
                )
                rows["planner_replan_count"].append(
                    int(getattr(ref, "replan_count", 0))
                )
                reference_diagnostics = getattr(
                    ref, "reference_diagnostics", None
                )
                diagnostics = (
                    reference_diagnostics()
                    if callable(reference_diagnostics)
                    else {
                        "reference_q_il": np.full(29, np.nan, np.float32),
                        "reference_dq_il": np.full(29, np.nan, np.float32),
                        "reference_root_position": np.full(3, np.nan, np.float32),
                        "reference_root_z": np.float32(np.nan),
                        "idle_hold_phase": np.int8(0),
                        "stand_capture_pending": np.bool_(False),
                        "stand_capture_count": np.int32(0),
                    }
                )
                for name in (
                    "reference_q_il",
                    "reference_dq_il",
                    "reference_root_position",
                    "reference_root_z",
                    "idle_hold_phase",
                    "stand_capture_pending",
                    "stand_capture_count",
                ):
                    rows[name].append(np.asarray(diagnostics[name]).copy())
                for side in ("left", "right"):
                    contact, contact_count, foot_position, corner_z = (
                        _foot_diagnostics(side)
                    )
                    rows[f"{side}_foot_contact"].append(contact)
                    rows[f"{side}_foot_contact_count"].append(contact_count)
                    rows[f"{side}_foot_position"].append(foot_position)
                    rows[f"{side}_foot_corner_z"].append(corner_z)

            if ref_dof is not None:
                f = min(ref.frame, len(ref_dof) - 1)
                errs.append(np.abs(q_true_after - ref_dof[f]).mean())
            heights.append(float(d.qpos[2]))
            qlog.append(q_true_after)
            tilts.append(
                float(np.degrees(np.arccos(np.clip(-projected_after[2], -1, 1))))
            )
            if d.qpos[2] < 0.4:
                fell = True
                print(f"FELL at step {step}/{n_steps} (height {d.qpos[2]:.2f})")
                break
            if viewer is not None:
                if teleop is not None:
                    draw_command(viewer, d, teleop)
                viewer.sync()
            if viser_view is not None and step % 2 == 0:      # 25 Hz mirror
                viser_view.update(d.qpos[:3], d.qpos[3:7], d.qpos[qadr])
            if plugin is not None:
                plugin.on_control_step(step)
            if (viewer is not None or viser_view is not None) and args.realtime:
                t_wall += args.control_dt
                lag = t_wall - _time.perf_counter()
                if lag > 0:
                    _time.sleep(lag)
            if not (args.remote and not getattr(teleop, "control_started", False)):
                ref.step()
    except KeyboardInterrupt:
        interrupted = True
        print("\nCtrl+C received; saving the completed simulation rows")
    finally:
        if viewer is not None:
            viewer.close()
    errs, heights, tilts = np.array(errs), np.array(heights), np.array(tilts)
    if not qlog:
        raise RuntimeError("simulation ended before a state row was recorded")
    Q = np.degrees(np.ptp(np.array(qlog), axis=0))
    print(f"\nsteps {len(heights)}/{n_steps}" + ("  FELL" if fell else ""))
    if errs.size:
        print(f"  track err  {np.degrees(errs.mean()):.1f} deg")
    print(f"  travel     {np.linalg.norm(d.qpos[:2]):.2f} m")
    print(f"  arm/leg    {Q[13:].max():.0f} / {Q[:12].max():.0f} deg")
    print(f"  height     {heights.min():.3f} .. {heights.max():.3f} m")
    print(f"  tilt       {tilts.max():.1f} deg")
    if args.record_dir is not None:
        _write_recording(
            args.record_dir.expanduser().resolve(),
            args=args,
            cfg=cfg,
            profile=profile,
            dof_names=dof_names,
            kp=kp,
            kd=kd,
            action_scale=action_scale,
            runner=runner,
            rows=rows,
            fell=fell,
            interrupted=interrupted,
        )
    if plugin is not None and args.record_dir is not None:
        plugin.finalize(args.record_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
