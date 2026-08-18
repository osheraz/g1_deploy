"""Official-style live PICO/planner reference window for SONIC.

The PICO wire protocol supplies planner-world movement/facing vectors and,
only in VR3PT mode, calibrated pelvis-local [left wrist, right wrist, head]
targets. Encoder mode follows fresh ``vr_position`` presence exactly as the
official ZMQ manager does; orientation is optional and uses the official
comfort-quaternion fallback.
"""
from __future__ import annotations

import atexit
import os
import threading
import time
from collections import deque

import numpy as np

from policy.sonic import robot_g1
from policy.sonic.model_profile import get_model_profile
from policy.sonic.planner_onnx import SonicKinematicPlanner
from policy.sonic.robot_profile import get_robot_profile
from policy.sonic.rotation_np import heading_quat, quat_inv, quat_mul


NUM_BODIES = len(robot_g1.BODY_NAMES)
CONTROL_FPS = 50.0
PLANNER_TICK_CONTROL_STEPS = 5       # 10 Hz planner on a 50 Hz control loop
PLANNER_TIMEOUT_S = 1.0

STATIC_MODES = frozenset((0, 4, 5, 6, 7, 9))
BOXING_MODES = frozenset((11, 12, 13, 15, 16))
RUN_MODE = 3
CRAWLING_MODE = 8
LOWER_BODY_IL_IDX = np.array(
    [0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18], dtype=np.int64
)
IDLE_ADAPT_TRIGGER = 0.10
IDLE_ADAPT_STOP = 0.05
IDLE_RECOVER_TRIGGER = 0.045
IDLE_CENTERING_STEPS = 75  # 1.5 s at the 50 Hz tracker rate
STAND_CAPTURE_STEPS = 50   # one continuous second at the 50 Hz tracker rate
STAND_CAPTURE_MAX_QD_RMS = 0.05
STAND_CAPTURE_MAX_JOINT_RANGE = 0.05
STAND_CAPTURE_MAX_TILT_DEG = 8.0
STAND_CAPTURE_MAX_BASE_SPEED_MPS = 0.05
STAND_CAPTURE_MAX_BASE_DRIFT_M = 0.015
STAND_CAPTURE_MAX_CORNER_Z_M = 0.015
STAND_CAPTURE_MAX_FOOT_CORNER_RANGE_M = 0.010
STAND_CAPTURE_MIN_COM_MARGIN_M = 0.015


def _normalize_direction(value) -> np.ndarray:
    v = np.asarray(value, dtype=np.float32).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else np.zeros(3, dtype=np.float32)


def _same_vector(a, b, eps_deg: float = 5.0) -> bool:
    """Directions within the official 5-degree replan epsilon."""
    a2 = np.asarray(a, np.float64)[:2]
    b2 = np.asarray(b, np.float64)[:2]
    na, nb = np.linalg.norm(a2), np.linalg.norm(b2)
    if na < 1e-6 and nb < 1e-6:
        return True
    if na < 1e-6 or nb < 1e-6:
        return False
    dot = float(np.clip(np.dot(a2 / na, b2 / nb), -1.0, 1.0))
    return np.degrees(np.arccos(dot)) <= eps_deg


class PicoReferenceWindow:
    """Planner trajectory plus optional calibrated VR three-point targets."""

    def __init__(
        self,
        config,
        pico_client,
        robot=None,
        threaded=True,
        gated=True,
        force_g1=False,
        force_smpl=False,
        idle_hold=True,
        replan_period_s=None,
    ):
        self.idle_hold = bool(idle_hold)
        self._idle_holding = False
        self._idle_settling = False
        self._idle_centering = False
        self._idle_hold_q_il = None
        self._idle_hold_rootq = None
        self._idle_hold_rootz = None
        self._stable_stand_q_il = None
        self._stable_stand_rootz = None
        self._idle_centering_start_q_il = None
        self._idle_centering_rootq = None
        self._idle_centering_start_rootz = None
        self._idle_centering_step = 0
        self._turn_plan_active = False
        self._stand_capture_requested = False
        self._stand_capture_samples = deque(maxlen=STAND_CAPTURE_STEPS)
        self._stand_capture_wait_ticks = 0
        self.stand_capture_count = 0
        self._last_reference_q_il = np.zeros(robot_g1.NUM_DOF, np.float32)
        self._last_reference_dq_il = np.zeros(robot_g1.NUM_DOF, np.float32)
        self._last_reference_root_pos = np.zeros(3, np.float32)
        self._last_reference_root_z = np.float32(0.0)
        planner_onnx = config.sonic_planner_onnx
        if not planner_onnx:
            raise SystemExit("PICO mode requires `sonic_planner_onnx` in the config")

        self.client = pico_client
        self.robot = robot
        self.config = config
        self.force_g1 = bool(force_g1)
        self.force_smpl = bool(force_smpl)
        if self.force_g1 and self.force_smpl:
            raise ValueError("G1 and SMPL encoder modes cannot both be forced")
        self.model_profile = get_model_profile(config.sonic_model_variant)
        self.robot_profile = get_robot_profile(config.sonic_robot_profile)
        self.planner = SonicKinematicPlanner(
            planner_onnx,
            version=getattr(config, "sonic_planner_version", None),
            default_height=getattr(config, "sonic_planner_default_height", 0.788740),
            random_seed=getattr(config, "sonic_planner_random_seed", 1234),
            backend=getattr(config, "sonic_planner_backend", "auto"),
            cache_dir=getattr(config, "sonic_engine_cache_dir", None),
            fp16=bool(getattr(config, "sonic_tensorrt_fp16", False)),
            allow_tf32=bool(getattr(config, "sonic_tensorrt_tf32", False)),
        )

        self.initialized = False
        self.done = False
        self._vr_active = False
        self._smpl_active = False
        self._vr_input_present = False
        self._stale = False
        self._stale_logged = False
        self._last_world_nav = {
            "movement": np.zeros(3, np.float32),
            "facing": np.array([1.0, 0.0, 0.0], np.float32),
            "speed": -1.0,
            "height": -1.0,
        }
        self._last_cmd = (0, self._last_world_nav.copy())

        # Official deployment uses a 10 Hz planner thread and a 50 Hz control
        # thread.  Synchronous mode is retained for faster-than-wall-clock sim.
        self.threaded = bool(threaded)
        self.gated = bool(gated)
        self.default_replan_period_s = float(
            getattr(config, "sonic_planner_replan_interval_s", 0.1)
            if replan_period_s is None else replan_period_s
        )
        if not 0.1 <= self.default_replan_period_s <= 10.0:
            raise ValueError("planner replan period must be in [0.1, 10.0] seconds")
        self._ticks = 0
        self._last_planner_tick = -PLANNER_TICK_CONTROL_STEPS
        self._replan_interval_s = 0.0
        self._lock = threading.Lock()
        self._replan_thread = None
        self._stop = False
        self._heading_alignment = np.array([1.0, 0.0, 0.0, 0.0])
        self._idle_readapt_original = np.zeros(robot_g1.NUM_DOF, np.float64)
        self._idle_readapt_stored = False
        self._idle_readapt_state = "idle"
        self._fatal_error: RuntimeError | None = None
        self.replan_count = 0
        atexit.register(self.stop)
        profile = (
            "PLANNER/G1-only" if self.force_g1
            else "STREAMED-MOTION/SMPL-only" if self.force_smpl
            else "dynamic PICO"
        )
        print(
            f"SONIC reference: official-style PICO planner {self.planner.model_path} "
            f"({profile})"
        )

    @property
    def sonic_mode(self) -> str:
        if self.force_smpl or (self._smpl_active and not self.force_g1):
            return "smpl"
        return "teleop" if self._vr_active and not self.force_g1 else "g1"

    def initialize_from_robot(self, base_quat_wxyz, dof_pos) -> None:
        """Initialize planner joints and official heading alignment from hardware."""
        q_il = self.robot_profile.physical_to_isaaclab(
            np.asarray(dof_pos, dtype=np.float32), fill=0.0
        )
        base_q = np.asarray(base_quat_wxyz, dtype=np.float64).reshape(4)
        self.planner.initialize(base_q, robot_g1.il_to_mj(q_il))
        # The initial measured frame is not automatically assumed to be the
        # "best" stand. Use the configured default as a deterministic bootstrap
        # and let the stability-gated capture command replace it with a robust
        # one-second median when the operator judges the robot ready.
        nominal_physical = np.asarray(
            self.config.physical_default_angles, dtype=np.float32
        )
        nominal_il = self.robot_profile.physical_to_isaaclab(
            nominal_physical, fill=0.0
        )
        self._stable_stand_q_il = np.asarray(nominal_il, np.float64).copy()
        self._stable_stand_rootz = float(self.planner.current_motion.root_pos[0, 2])

        # Keep the planner's native 29-DoF trajectory intact: it is recurrent
        # model context, not measured 23-DoF proprioception.  The V2 IDLE plan
        # contains a learned stance transition, so the tracker receives a
        # separate measured-pose hold while the planner retains coherent state.
        if self.idle_hold or os.environ.get("SONIC_FLAT_IDLE"):
            self._capture_idle_hold(q_il=nominal_il)
            print(
                "[IDLE HOLD] Using configured default stance as the bootstrap; "
                "press Numpad Enter or F1+F3 to capture a verified steady stance"
            )
            print(
                "[PLANNER] IDLE stable stand reached: stable-pose hold active "
                "(configured bootstrap)"
            )

        # C++ generates in a normalized +X planner frame and maps its reference
        # heading onto the measured base heading in the encoder gatherer.
        initial_ref = self.planner.current_motion.root_quat_wxyz[0]
        self._heading_alignment = quat_mul(
            heading_quat(base_q), quat_inv(heading_quat(initial_ref))
        )
        base_heading = heading_quat(base_q)
        yaw_deg = np.degrees(2.0 * np.arctan2(base_heading[3], base_heading[0]))
        self.initialized = True
        print(
            "SONIC planner calibrated from measured robot state: "
            f"{self.robot_profile.physical_dof} physical joints -> 29 slots, "
            f"pelvis heading={yaw_deg:+.1f}deg, "
            f"encoder={'G1 (forced)' if self.force_g1 else 'message-selected'}"
        )
        if self.threaded and self._replan_thread is None:
            self._replan_thread = threading.Thread(
                target=self._replan_loop, name="sonic-planner", daemon=True
            )
            self._replan_thread.start()

    def reset(self) -> None:
        self.initialized = False
        self._vr_active = False
        self._smpl_active = False
        self._vr_input_present = False
        self._stale = False
        self._stand_capture_requested = False
        self._stand_capture_samples.clear()

    def step(self) -> None:
        """Advance the installed 50 Hz trajectory after one policy command."""
        if self.initialized:
            consume_capture = getattr(
                self.client, "consume_idle_capture_request", None
            )
            if callable(consume_capture) and consume_capture():
                self.request_stable_stand_capture("F1+F3")
            self._ticks += 1
            with self._lock:
                self.planner.advance(1)
                if self._idle_centering:
                    self._idle_centering_step += 1
                    if self._idle_centering_step >= IDLE_CENTERING_STEPS:
                        self._finish_idle_centering()
                self._idle_readapt_at_horizon()
            self._update_stable_stand_capture()

    def request_stable_stand_capture(self, source: str = "operator") -> bool:
        """Request a stable measured-joint posture for future IDLE returns."""
        if not self.idle_hold:
            print(
                f"[IDLE CAPTURE] Ignored {source} request: IDLE hold is disabled"
            )
            return False
        self._stand_capture_requested = True
        self._stand_capture_samples.clear()
        self._stand_capture_wait_ticks = 0
        print(
            f"[IDLE CAPTURE] {source}: keep both sticks neutral and stand still; "
            "capturing requires one continuous stable second"
        )
        return True

    def _capture_input_is_stationary(self) -> bool:
        stationary = getattr(self.client, "idle_capture_stationary", None)
        if callable(stationary):
            return bool(stationary())
        mode = self.client.mode()
        nav = self.client.nav_command()
        return (
            mode is not None
            and int(mode) == 0
            and nav is not None
            and float(np.linalg.norm(nav["movement"])) < 1e-6
        )

    def _robot_capture_sample(self):
        if self.robot is None or not hasattr(self.robot, "qj"):
            return None
        q = np.asarray(self.robot.qj, np.float64).reshape(
            self.robot_profile.physical_dof
        )
        dq_value = getattr(self.robot, "dqj", None)
        if dq_value is None:
            return None
        dq = np.asarray(dq_value, np.float64).reshape(
            self.robot_profile.physical_dof
        )
        base_quat = getattr(self.robot, "base_quat_wxyz", None)
        if base_quat is None:
            get_imu = getattr(self.robot, "get_imu", None)
            if callable(get_imu):
                base_quat = get_imu()[0]
        if base_quat is None:
            return None
        quat = np.asarray(base_quat, np.float64).reshape(4)
        quat_norm = float(np.linalg.norm(quat))
        if not np.isfinite(quat_norm) or quat_norm < 1e-8:
            return None
        quat /= quat_norm
        # World-up component of the pelvis local z-axis. Its arccos is the
        # roll/pitch tilt magnitude, independent of yaw.
        up_z = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
        tilt_deg = float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))
        root_z = getattr(self.robot, "root_z", np.nan)
        support_diagnostics = getattr(
            self.robot, "stand_support_diagnostics", None
        )
        support = support_diagnostics() if callable(support_diagnostics) else None
        return {
            "q": q.copy(),
            "dq": dq.copy(),
            "tilt_deg": tilt_deg,
            "root_z": float(root_z),
            "support": support,
        }

    def _update_stable_stand_capture(self) -> None:
        if not self._stand_capture_requested:
            return
        self._stand_capture_wait_ticks += 1
        if not self._capture_input_is_stationary() or not self._idle_holding:
            self._stand_capture_samples.clear()
            if self._stand_capture_wait_ticks % STAND_CAPTURE_STEPS == 0:
                print(
                    "[IDLE CAPTURE] Waiting: release both sticks and wait for "
                    "the stable IDLE hold phase"
                )
            return
        sample = self._robot_capture_sample()
        if sample is None:
            self._stand_capture_requested = False
            print("[IDLE CAPTURE] Failed: measured q/dq or pelvis orientation unavailable")
            return
        self._stand_capture_samples.append(sample)
        if len(self._stand_capture_samples) < STAND_CAPTURE_STEPS:
            return

        q = np.stack([sample["q"] for sample in self._stand_capture_samples])
        dq = np.stack([sample["dq"] for sample in self._stand_capture_samples])
        tilt = np.asarray(
            [sample["tilt_deg"] for sample in self._stand_capture_samples]
        )
        qd_rms = float(np.sqrt(np.mean(dq ** 2)))
        joint_range = float(np.max(np.ptp(q, axis=0)))
        tilt_max = float(np.max(tilt))
        motion_stable = (
            qd_rms <= STAND_CAPTURE_MAX_QD_RMS
            and joint_range <= STAND_CAPTURE_MAX_JOINT_RANGE
            and tilt_max <= STAND_CAPTURE_MAX_TILT_DEG
        )
        support = [sample["support"] for sample in self._stand_capture_samples]
        support_observable = all(item is not None for item in support)
        support_partially_observed = any(item is not None for item in support)
        if support_observable:
            full_support = all(
                item["left_supported"] and item["right_supported"]
                for item in support
            )
            base_speed_max = float(max(
                item["base_planar_speed_mps"] for item in support
            ))
            base_xy = np.stack([item["base_xy"] for item in support])
            base_drift = float(np.linalg.norm(np.ptp(base_xy, axis=0)))
            corner_z_max = float(max(item["max_corner_z_m"] for item in support))
            foot_corner_range_max = float(max(
                item["max_foot_corner_range_m"] for item in support
            ))
            com_margin_min = float(min(
                item["com_support_margin_m"] for item in support
            ))
            support_stable = (
                full_support
                and base_speed_max <= STAND_CAPTURE_MAX_BASE_SPEED_MPS
                and base_drift <= STAND_CAPTURE_MAX_BASE_DRIFT_M
                and corner_z_max <= STAND_CAPTURE_MAX_CORNER_Z_M
                and foot_corner_range_max <= STAND_CAPTURE_MAX_FOOT_CORNER_RANGE_M
                and com_margin_min >= STAND_CAPTURE_MIN_COM_MARGIN_M
            )
        elif not support_partially_observed:
            # The physical HG LowState contains joint/IMU state but no validated
            # bilateral contact, root translation, or support polygon. Mirror
            # the measured-pose C++ capture protocol while making it more robust:
            # qualify a full second of measured joint and pelvis stability, and
            # leave sole contact as an explicit operator-observed condition.
            full_support = None
            base_speed_max = base_drift = corner_z_max = np.nan
            foot_corner_range_max = com_margin_min = np.nan
            support_stable = True
        else:
            # A simulator should not lose support observability midway through
            # a qualification window. Do not silently downgrade a partial
            # diagnostic stream to the physical operator-qualified path.
            full_support = False
            base_speed_max = base_drift = corner_z_max = np.nan
            foot_corner_range_max = com_margin_min = np.nan
            support_stable = False
        stable = motion_stable and support_stable
        if not stable:
            if self._stand_capture_wait_ticks % STAND_CAPTURE_STEPS == 0:
                details = (
                    f"dq_rms={qd_rms:.3f}rad/s, "
                    f"joint_range={joint_range:.3f}rad, "
                    f"tilt_max={tilt_max:.1f}deg"
                )
                if support_observable:
                    details += (
                        f", full_support={full_support}, "
                        f"base_speed={base_speed_max:.3f}m/s, "
                        f"base_drift={base_drift:.3f}m, "
                        f"corner_z={corner_z_max:.3f}m, "
                        f"foot_range={foot_corner_range_max:.3f}m, "
                        f"com_margin={com_margin_min:.3f}m"
                    )
                print(f"[IDLE CAPTURE] Waiting for stability: {details}")
            return

        # Match the real-robot C++ protocol: the robust median of the measured
        # physical joints becomes the tracker-only IDLE reference. The native
        # 29-DoF planner trajectory and action history remain untouched, and
        # unavailable physical joints retain their neutral zero slots.
        measured_q = np.median(q, axis=0)
        measured_q_il = self.robot_profile.physical_to_isaaclab(
            measured_q, fill=0.0
        )
        self._stable_stand_q_il = np.asarray(measured_q_il, np.float64).copy()
        self._idle_hold_q_il = self._stable_stand_q_il.copy()
        measured_root_z = np.asarray(
            [sample["root_z"] for sample in self._stand_capture_samples],
            np.float64,
        )
        if np.isfinite(measured_root_z).all():
            self._stable_stand_rootz = float(np.median(measured_root_z))
            self._idle_hold_rootz = self._stable_stand_rootz
        self._stand_capture_requested = False
        self._stand_capture_samples.clear()
        self.stand_capture_count += 1
        if support_observable:
            print(
                "[IDLE CAPTURE] Saved support-qualified measured standing pose; "
                "later planner-IDLE returns will center to this pose: "
                f"dq_rms={qd_rms:.3f}rad/s, "
                f"joint_range={joint_range:.3f}rad, "
                f"tilt_max={tilt_max:.1f}deg, "
                f"base_drift={base_drift:.3f}m, "
                f"com_margin={com_margin_min:.3f}m"
            )
        else:
            print(
                "[IDLE CAPTURE] Saved operator-qualified measured standing pose; "
                "later planner-IDLE returns will center to this pose: "
                f"dq_rms={qd_rms:.3f}rad/s, "
                f"joint_range={joint_range:.3f}rad, "
                f"tilt_max={tilt_max:.1f}deg. "
                "Physical contact is not observable; capture assumes the operator "
                "confirmed both soles were planted."
            )

    def _idle_readapt_at_horizon(self) -> None:
        """Mirror the official lower-body end-of-IDLE adaptation state machine."""
        if self.idle_hold:
            # Flat IDLE is an explicit tracker-reference override.  Do not also
            # mutate the planner's last frame with the legacy readaptation path.
            return
        motion = self.planner.current_motion
        if (
            self.robot is None
            or motion is None
            or self.planner.current_frame != motion.timesteps - 1
            or self._last_cmd[0] != 0
        ):
            return

        measured_il = self.robot_profile.physical_to_isaaclab(
            np.asarray(self.robot.qj, np.float64), fill=0.0
        )
        target = motion.joint_pos_il[-1]
        if not self._idle_readapt_stored:
            self._idle_readapt_original[LOWER_BODY_IL_IDX] = target[LOWER_BODY_IL_IDX]
            self._idle_readapt_stored = True
            self._idle_readapt_state = "idle"

        avg_error = float(np.mean(np.abs(
            target[LOWER_BODY_IL_IDX] - measured_il[LOWER_BODY_IL_IDX]
        )))
        if self._idle_readapt_state == "idle":
            if avg_error > IDLE_ADAPT_TRIGGER:
                self._idle_readapt_state = "adapting"
            elif avg_error < IDLE_RECOVER_TRIGGER:
                self._idle_readapt_state = "recovering"
        elif self._idle_readapt_state == "adapting":
            if avg_error < IDLE_ADAPT_STOP:
                self._idle_readapt_state = "idle"
        elif self._idle_readapt_state == "recovering":
            if avg_error > IDLE_ADAPT_TRIGGER:
                self._idle_readapt_state = "adapting"

        if self._idle_readapt_state == "adapting":
            target[LOWER_BODY_IL_IDX] = (
                0.98 * target[LOWER_BODY_IL_IDX]
                + 0.02 * measured_il[LOWER_BODY_IL_IDX]
            )
        elif self._idle_readapt_state == "recovering":
            target[LOWER_BODY_IL_IDX] = (
                0.98 * target[LOWER_BODY_IL_IDX]
                + 0.02 * self._idle_readapt_original[LOWER_BODY_IL_IDX]
            )

    def stop(self) -> None:
        self._stop = True
        if self._replan_thread is not None:
            self._replan_thread.join(timeout=1.0)
        stop = getattr(self.client, "stop", None)
        if callable(stop):
            stop()

    def _capture_idle_hold(self, *, q_il=None) -> None:
        """Capture a flat tracker reference without modifying planner context."""
        m = self.planner.current_motion
        cur = min(self.planner.current_frame, m.timesteps - 1)
        if q_il is not None:
            hold_il = np.asarray(q_il, np.float64).reshape(robot_g1.NUM_DOF)
        elif self.robot is not None:
            hold_il = self.robot_profile.physical_to_isaaclab(
                np.asarray(self.robot.qj, np.float64), fill=0.0
            )
        else:
            hold_il = m.joint_pos_il[cur].copy()
        self._idle_hold_q_il = np.asarray(hold_il, np.float64).copy()
        self._idle_hold_rootq = np.asarray(m.root_quat_wxyz[cur], np.float64).copy()
        self._idle_hold_rootz = float(m.root_pos[cur, 2])
        self._idle_holding = True
        self._idle_settling = False
        self._idle_centering = False
        self._turn_plan_active = False

    def _begin_idle_centering(self) -> None:
        """Move smoothly from the planner's terminal stance to stable stand."""
        if self._stable_stand_q_il is None:
            self._capture_idle_hold()
            return
        m = self.planner.current_motion
        cur = min(self.planner.current_frame, m.timesteps - 1)
        if self.robot is not None:
            start = self.robot_profile.physical_to_isaaclab(
                np.asarray(self.robot.qj, np.float64), fill=0.0
            )
        else:
            start = m.joint_pos_il[cur].copy()
        self._idle_centering_start_q_il = np.asarray(start, np.float64).copy()
        self._idle_centering_rootq = np.asarray(
            m.root_quat_wxyz[cur], np.float64
        ).copy()
        self._idle_centering_start_rootz = float(m.root_pos[cur, 2])
        self._idle_centering_step = 0
        self._idle_holding = False
        self._idle_settling = False
        self._idle_centering = True
        self._turn_plan_active = False
        print(
            "[PLANNER] IDLE transition complete: smoothly centering both feet "
            "to the configured/captured stable stand"
        )

    def _finish_idle_centering(self) -> None:
        self._idle_hold_q_il = self._stable_stand_q_il.copy()
        self._idle_hold_rootq = self._idle_centering_rootq.copy()
        self._idle_hold_rootz = float(self._stable_stand_rootz)
        self._idle_holding = True
        self._idle_settling = False
        self._idle_centering = False
        self._turn_plan_active = False
        print("[PLANNER] IDLE stable stand reached: stable-pose hold active")

    def _apply_idle_hold_to_plan(self, plan: dict) -> dict:
        """Return a held tracker window while leaving the planner-owned data intact."""
        if getattr(self, "_idle_centering", False):
            held = dict(plan)
            offsets = (
                self._idle_centering_step
                + np.arange(np.asarray(plan["dof_fut"]).shape[0])
                * self.model_profile.g1_future_stride
            )
            phase = np.clip(offsets / IDLE_CENTERING_STEPS, 0.0, 1.0)
            alpha = 0.5 - 0.5 * np.cos(np.pi * phase)
            dalpha_dt = (
                0.5 * np.pi * CONTROL_FPS / IDLE_CENTERING_STEPS
                * np.sin(np.pi * phase)
            )
            delta = self._stable_stand_q_il - self._idle_centering_start_q_il
            held["dof_fut"] = (
                self._idle_centering_start_q_il[None]
                + alpha[:, None] * delta[None]
            )
            held["dvel_fut"] = dalpha_dt[:, None] * delta[None]
            held["rootq_fut"] = np.broadcast_to(
                self._idle_centering_rootq, np.asarray(plan["rootq_fut"]).shape
            ).copy()
            held["root_quat_cur"] = self._idle_centering_rootq.copy()
            rootz = (
                self._idle_centering_start_rootz
                + alpha * (self._stable_stand_rootz - self._idle_centering_start_rootz)
            )
            held["rootz_fut"] = rootz
            held["rootz_cur"] = np.asarray(rootz[0]).copy()
            return held
        if not self._idle_holding or self._idle_hold_q_il is None:
            return plan
        held = dict(plan)
        held["dof_fut"] = np.broadcast_to(
            self._idle_hold_q_il, np.asarray(plan["dof_fut"]).shape
        ).copy()
        held["dvel_fut"] = np.zeros_like(plan["dvel_fut"])
        if self._idle_hold_rootq is not None:
            held["rootq_fut"] = np.broadcast_to(
                self._idle_hold_rootq, np.asarray(plan["rootq_fut"]).shape
            ).copy()
            held["root_quat_cur"] = self._idle_hold_rootq.copy()
        if self._idle_hold_rootz is not None:
            held["rootz_fut"] = np.full_like(plan["rootz_fut"], self._idle_hold_rootz)
            held["rootz_cur"] = np.asarray(self._idle_hold_rootz).copy()
        return held

    def _replan_loop(self) -> None:
        while not self._stop:
            self._maybe_replan()
            time.sleep(0.002)

    def _client_age_s(self) -> float:
        age = getattr(self.client, "age_s", None)
        if callable(age):
            return float(age())
        return 0.0

    def _fresh_vr(self):
        if self._client_age_s() >= PLANNER_TIMEOUT_S:
            return None
        return self.client.vr_3point()

    def _fresh_smpl(self):
        get_reference = getattr(self.client, "smpl_reference", None)
        if not callable(get_reference):
            return None
        return get_reference(self.model_profile.smpl_future_frames)

    def _update_encoder_mode(self) -> None:
        self._vr_input_present = self._fresh_vr() is not None
        self._vr_active = self._vr_input_present and not self.force_g1

    def _to_planner_world(self, nav: dict) -> dict:
        """Convert only explicitly local stand-ins; real PICO is already world-frame."""
        movement = np.asarray(nav["movement"], np.float32).reshape(3)
        facing = np.asarray(nav["facing"], np.float32).reshape(3)
        if getattr(self.client, "navigation_frame", "world") == "local":
            with self._lock:
                q = self.planner.current_motion.root_quat_wxyz[
                    min(self.planner.current_frame, self.planner.current_motion.timesteps - 1)
                ]
            w, x, y, z = np.asarray(q, np.float64)
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            c, s = np.cos(yaw), np.sin(yaw)

            def rotate(v):
                return np.array(
                    [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]],
                    dtype=np.float32,
                )

            movement, facing = rotate(movement), rotate(facing)
        return {
            "movement": _normalize_direction(movement),
            "facing": _normalize_direction(facing),
            "speed": float(nav["speed"]),
            "height": float(nav["height"]),
        }

    def _current_command(self) -> tuple[int, dict]:
        age = self._client_age_s()
        if age >= PLANNER_TIMEOUT_S:
            self._stale = True
            if not self._stale_logged:
                print(f"[PICO] planner input stale ({age:.3f}s): transitioning to IDLE")
                self._stale_logged = True
            return 0, {
                "movement": np.zeros(3, np.float32),
                "facing": self._last_world_nav["facing"].copy(),
                "speed": -1.0,
                "height": -1.0,
            }

        mode = self.client.mode()
        nav = self.client.nav_command()
        if mode is None or nav is None:
            return self._last_cmd
        self._stale = False
        self._stale_logged = False
        world = self._to_planner_world(nav)
        if int(mode) in STATIC_MODES:
            world["speed"] = -1.0
        if int(mode) in (4, 5, 6, 7, 8, 14) and world["height"] < 0.2:
            world["height"] = 0.2
        self._last_world_nav = {k: v.copy() if isinstance(v, np.ndarray) else v
                                for k, v in world.items()}
        return int(mode), world

    def _period_for_mode(self, mode: int) -> float:
        if mode == RUN_MODE:
            return 0.1
        if mode == CRAWLING_MODE:
            return 0.2
        if mode in BOXING_MODES:
            return 1.0
        return self.default_replan_period_s

    def _should_replan(self, mode: int, nav: dict) -> bool:
        last_mode, last_nav = self._last_cmd
        period = self._period_for_mode(mode)
        self._replan_interval_s += 0.1
        time_to_replan = self._replan_interval_s + 1e-9 >= period
        if time_to_replan:
            self._replan_interval_s = 0.0

        if not self.gated:
            return True
        if mode != last_mode:
            return True
        if not _same_vector(nav["facing"], last_nav["facing"]):
            return True
        if nav["height"] != last_nav["height"]:
            return True
        if mode not in STATIC_MODES:
            # official planner_command_speed_epsilon; exact compare replans on
            # every stick wobble in the stick-scaled modes
            if abs(nav["speed"] - last_nav["speed"]) > 0.05:
                return True
            if not _same_vector(nav["movement"], last_nav["movement"]):
                return True
            if time_to_replan and nav["speed"] != 0.0:
                return True
        return False

    def _maybe_replan(self) -> None:
        if not self.initialized or self._fatal_error is not None:
            return
        if self._ticks - self._last_planner_tick < PLANNER_TICK_CONTROL_STEPS:
            return
        self._last_planner_tick = self._ticks
        mode, nav = self._current_command()

        # Startup IDLE is already held at the configured or operator-captured
        # standing posture. When locomotion stops, however, do not capture the
        # current mid-stride
        # posture.  Request the planner's IDLE transition, track it through its
        # horizon, and only then center to the known stable startup stance.
        # Facing changes likewise run the generated turn before re-holding.
        if self.idle_hold and mode == 0:
            last_mode, last_nav = self._last_cmd
            facing_changed = not _same_vector(
                nav["facing"], last_nav["facing"]
            )
            if not facing_changed:
                entering_idle = last_mode != 0
                with self._lock:
                    m = self.planner.current_motion
                    exhausted = self.planner.current_frame >= m.timesteps - 1
                    if (
                        not entering_idle
                        and exhausted
                        and not self._idle_holding
                        and not self._idle_centering
                    ):
                        self._begin_idle_centering()
                if not entering_idle:
                    # Do not advance the last *installed* facing baseline here.
                    # A small continuous turn may move less than the 5-degree
                    # epsilon per planner tick; updating this baseline without
                    # replanning would prevent those increments from ever
                    # accumulating into a turn request.
                    return

        if not self._should_replan(mode, nav):
            return

        # Copy the four-frame context under lock; inference must not block the
        # 50 Hz reference gather/advance path.
        with self._lock:
            gen_frame, context = self.planner.prepare_replan()
        try:
            turn_in_place = mode == 0 and not _same_vector(
                nav["facing"], self._last_cmd[1]["facing"]
            )
            generated = self.planner.infer_replan(
                context,
                mode=mode,
                target_vel=nav["speed"],
                height=nav["height"],
                movement_direction=nav["movement"][None],
                facing_direction=nav["facing"][None],
            )
            with self._lock:
                self.planner.install_replan(generated, gen_frame)
                self._idle_readapt_stored = False
                if self.idle_hold and mode == 0 and not turn_in_place:
                    self._idle_holding = False
                    self._idle_settling = True
                    self._turn_plan_active = False
                    print(
                        "[PLANNER] IDLE transition: settling to a terminal "
                        "standing stance before stable-stance centering"
                    )
                elif self.idle_hold and turn_in_place:
                    self._idle_holding = False
                    self._idle_settling = False
                    self._idle_centering = False
                    if not self._turn_plan_active:
                        print(
                            "[PLANNER] turn-in-place active: changing facing "
                            "without entering the IDLE stop transition"
                        )
                    self._turn_plan_active = True
                else:
                    self._idle_holding = False
                    self._idle_settling = False
                    self._idle_centering = False
                    self._turn_plan_active = False
            self._last_cmd = (
                mode,
                {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in nav.items()},
            )
            self.replan_count += 1
        except Exception as exc:
            # Official C++ disables the planner after an inference/update
            # failure. Surface it to the 50 Hz loop so deploy_eskin's finally
            # path sends damping instead of silently walking a stale plan.
            self._fatal_error = RuntimeError(f"SONIC planner update failed: {exc}")
            print(f"[PICO] fatal planner error: {exc}")

    def window(self) -> dict:
        if not self.initialized:
            raise RuntimeError("initialize_from_robot() is required before window()")
        if self._fatal_error is not None:
            raise self._fatal_error
        if not self.threaded:
            self._maybe_replan()
        # Read VR exactly once. Mode selection and targets must come from the
        # same message snapshot; otherwise a VR3 transition could produce one
        # teleop-encoder tick without VR targets.
        smpl = self._fresh_smpl()
        streaming_active = getattr(self.client, "streaming_motion_active", lambda: False)()
        if self.force_smpl and not streaming_active:
            raise RuntimeError(
                "--mode smpl requires the official command topic to select "
                "STREAMED_MOTION (planner=false)"
            )
        if (streaming_active or self.force_smpl) and smpl is None and not self.force_g1:
            raise RuntimeError(
                "PICO selected streamed-motion/SMPL mode but no complete fresh "
                "pose window is available; refusing to fall through to G1"
            )
        vr = None if smpl is not None else self._fresh_vr()
        self._smpl_active = smpl is not None and not self.force_g1
        self._vr_input_present = vr is not None
        self._vr_active = self._vr_input_present and not self.force_g1
        with self._lock:
            plan = self.planner.gather(
                self.model_profile.g1_future_frames,
                self.model_profile.g1_future_stride,
            )
            plan = self._apply_idle_hold_to_plan(plan)

        rootq_fut = quat_mul(self._heading_alignment[None], plan["rootq_fut"])
        rootq_cur = quat_mul(self._heading_alignment, plan["root_quat_cur"])
        body_pos = np.zeros((1, NUM_BODIES, 3), dtype=np.float32)
        body_quat = np.zeros((1, NUM_BODIES, 4), dtype=np.float32)
        body_quat[..., 0] = 1.0
        body_pos[0, 0] = plan["root_pos_cur"]
        body_quat[0, 0] = rootq_cur

        result = {
            "dof_fut": np.asarray(plan["dof_fut"], np.float32)[None],
            "dvel_fut": np.asarray(plan["dvel_fut"], np.float32)[None],
            "rootq_fut": np.asarray(rootq_fut, np.float32)[None],
            "rootz_fut": np.asarray(plan["rootz_fut"], np.float32)[None],
            "body_pos_cur": body_pos,
            "body_quat_cur": body_quat,
            "rootz_cur": np.asarray(plan["rootz_cur"], np.float32).reshape(1),
            "wrist_dof_smpl": np.zeros(
                (1, self.model_profile.smpl_future_frames, 6), np.float32
            ),
            # Adapter metadata is not part of the official encoder tensor. It
            # describes a captured stand and whether this window represents an
            # active planner gait. SonicRunner may transform only its local
            # encoder-facing copy; planner recurrence remains untouched.
            "stable_stand_q_il": np.asarray(
                self._stable_stand_q_il, np.float32
            ).copy(),
            "stable_stand_root_z": np.float32(self._stable_stand_rootz),
            "reference_gait_active": np.bool_(
                self.sonic_mode == "g1"
                and not self._idle_holding
                and not self._idle_centering
                and (
                    self._last_cmd[0] != 0
                    or self._idle_settling
                    or self._turn_plan_active
                )
            ),
        }
        if self._smpl_active and smpl is not None:
            result["smpl_joints_fut"] = np.asarray(
                smpl["smpl_joints_local"], np.float32
            )[None]
            result["smpl_rootq_fut"] = np.asarray(
                smpl["root_quat_wxyz"], np.float32
            )[None]
            result["wrist_dof_smpl"] = np.asarray(
                smpl["joint_pos_il"], np.float32
            )[:, [23, 24, 25, 26, 27, 28]][None]
        if self._vr_active and vr is not None:
            pos, quat = vr
            targets = (
                np.asarray(pos, np.float32).reshape(1, 3, 3),
                np.asarray(quat, np.float32).reshape(1, 3, 4),
            )
            result["vr_targets_encoder"] = targets
            result["vr_targets_local"] = True
        self._last_reference_q_il = np.asarray(
            result["dof_fut"][0, 0], np.float32
        ).copy()
        self._last_reference_dq_il = np.asarray(
            result["dvel_fut"][0, 0], np.float32
        ).copy()
        self._last_reference_root_pos = np.asarray(
            result["body_pos_cur"][0, 0], np.float32
        ).copy()
        self._last_reference_root_z = np.float32(result["rootz_cur"][0])
        return result

    def reference_diagnostics(self) -> dict[str, np.ndarray | np.generic]:
        """Small fixed-shape snapshot for offline planner/plant diagnosis."""
        idle_phase = (
            3 if self._idle_holding
            else 2 if self._idle_centering
            else 1 if self._idle_settling
            else 0
        )
        return {
            "reference_q_il": self._last_reference_q_il.copy(),
            "reference_dq_il": self._last_reference_dq_il.copy(),
            "reference_root_position": self._last_reference_root_pos.copy(),
            "reference_root_z": np.float32(self._last_reference_root_z),
            "idle_hold_phase": np.int8(idle_phase),
            "stand_capture_pending": np.bool_(self._stand_capture_requested),
            "stand_capture_count": np.int32(self.stand_capture_count),
        }

    def teleop_fields(self) -> dict:
        """Fixed-shape fields so every recorded control row has the same schema."""
        mode = self.client.mode()
        nav = self.client.nav_command()
        vr = self._fresh_vr()
        if nav is None:
            movement = facing = np.full(3, np.nan, np.float32)
            speed = height = np.float32(np.nan)
        else:
            movement = np.asarray(nav["movement"], np.float32)
            facing = np.asarray(nav["facing"], np.float32)
            speed, height = np.float32(nav["speed"]), np.float32(nav["height"])
        if vr is None:
            vr_pos = np.full((3, 3), np.nan, np.float32)
            vr_quat = np.full((3, 4), np.nan, np.float32)
        else:
            vr_pos = np.asarray(vr[0], np.float32).reshape(3, 3)
            vr_quat = np.asarray(vr[1], np.float32).reshape(3, 4)
        return {
            "teleop_mode": np.int32(-1 if mode is None else mode),
            "teleop_movement": movement,
            "teleop_facing": facing,
            "teleop_speed": speed,
            "teleop_height": height,
            "teleop_vr_pos": vr_pos,
            "teleop_vr_quat_wxyz": vr_quat,
            "teleop_vr_valid": np.bool_(vr is not None),
            "teleop_vr_ignored_by_planner_profile": np.bool_(
                self.force_g1 and vr is not None
            ),
            "teleop_age_s": np.float32(self._client_age_s()),
            "teleop_stale": np.bool_(self._stale),
            "sonic_encoder_mode": np.int32(
                2 if self._smpl_active else 1 if self._vr_active else 0
            ),
            "planner_inference_ms": np.float32(self.planner.last_inference_ms),
            "planner_total_ms": np.float32(self.planner.last_total_ms),
            **self.reference_diagnostics(),
        }
