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

import numpy as np

from policy.sonic import robot_g1
from policy.sonic.planner_onnx import SonicKinematicPlanner
from policy.sonic.rotation_np import heading_quat, quat_inv, quat_mul


FUTURE_FRAMES = 10
FUTURE_STRIDE = 5
NUM_BODIES = len(robot_g1.BODY_NAMES)
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
        idle_hold=False,
    ):
        self.idle_hold = bool(idle_hold)
        self._idle_holding = False
        planner_onnx = config.sonic_planner_onnx
        if not planner_onnx:
            raise SystemExit("PICO mode requires `sonic_planner_onnx` in the config")

        self.client = pico_client
        self.robot = robot
        self.config = config
        self.force_g1 = bool(force_g1)
        self.planner = SonicKinematicPlanner(
            planner_onnx,
            version=getattr(config, "sonic_planner_version", None),
            default_height=getattr(config, "sonic_planner_default_height", 0.788740),
        )

        self.initialized = False
        self.done = False
        self._vr_active = False
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
        atexit.register(self.stop)
        profile = "PLANNER/G1-only" if self.force_g1 else "dynamic PICO"
        print(
            f"SONIC reference: official-style PICO planner {self.planner.model_path} "
            f"({profile})"
        )

    @property
    def sonic_mode(self) -> str:
        return "teleop" if self._vr_active and not self.force_g1 else "g1"

    def initialize_from_robot(self, base_quat_wxyz, dof_pos_23) -> None:
        """Initialize planner joints and official heading alignment from hardware."""
        q_il = np.zeros(robot_g1.NUM_DOF, dtype=np.float32)
        q_il[robot_g1.IL_FROM_OUR23] = np.asarray(dof_pos_23, dtype=np.float32)
        base_q = np.asarray(base_quat_wxyz, dtype=np.float64).reshape(4)
        self.planner.initialize(base_q, robot_g1.il_to_mj(q_il))

        # The V2 planner's IDLE plan contains stepping (stance transition), so
        # tracking it makes the robot walk on its own. idle_hold replaces the
        # initial idle plan with a constant hold of the measured pose; sim gate:
        # planner idle plan = 2.7m drift / 20s, hold = 0.01m.
        if self.idle_hold or os.environ.get("SONIC_FLAT_IDLE"):
            m = self.planner.current_motion
            m.joint_pos_il[:] = q_il[None]
            m.joint_vel_il[:] = 0.0
            m.root_pos[:, :2] = 0.0
            m.root_quat_wxyz[:] = m.root_quat_wxyz[0]
            self._idle_holding = True

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
            f"23 physical joints -> 29 slots, pelvis heading={yaw_deg:+.1f}deg, "
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
        self._vr_input_present = False
        self._stale = False

    def step(self) -> None:
        """Advance the installed 50 Hz trajectory after one policy command."""
        if self.initialized:
            self._ticks += 1
            with self._lock:
                self.planner.advance(1)
                self._idle_readapt_at_horizon()

    def _idle_readapt_at_horizon(self) -> None:
        """Mirror the official lower-body end-of-IDLE adaptation state machine."""
        motion = self.planner.current_motion
        if (
            self.robot is None
            or motion is None
            or self.planner.current_frame != motion.timesteps - 1
            or self._last_cmd[0] != 0
        ):
            return

        measured_il = np.zeros(robot_g1.NUM_DOF, np.float64)
        measured_il[robot_g1.IL_FROM_OUR23] = np.asarray(
            self.robot.qj, np.float64
        ).reshape(23)
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

    def _install_idle_hold(self) -> None:
        """IDLE = hold the configured default pose (the validated neutral
        reference) instead of the planner's stepping idle plan."""
        m = self.planner.current_motion
        cur = min(self.planner.current_frame, m.timesteps - 1)
        default_il = np.zeros(robot_g1.NUM_DOF, np.float64)
        default_il[robot_g1.IL_FROM_OUR23] = np.asarray(
            self.config.default_angles_23, np.float64
        )
        m.joint_pos_il[:] = default_il[None]
        m.joint_vel_il[:] = 0.0
        m.root_pos[:] = m.root_pos[cur]
        m.root_quat_wxyz[:] = m.root_quat_wxyz[cur]
        self.planner.current_frame = 0

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

    @staticmethod
    def _period_for_mode(mode: int) -> float:
        if mode == RUN_MODE:
            return 0.1
        if mode == CRAWLING_MODE:
            return 0.2
        if mode in BOXING_MODES:
            return 1.0
        return 1.0

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
            if nav["speed"] != last_nav["speed"]:
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

        # idle-hold: stand on the default pose, but let facing changes fall
        # through to a real turn-in-place replan; re-hold when the turn ends
        if self.idle_hold and mode == 0:
            last_mode, last_nav = self._last_cmd
            if _same_vector(nav["facing"], last_nav["facing"]):
                with self._lock:
                    m = self.planner.current_motion
                    exhausted = self.planner.current_frame >= m.timesteps - 1
                    if last_mode != 0 or (exhausted and not self._idle_holding):
                        self._install_idle_hold()
                        self._idle_holding = True
                self._last_cmd = (
                    0,
                    {k: v.copy() if isinstance(v, np.ndarray) else v
                     for k, v in nav.items()},
                )
                return

        if not self._should_replan(mode, nav):
            return

        # Copy the four-frame context under lock; inference must not block the
        # 50 Hz reference gather/advance path.
        with self._lock:
            gen_frame, context = self.planner.prepare_replan()
        try:
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
                self._idle_holding = False
            self._last_cmd = (
                mode,
                {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in nav.items()},
            )
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
        vr = self._fresh_vr()
        self._vr_input_present = vr is not None
        self._vr_active = self._vr_input_present and not self.force_g1
        with self._lock:
            plan = self.planner.gather(FUTURE_FRAMES, FUTURE_STRIDE)

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
            "wrist_dof_smpl": np.zeros((1, FUTURE_FRAMES, 6), np.float32),
        }
        if self._vr_active and vr is not None:
            pos, quat = vr
            targets = (
                np.asarray(pos, np.float32).reshape(1, 3, 3),
                np.asarray(quat, np.float32).reshape(1, 3, 4),
            )
            result["vr_targets_encoder"] = targets
            result["vr_targets_local"] = True
        return result

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
            "sonic_encoder_mode": np.int32(1 if self._vr_active else 0),
            "planner_inference_ms": np.float32(self.planner.last_inference_ms),
            "planner_total_ms": np.float32(self.planner.last_total_ms),
        }
