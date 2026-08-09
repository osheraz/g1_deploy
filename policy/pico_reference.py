"""Live PICO teleop reference window for the SONIC tracker.

Transport: the `planner` ZMQ topic, decoded by ``common/pico_client.py``.
Planner:   ``sonic/planner/kinematic.py`` SonicKinematicPlanner.
Window:    assembled as in ``teleop_planner_runner._build_g()``.

Command mapping, ZMQ topic -> planner:

    mode      -> update(mode=...)             LocomotionMode, incl. carrying
    speed     -> update(target_vel=...)
    height    -> update(height=...)
    movement  -> update(movement_direction=)  base navigation
    facing    -> update(facing_direction=)

The planner ONNX is ~774 MB, so it loads from the checkout named by
``sonic_repo`` rather than being vendored.
"""
from __future__ import annotations

import atexit
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Future window: 10 frames, stride 5 -- sonic/constants.py FUTURE_FRAMES/STRIDE.
FUTURE_FRAMES = 10
FUTURE_STRIDE = 5
NUM_BODIES = 14          # len(robot_g1.BODY_NAMES)

# official_gated replan policy, values from teleop_planner_runner.yaml
REPLAN_MIN_TICKS = 5         # 0.1 s between replans
REPLAN_PERIOD_TICKS = 50     # 1 s periodic while moving
SPEED_EPS = 0.05
DIR_EPS = np.radians(5.0)


def _direction_changed(cur, prev) -> bool:
    cur = np.asarray(cur, np.float32)[:2]
    prev = np.asarray(prev, np.float32)[:2]
    cn, pn = float(np.linalg.norm(cur)), float(np.linalg.norm(prev))
    if cn < 1e-6 and pn < 1e-6:
        return False
    if cn < 1e-6 or pn < 1e-6:
        return True
    dot = float(np.dot(cur / cn, prev / pn))
    return float(np.arccos(np.clip(dot, -1.0, 1.0))) > DIR_EPS


class PicoReferenceWindow:
    """Live 3-point teleop: planner-generated window + VR endpoint targets."""

    def __init__(self, config, pico_client, robot=None, threaded=True, gated=True):
        repo = config.sonic_repo
        if not repo:
            raise SystemExit(
                "PICO mode needs the SONIC research checkout for the kinematic\n"
                "planner (its ONNX is ~774 MB and is not vendored here).\n"
                "Set `sonic_repo:` in the config, e.g. ~/tactile."
            )
        repo = Path(repo).expanduser()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        from sonic.planner.kinematic import SonicKinematicPlanner
        from policy.sonic import robot_g1

        self.robot_g1 = robot_g1
        self.client = pico_client
        self.robot = robot
        self.config = config

        planner_onnx = config.sonic_planner_onnx
        kwargs = {"model_path": Path(planner_onnx).expanduser()} if planner_onnx else {}
        self.planner = SonicKinematicPlanner(**kwargs)

        self.initialized = False
        self.done = False
        self.sonic_mode = "teleop"   # g1 mode ignores the 3-point targets
        self._last_seq = -1
        self._last_cmd = None
        # official deploy runs the planner in its own thread (10 Hz); the
        # 50 Hz control loop must never block on the ~60 ms replan
        # threaded matches the official deploy (planner thread, 50 Hz control
        # never blocks). Synchronous is for faster-than-wall-clock headless sim.
        self.threaded = threaded
        self.gated = gated       # official_gated: replan on change / 1 Hz moving
        self._ticks = 0
        self._last_replan_tick = -REPLAN_PERIOD_TICKS
        self._lock = threading.Lock()
        self._replan_thread = None
        self._stop = False
        atexit.register(self.stop)      # onnxruntime crashes if a run spans exit
        print(f"SONIC reference: live PICO teleop, planner from {repo}")

    # -- lifecycle -------------------------------------------------------

    def initialize_from_robot(self, base_quat_wxyz, dof_pos_23) -> None:
        """Seed the planner from the robot's current state.

        Planner works in 29-DoF MuJoCo order; the robot reports 23, so absent
        joints come from the default pose.
        """
        rg = self.robot_g1
        # Absent joints (MISSING_IL) default to 0, which is their value in the
        # 29-dof default pose.
        q_il = np.zeros(rg.NUM_DOF, dtype=np.float32)
        q_il[rg.IL_FROM_OUR23] = np.asarray(dof_pos_23, dtype=np.float32)
        self.planner.initialize(
            base_quat_wxyz=np.asarray(base_quat_wxyz, dtype=np.float32),
            joint_positions_mj=rg.il_to_mj(q_il),
        )
        self.initialized = True
        if self.threaded and self._replan_thread is None:
            self._replan_thread = threading.Thread(
                target=self._replan_loop, daemon=True
            )
            self._replan_thread.start()

    def reset(self) -> None:
        self.initialized = False
        self._last_seq = -1

    def step(self) -> None:
        """Advance the planner one control step (50 Hz)."""
        if self.initialized:
            self._ticks += 1
            with self._lock:
                self.planner.advance(1)

    # -- replan thread ---------------------------------------------------

    def stop(self) -> None:
        self._stop = True
        if self._replan_thread is not None:
            self._replan_thread.join(timeout=1.0)

    def _replan_loop(self) -> None:
        while not self._stop:
            self._maybe_replan()
            time.sleep(0.01)

    def _maybe_replan(self) -> None:
        if self.client.latest_seq == self._last_seq:
            return
        self._last_seq = self.client.latest_seq
        nav = self.client.nav_command()
        mode = self.client.mode()
        if nav is None or mode is None:
            return
        if self.gated and not self._should_replan(int(mode), nav):
            return
        self._last_cmd = (int(mode), nav)
        self._last_replan_tick = self._ticks
        try:
            self._replan(int(mode), nav)
        except RuntimeError as e:       # late blend etc: skip, retry next
            print(f"replan skipped: {e}")

    def _should_replan(self, mode: int, nav: dict) -> bool:
        """teleop_planner_runner official_gated, minus modes we don't use."""
        if self._last_cmd is None:
            return True
        lmode, lnav = self._last_cmd
        if mode != lmode:
            return True
        elapsed = self._ticks - self._last_replan_tick
        if elapsed < REPLAN_MIN_TICKS:
            return False
        if float(nav["height"]) != float(lnav["height"]):
            return True
        if _direction_changed(nav["facing"], lnav["facing"]):
            return True
        if abs(float(nav["speed"]) - float(lnav["speed"])) > SPEED_EPS:
            return True
        if _direction_changed(nav["movement"], lnav["movement"]):
            return True
        return elapsed >= REPLAN_PERIOD_TICKS and float(nav["speed"]) != 0.0

    def _replan(self, mode: int, nav: dict) -> None:
        """planner.update(), split so the ONNX inference runs unlocked."""
        p = self.planner
        with self._lock:
            gen_frame = int(p.current_frame + p.motion_look_ahead_steps)
            context = p._context_from_motion(p.current_motion, gen_frame)
            q = p.current_motion.root_quat_wxyz[
                min(p.current_frame, p.current_motion.timesteps - 1)]

        # commands are pelvis-relative; the planner wants WORLD xy (see the
        # official clip_controller: movement_direction_world_xy)
        w, x, y, z = np.asarray(q, np.float64).reshape(4)
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = np.cos(yaw), np.sin(yaw)

        def to_world(v):
            v = np.asarray(v, dtype=np.float32)
            return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]],
                            dtype=np.float32)

        generated, _ = p._run_planner(
            context,
            mode=mode,
            target_vel=float(nav["speed"]),
            height=float(nav["height"]),
            movement_direction=to_world(nav["movement"]),
            facing_direction=to_world(nav["facing"]),
        )
        with self._lock:
            p.current_motion = p._blend_generated_motion(
                p.current_motion, generated,
                gen_frame=gen_frame, current_frame=p.current_frame,
            )
            p.current_frame = 0

    def window(self) -> dict:
        if not self.initialized:
            raise RuntimeError(
                "PicoReferenceWindow.initialize_from_robot() must be called "
                "with the robot's current state before the first window()"
            )
        if not self.threaded:
            self._maybe_replan()
        with self._lock:
            plan = self.planner.gather(FUTURE_FRAMES, FUTURE_STRIDE)

        root_pos_cur = np.asarray(plan["root_pos_cur"], dtype=np.float32)
        root_quat_cur = np.asarray(plan["root_quat_cur"], dtype=np.float32)

        # Body poses zero except the root.
        body_pos = np.zeros((1, NUM_BODIES, 3), dtype=np.float32)
        body_quat = np.zeros((1, NUM_BODIES, 4), dtype=np.float32)
        body_quat[..., 0] = 1.0
        body_pos[0, 0] = root_pos_cur
        body_quat[0, 0] = root_quat_cur

        g = {
            "dof_fut": np.asarray(plan["dof_fut"])[None],
            "dvel_fut": np.asarray(plan["dvel_fut"])[None],
            "rootq_fut": np.asarray(plan["rootq_fut"])[None],
            "rootz_fut": np.asarray(plan["rootz_fut"])[None],
            "body_pos_cur": body_pos,
            "body_quat_cur": body_quat,
            "rootz_cur": np.asarray(plan["rootz_cur"], dtype=np.float32).reshape(1),
            "wrist_dof_smpl": np.zeros((1, FUTURE_FRAMES, 6), dtype=np.float32),
        }

        # Live endpoints are already pelvis-local.
        vr = self.client.vr_3point()
        if vr is not None:
            pos, quat = vr
            targets = (pos.reshape(1, 3, 3).astype(np.float32),
                       quat.reshape(1, 3, 4).astype(np.float32))
            g["vr_targets"] = targets
            g["vr_targets_encoder"] = targets
            g["vr_targets_local"] = True
        return g

    # -- passthrough for recording --------------------------------------

    def teleop_fields(self) -> dict:
        """Flat arrays for the episode log: mode, nav, hands, VR endpoints."""
        out: dict[str, np.ndarray] = {}
        mode = self.client.mode()
        if mode is not None:
            out["teleop_mode"] = np.int32(mode)
        nav = self.client.nav_command()
        if nav is not None:
            out["teleop_movement"] = np.asarray(nav["movement"], np.float32)
            out["teleop_facing"] = np.asarray(nav["facing"], np.float32)
            out["teleop_speed"] = np.float32(nav["speed"])
            out["teleop_height"] = np.float32(nav["height"])
        hands = self.client.hands()
        if hands is not None:
            out["teleop_left_hand"] = np.asarray(hands["left"], np.float32)
            out["teleop_right_hand"] = np.asarray(hands["right"], np.float32)
        vr = self.client.vr_3point()
        if vr is not None:
            out["teleop_vr_pos"] = vr[0].astype(np.float32)
            out["teleop_vr_quat_wxyz"] = vr[1].astype(np.float32)
        return out
