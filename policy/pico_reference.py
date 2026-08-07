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

import sys
from pathlib import Path

import numpy as np

# Future window: 10 frames, stride 5 -- sonic/constants.py FUTURE_FRAMES/STRIDE.
FUTURE_FRAMES = 10
FUTURE_STRIDE = 5
NUM_BODIES = 14          # len(robot_g1.BODY_NAMES)


class PicoReferenceWindow:
    """Live 3-point teleop: planner-generated window + VR endpoint targets."""

    def __init__(self, config, pico_client, robot=None):
        repo = getattr(config, "sonic_repo", None)
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

        planner_onnx = getattr(config, "sonic_planner_onnx", None)
        kwargs = {"model_path": Path(planner_onnx).expanduser()} if planner_onnx else {}
        self.planner = SonicKinematicPlanner(**kwargs)

        self._initialized = False
        self._last_seq = -1
        self._last_cmd = None
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
        self._initialized = True

    def reset(self) -> None:
        self._initialized = False
        self._last_seq = -1

    def step(self) -> None:
        """Advance the planner one control step (50 Hz)."""
        if self._initialized:
            self.planner.advance(1)

    # -- per tick --------------------------------------------------------

    def _replan_if_new_command(self) -> None:
        """Re-run the planner only when a fresh teleop command arrived."""
        if self.client.latest_seq == self._last_seq:
            return
        self._last_seq = self.client.latest_seq
        nav = self.client.nav_command()
        mode = self.client.mode()
        if nav is None or mode is None:
            return
        self._last_cmd = (mode, nav)
        self.planner.update(
            mode=int(mode),
            target_vel=float(nav["speed"]),
            height=float(nav["height"]),
            movement_direction=np.asarray(nav["movement"], dtype=np.float32),
            facing_direction=np.asarray(nav["facing"], dtype=np.float32),
        )

    def window(self) -> dict:
        if not self._initialized:
            raise RuntimeError(
                "PicoReferenceWindow.initialize_from_robot() must be called "
                "with the robot's current state before the first window()"
            )
        self._replan_if_new_command()
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
