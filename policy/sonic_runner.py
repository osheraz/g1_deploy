"""SONIC low-level tracker on the real robot.

SONIC is not the single-``actor_obs`` architecture ``PolicyRunner`` handles. It
is a pair of ONNX models:

    encoder(reference window, 1762) -> token (64)
    decoder(token + 10-step proprio history, 994) -> joint action (29, IsaacLab order)

This is the real-robot counterpart of ``sonic/tracker.py`` in the tactile repo:
same observation builders and joint ordering, but fed from ``RobotInterface``
instead of a simulator, and with ``num_envs == 1``.

The observation builders, joint tables and the released ONNX all live in
``policy/sonic/``, vendored so this deploy is standalone.

The **reference window** (what the operator wants the robot to do) is supplied
by a provider, not by this class. See ``ReferenceWindowProvider`` below for the
contract; live PICO teleop needs a planner or a distilled student to produce it,
whereas clip replay works today.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol

import numpy as np


class ReferenceWindowProvider(Protocol):
    """Supplies the encoder's reference window for the current tick.

    Must return the keys ``sonic.policy.obs.build_encoder_obs`` expects, each
    with a leading batch dimension of 1:

        dof_fut       (1,10,29)   future joint targets, IsaacLab order
        dvel_fut      (1,10,29)   future joint velocities
        rootq_fut     (1,10,4)    future root quaternions, wxyz
        rootz_fut     (1,10)      future root heights
        body_pos_cur  (1,14,3)    current body positions
        body_quat_cur (1,14,4)    current body quaternions, wxyz
        rootz_cur     (1,)        current root height
        wrist_dof_smpl(1,...)     wrist dofs
        vr_targets    optional     3-point teleop targets
    """

    def window(self) -> dict:
        ...


class SonicRunner:
    """Encoder/decoder ONNX tracker driven by real robot state."""

    HISTORY = 10

    def __init__(self, config, reference: ReferenceWindowProvider):
        import onnxruntime as ort

        from policy.sonic import obs as sonic_obs, robot_g1
        from policy.sonic.history import (
            action_state_from_decoded,
            clear_action_state_rows,
        )

        self.robot_g1 = robot_g1
        self.obs = sonic_obs
        self._action_state_from_decoded = action_state_from_decoded
        self._clear_action_state_rows = clear_action_state_rows

        self.config = config
        self.reference = reference
        self.mode = getattr(config, "sonic_mode", "g1")

        # Tiny MLPs: single-threaded is faster than the pool and avoids
        # setaffinity churn in the control loop.
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        vendored = Path(__file__).resolve().parent / "sonic"
        enc = Path(getattr(config, "sonic_encoder", vendored / "model_encoder.onnx"))
        dec = Path(getattr(config, "sonic_decoder", vendored / "model_decoder.onnx"))

        self._enc = ort.InferenceSession(str(enc.expanduser()),
                                         sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self._dec = ort.InferenceSession(str(dec.expanduser()),
                                         sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self._enc_io = (self._enc.get_inputs()[0].name, self._enc.get_outputs()[0].name)
        self._dec_io = (self._dec.get_inputs()[0].name, self._dec.get_outputs()[0].name)

        n_dof = robot_g1.NUM_DOF  # 29, IsaacLab order

        # The released decoder is 29-DoF; this robot exposes 23. Absent joints
        # are held at their default and masked out of the action history, the
        # same "phantom" handling sonic/tracker.py uses.
        self.default_23 = np.asarray(config.default_angles_23, np.float64)
        self.default_il = np.zeros(n_dof)
        self.default_il[robot_g1.IL_FROM_OUR23] = self.default_23
        # phantom_hold_ref_dof: also hold the absent joints in the ENCODER's
        # reference, not just the decoder output. Matches sonic/tracker.py.
        self.hold_idx = robot_g1.MISSING_IL
        self.encoder_hold_idx = (
            self.hold_idx if getattr(config, "phantom_hold_ref_dof", True) else None
        )

        self._last_decoded_action_il = np.zeros((1, n_dof))
        self._last_act_il = np.zeros((1, n_dof))
        self._reset = np.ones(1, dtype=bool)

        self._h_angvel = np.zeros((1, self.HISTORY, 3))
        self._h_jpos = np.zeros((1, self.HISTORY, n_dof))
        self._h_jvel = np.zeros((1, self.HISTORY, n_dof))
        self._h_lastact = np.zeros((1, self.HISTORY, n_dof))
        self._h_grav = np.zeros((1, self.HISTORY, 3))

        self.act_count = 0
        self.action = np.zeros(config.num_actions, np.float32)
        self.action_ema = getattr(config, "action_ema", 1.0)

        print(f"SONIC: encoder={enc}")
        print(f"       decoder={dec}")

    def reset(self) -> None:
        """Warm-fill the history rings on the next tick."""
        self._reset[:] = True
        self._clear_action_state_rows(
            self._last_decoded_action_il, self._last_act_il, np.array([0])
        )

    def _encode(self, x):
        name_in, name_out = self._enc_io
        return self._enc.run([name_out], {name_in: x.astype(np.float32)})[0]

    def _decode(self, x):
        name_in, name_out = self._dec_io
        return self._dec.run([name_out], {name_in: x.astype(np.float32)})[0]

    def _proprio_il(self, dof_pos_23, dof_vel_23):
        """23-DoF robot state -> 29-DoF IsaacLab, absent joints locked at default."""
        q = self.default_il[None].copy()
        v = np.zeros((1, self.robot_g1.NUM_DOF))
        q[:, self.robot_g1.IL_FROM_OUR23] = dof_pos_23
        v[:, self.robot_g1.IL_FROM_OUR23] = dof_vel_23
        return q - self.default_il[None], v

    def compute_action(self, dof_pos, dof_vel, base_quat_wxyz, base_ang_vel_world,
                       **_ignored) -> np.ndarray:
        """One control step. ``dof_pos``/``dof_vel`` are RAW 23-DoF robot order.

        Extra keyword arguments are accepted and ignored so this is a drop-in
        for ``PolicyRunner.compute_action`` where the caller passes depth etc.
        """
        g = self.reference.window()

        base_quat = np.asarray(base_quat_wxyz, np.float64).reshape(1, 4)
        vr_targets = g.get("vr_targets_encoder", g.get("vr_targets"))
        enc_obs = self.obs.build_encoder_obs(
            self.mode, g["dof_fut"], g["dvel_fut"], g["rootq_fut"], g["rootz_fut"],
            g["body_pos_cur"], g["body_quat_cur"], g["rootz_cur"], base_quat,
            g["wrist_dof_smpl"], vr_targets=vr_targets,
            hold_idx=self.encoder_hold_idx,
            vr_targets_local=bool(g.get("vr_targets_local", False)),
        )
        token = self._encode(enc_obs)

        world_angvel = np.asarray(base_ang_vel_world, np.float64).reshape(1, 3)
        ang_vel = self.obs.quat_apply(self.obs.quat_inv(base_quat), world_angvel)
        jpos_rel, jvel = self._proprio_il(
            np.asarray(dof_pos, np.float64).reshape(1, -1),
            np.asarray(dof_vel, np.float64).reshape(1, -1),
        )
        grav = self.obs.gravity_dir(base_quat)

        rm = self._reset
        for h, v in ((self._h_angvel, ang_vel), (self._h_jpos, jpos_rel),
                     (self._h_jvel, jvel), (self._h_lastact, self._last_act_il),
                     (self._h_grav, grav)):
            h[~rm] = np.concatenate([h[~rm, 1:], v[~rm, None]], axis=1)
            h[rm] = v[rm, None]
        self._reset[:] = False

        history = {"ang_vel": self._h_angvel, "joint_pos_rel": self._h_jpos,
                   "joint_vel": self._h_jvel, "last_action": self._h_lastact,
                   "gravity": self._h_grav}
        action_il = self._decode(self.obs.build_decoder_obs(token, history))

        self._last_decoded_action_il, self._last_act_il = (
            self._action_state_from_decoded(action_il, self.hold_idx)
        )

        act = action_il[0, self.robot_g1.IL_FROM_OUR23].astype(np.float32)
        self.action = self.action_ema * act + (1.0 - self.action_ema) * self.action
        self.act_count += 1
        return self.action
