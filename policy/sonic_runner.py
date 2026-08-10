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
from typing import Protocol
import time

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
        self.robot_g1 = robot_g1
        self.obs = sonic_obs

        self.config = config
        self.reference = reference
        self.mode = reference.sonic_mode
        if config.sonic_model_variant != "default":
            raise ValueError(
                "g1_deploy currently validates only sonic_model_variant: default; "
                f"got {config.sonic_model_variant!r}"
            )
        if int(config.num_actions) != len(robot_g1.IL_FROM_OUR23):
            raise ValueError(
                "this SONIC runner is the explicit 23-DoF adapter; "
                f"num_actions={config.num_actions}, expected 23"
            )
        if not bool(config.phantom_23dof):
            raise ValueError(
                "23-DoF SONIC deployment requires phantom_23dof: true; "
                "the flag cannot be disabled without a separate 29-DoF runner"
            )
        for name in ("default_angles_23", "action_scale", "dof_lower", "dof_upper"):
            value = np.asarray(getattr(config, name))
            if value.shape != (23,) or not np.isfinite(value).all():
                raise ValueError(
                    f"SONIC config {name} must be one finite value per physical "
                    f"joint; got shape {value.shape}"
                )

        # Tiny MLPs: single-threaded is faster than the pool and avoids
        # setaffinity churn in the control loop.
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        vendored = Path(__file__).resolve().parent / "sonic"
        enc = Path(config.sonic_encoder or vendored / "model_encoder.onnx")
        dec = Path(config.sonic_decoder or vendored / "model_decoder.onnx")

        available = set(ort.get_available_providers())
        providers = [
            name for name in (
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            )
            if name in available
        ]
        if not providers:
            raise RuntimeError(
                f"ONNX Runtime exposes no supported provider; "
                f"available={sorted(available)}"
            )

        self._enc = ort.InferenceSession(str(enc.expanduser()),
                                         sess_options=so,
                                         providers=providers)
        self._dec = ort.InferenceSession(str(dec.expanduser()),
                                         sess_options=so,
                                         providers=providers)
        self._enc_io = (self._enc.get_inputs()[0].name, self._enc.get_outputs()[0].name)
        self._dec_io = (self._dec.get_inputs()[0].name, self._dec.get_outputs()[0].name)
        self.encoder_providers = tuple(self._enc.get_providers())
        self.decoder_providers = tuple(self._dec.get_providers())

        enc_in = tuple(self._enc.get_inputs()[0].shape)
        enc_out = tuple(self._enc.get_outputs()[0].shape)
        dec_in = tuple(self._dec.get_inputs()[0].shape)
        dec_out = tuple(self._dec.get_outputs()[0].shape)
        expected = ((1, 1762), (1, 64), (1, 994), (1, 29))
        actual = (enc_in, enc_out, dec_in, dec_out)
        if actual != expected:
            raise ValueError(
                "this wrapper implements the official Default SONIC observation "
                f"layout {expected}, but the selected ONNX pair reports {actual}. "
                "Do not mix Default, low_latency, or sonic_v1_1 model/config files; "
                "the low-latency encoder requires its distinct 1247-D step1 builder."
            )
        n_dof = robot_g1.NUM_DOF  # 29, IsaacLab order

        # The released decoder is 29-DoF; this robot exposes 23.  Missing joints
        # are omitted only when applying the command.  The official controller
        # feeds the raw 29-D decoder output back as last_action history.
        self.default_23 = np.asarray(config.default_angles_23, np.float64)
        self.default_il = np.zeros(n_dof)
        self.default_il[robot_g1.IL_FROM_OUR23] = self.default_23
        # phantom_hold_ref_dof: also hold the absent joints in the ENCODER's
        # reference, not just the decoder output. Matches sonic/tracker.py.
        self.hold_idx = robot_g1.MISSING_IL
        self.encoder_hold_idx = (
            self.hold_idx if config.phantom_hold_ref_dof else None
        )

        self._last_act_il = np.zeros((1, n_dof))

        self._h_angvel = np.zeros((1, self.HISTORY, 3))
        self._h_jpos = np.zeros((1, self.HISTORY, n_dof))
        self._h_jvel = np.zeros((1, self.HISTORY, n_dof))
        self._h_lastact = np.zeros((1, self.HISTORY, n_dof))
        self._h_grav = np.zeros((1, self.HISTORY, 3))

        self.act_count = 0
        self.action = np.zeros(config.num_actions, np.float32)
        self.action_ema = config.action_ema
        self.last_action_il_raw = np.zeros(n_dof, np.float32)
        self.last_action_23_raw = np.zeros(config.num_actions, np.float32)
        self.last_token = np.zeros(64, np.float32)
        self.last_encoder_obs = np.zeros(1762, np.float32)
        self.last_decoder_obs = np.zeros(994, np.float32)
        self.last_encoder_ms = np.nan
        self.last_decoder_ms = np.nan
        self.last_policy_ms = np.nan

        print(f"SONIC: encoder={enc}")
        print(f"       decoder={dec}")
        print(f"       encoder providers={self.encoder_providers}")
        print(f"       decoder providers={self.decoder_providers}")

    def reset(self) -> None:
        """Clear history; official StateLogger zero-pads unavailable past rows."""
        self._last_act_il.fill(0.0)
        for history in (
            self._h_angvel, self._h_jpos, self._h_jvel,
            self._h_lastact, self._h_grav,
        ):
            history.fill(0.0)
        self.action.fill(0.0)

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

    def compute_action(self, dof_pos, dof_vel, base_quat_wxyz, base_ang_vel_body,
                       **_ignored) -> np.ndarray:
        """One control step. ``dof_pos``/``dof_vel`` are RAW 23-DoF robot order.

        ``base_ang_vel_body`` is angular velocity in the BODY frame, which is
        what both the robot IMU gyroscope and MuJoCo's freejoint qvel[3:6]
        report. sonic/tracker.py receives IsaacLab's world-frame value and
        rotates it into the body frame; that rotation is already done for us,
        so doing it again would corrupt the observation.

        Extra keyword arguments are accepted and ignored so this is a drop-in
        for ``PolicyRunner.compute_action`` where the caller passes depth etc.
        """
        policy_start = time.perf_counter()
        g = self.reference.window()
        # Encoder mode is dynamic for PICO: fresh VR3 fields select teleop;
        # navigation-only and one-second-stale input select G1.
        self.mode = self.reference.sonic_mode

        base_quat = np.asarray(base_quat_wxyz, np.float64).reshape(1, 4)
        vr_targets = g.get("vr_targets_encoder", g.get("vr_targets"))
        enc_obs = self.obs.build_encoder_obs(
            self.mode, g["dof_fut"], g["dvel_fut"], g["rootq_fut"], g["rootz_fut"],
            g["body_pos_cur"], g["body_quat_cur"], g["rootz_cur"], base_quat,
            g["wrist_dof_smpl"], vr_targets=vr_targets,
            hold_idx=self.encoder_hold_idx,
            vr_targets_local=bool(g.get("vr_targets_local", False)),
        )
        encoder_start = time.perf_counter()
        token = self._encode(enc_obs)
        self.last_encoder_ms = (time.perf_counter() - encoder_start) * 1e3

        ang_vel = np.asarray(base_ang_vel_body, np.float64).reshape(1, 3)
        jpos_rel, jvel = self._proprio_il(
            np.asarray(dof_pos, np.float64).reshape(1, -1),
            np.asarray(dof_vel, np.float64).reshape(1, -1),
        )
        grav = self.obs.gravity_dir(base_quat)

        for h, v in ((self._h_angvel, ang_vel), (self._h_jpos, jpos_rel),
                     (self._h_jvel, jvel), (self._h_lastact, self._last_act_il),
                     (self._h_grav, grav)):
            h[:, :-1] = h[:, 1:]
            h[:, -1] = v

        history = {"ang_vel": self._h_angvel, "joint_pos_rel": self._h_jpos,
                   "joint_vel": self._h_jvel, "last_action": self._h_lastact,
                   "gravity": self._h_grav}
        decoder_obs = self.obs.build_decoder_obs(token, history)
        decoder_start = time.perf_counter()
        action_il = self._decode(decoder_obs)
        self.last_decoder_ms = (time.perf_counter() - decoder_start) * 1e3

        # Preserve all six unavailable outputs in recurrent history. They are
        # latent policy state even though the physical 23-DoF robot cannot
        # apply them.
        self._last_act_il = np.asarray(action_il, np.float64).copy()

        act = action_il[0, self.robot_g1.IL_FROM_OUR23].astype(np.float32)
        self.action = self.action_ema * act + (1.0 - self.action_ema) * self.action
        self.last_action_il_raw = action_il[0].astype(np.float32).copy()
        self.last_action_23_raw = act.copy()
        self.last_token = token[0].astype(np.float32).copy()
        self.last_encoder_obs = enc_obs[0].astype(np.float32).copy()
        self.last_decoder_obs = decoder_obs[0].astype(np.float32).copy()
        self.last_policy_ms = (time.perf_counter() - policy_start) * 1e3
        self.act_count += 1
        return self.action

    def diagnostic_fields(self) -> dict[str, np.ndarray]:
        """Fixed-shape, per-control-step data for deployment episode logs."""
        return {
            "sonic_action_il_raw": self.last_action_il_raw.copy(),
            "sonic_action_23_raw": self.last_action_23_raw.copy(),
            "sonic_missing_action_raw": self.last_action_il_raw[self.hold_idx].copy(),
            "sonic_token": self.last_token.copy(),
            "sonic_encoder_obs": self.last_encoder_obs.copy(),
            "sonic_decoder_obs": self.last_decoder_obs.copy(),
            "encoder_inference_ms": np.float32(self.last_encoder_ms),
            "decoder_inference_ms": np.float32(self.last_decoder_ms),
            "policy_total_ms": np.float32(self.last_policy_ms),
            "sonic_encoder_mode": np.int32(1 if self.mode == "teleop" else 0),
        }
