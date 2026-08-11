"""SONIC low-level tracker on the real robot.

SONIC is not the single-``actor_obs`` architecture ``PolicyRunner`` handles. It
is a pair of ONNX models:

    encoder(reference window, variant-specific) -> token (64)
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
        from policy.sonic import obs as sonic_obs, robot_g1
        from policy.sonic.inference_backend import create_session
        from policy.sonic.model_profile import get_model_profile
        from policy.sonic.robot_profile import get_robot_profile
        self.robot_g1 = robot_g1
        self.obs = sonic_obs

        self.config = config
        self.reference = reference
        self.mode = reference.sonic_mode
        self.model_profile = get_model_profile(config.sonic_model_variant)
        self.robot_profile = get_robot_profile(config.sonic_robot_profile)
        if int(config.num_actions) != self.robot_profile.physical_dof:
            raise ValueError(
                f"profile {self.robot_profile.name} has "
                f"{self.robot_profile.physical_dof} physical joints, but "
                f"num_actions={config.num_actions}"
            )
        configured_motors = np.asarray(config.action_joint2motor_idx, np.int64)
        configured_fixed = np.asarray(config.fixed_joint2motor_idx, np.int64)
        if not np.array_equal(
            configured_motors, self.robot_profile.physical_motor_indices
        ):
            raise ValueError(
                f"{self.robot_profile.name} requires physical motor order "
                f"{self.robot_profile.physical_motor_indices.tolist()}, got "
                f"{configured_motors.tolist()}"
            )
        if not np.array_equal(configured_fixed, self.robot_profile.missing_motor):
            raise ValueError(
                f"{self.robot_profile.name} requires fixed/missing motors "
                f"{self.robot_profile.missing_motor.tolist()}, got "
                f"{configured_fixed.tolist()}"
            )
        for name in ("physical_default_angles", "action_scale", "dof_lower", "dof_upper"):
            value = np.asarray(getattr(config, name))
            if value.shape != (self.robot_profile.physical_dof,) or not np.isfinite(value).all():
                raise ValueError(
                    f"SONIC config {name} must be one finite value per physical "
                    f"joint; got shape {value.shape}"
                )
        full_default = np.asarray(config.default_angles, np.float64)
        if full_default.shape != (29,):
            raise ValueError(f"SONIC default_angles must be full 29-D motor order, got {full_default.shape}")
        if not np.allclose(
            full_default[self.robot_profile.physical_motor_indices],
            config.physical_default_angles,
            atol=1e-7,
            rtol=0.0,
        ):
            raise ValueError(
                "physical_default_angles does not match default_angles at the "
                "selected embodiment's motor indices"
            )

        vendored = Path(__file__).resolve().parent / "sonic"
        enc = Path(config.sonic_encoder or vendored / "model_encoder.onnx")
        dec = Path(config.sonic_decoder or vendored / "model_decoder.onnx")
        backend_args = dict(
            backend=config.sonic_inference_backend,
            cache_dir=config.sonic_engine_cache_dir,
            fp16=bool(config.sonic_tensorrt_fp16),
        )
        self._enc = create_session(
            enc.expanduser(),
            input_shapes={"obs_dict": (1, self.model_profile.encoder_input_dim)},
            **backend_args,
        )
        self._dec = create_session(
            dec.expanduser(), input_shapes={"obs_dict": (1, 994)}, **backend_args
        )
        self._enc_io = (self._enc.get_inputs()[0].name, self._enc.get_outputs()[0].name)
        self._dec_io = (self._dec.get_inputs()[0].name, self._dec.get_outputs()[0].name)
        self.encoder_providers = tuple(self._enc.get_providers())
        self.decoder_providers = tuple(self._dec.get_providers())

        enc_in = tuple(self._enc.get_inputs()[0].shape)
        enc_out = tuple(self._enc.get_outputs()[0].shape)
        dec_in = tuple(self._dec.get_inputs()[0].shape)
        dec_out = tuple(self._dec.get_outputs()[0].shape)
        expected = self.model_profile.expected_shapes
        actual = (enc_in, enc_out, dec_in, dec_out)
        if actual != expected:
            raise ValueError(
                f"{self.model_profile.name} requires ONNX shapes {expected}, but "
                f"the selected pair reports {actual}. Encoder, decoder, and "
                "sonic_model_variant must come from the same official model card row."
            )
        n_dof = robot_g1.NUM_DOF  # 29, IsaacLab order

        # The released decoder is 29-DoF; this robot exposes 23.  Missing joints
        # are omitted only when applying the command.  The official controller
        # feeds the raw 29-D decoder output back as last_action history.
        self.default_physical = np.asarray(config.physical_default_angles, np.float64)
        self.default_il = self.robot_profile.physical_to_isaaclab(
            self.default_physical, fill=0.0
        )
        # phantom_hold_ref_dof: also hold the absent joints in the ENCODER's
        # reference, not just the decoder output. Matches sonic/tracker.py.
        self.hold_idx = self.robot_profile.missing_isaaclab
        self.encoder_hold_idx = (
            self.hold_idx if config.sonic_mask_missing_reference else None
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
        self.last_action_physical_raw = np.zeros(config.num_actions, np.float32)
        self.last_token = np.zeros(64, np.float32)
        self.last_encoder_obs = np.zeros(self.model_profile.encoder_input_dim, np.float32)
        self.last_decoder_obs = np.zeros(994, np.float32)
        self.last_encoder_ms = np.nan
        self.last_decoder_ms = np.nan
        self.last_policy_ms = np.nan

        print(
            f"SONIC: model={self.model_profile.name}, "
            f"robot={self.robot_profile.name}, encoder={enc}"
        )
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

    def _proprio_il(self, dof_pos, dof_vel):
        """Physical robot state -> official 29-D SONIC proprioception."""
        q = self.robot_profile.physical_to_isaaclab(dof_pos, fill=0.0)
        v = self.robot_profile.physical_to_isaaclab(dof_vel, fill=0.0)
        return q - self.default_il[None], v

    def compute_action(self, dof_pos, dof_vel, base_quat_wxyz, base_ang_vel_body,
                       **_ignored) -> np.ndarray:
        """One control step. ``dof_pos``/``dof_vel`` are physical robot order.

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
            model_profile=self.model_profile,
            smpl_joints_fut=g.get("smpl_joints_fut"),
            smpl_rootq_fut=g.get("smpl_rootq_fut"),
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

        act = self.robot_profile.isaaclab_to_physical(action_il[0]).astype(np.float32)
        self.action = self.action_ema * act + (1.0 - self.action_ema) * self.action
        self.last_action_il_raw = action_il[0].astype(np.float32).copy()
        self.last_action_physical_raw = act.copy()
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
            "sonic_action_physical_raw": self.last_action_physical_raw.copy(),
            "sonic_missing_action_raw": self.last_action_il_raw[self.hold_idx].copy(),
            "sonic_token": self.last_token.copy(),
            "sonic_encoder_obs": self.last_encoder_obs.copy(),
            "sonic_decoder_obs": self.last_decoder_obs.copy(),
            "encoder_inference_ms": np.float32(self.last_encoder_ms),
            "decoder_inference_ms": np.float32(self.last_decoder_ms),
            "policy_total_ms": np.float32(self.last_policy_ms),
            "sonic_encoder_mode": np.int32({"g1": 0, "teleop": 1, "smpl": 2}[self.mode]),
        }
