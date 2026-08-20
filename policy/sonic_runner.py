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

    def __init__(
        self,
        config,
        reference: ReferenceWindowProvider,
        *,
        allow_legacy_adaptation: bool = False,
    ):
        from policy.sonic import obs as sonic_obs, robot_g1
        from policy.sonic.inference_backend import create_session
        from policy.sonic.action_adaptation import MotorActionAdapter
        from policy.sonic.model_profile import get_model_profile
        from policy.sonic.reference_adaptation import (
            GaitReferenceAdapter,
            LandingHoldAdapter,
        )
        from policy.sonic.reduced_adaptation import ReducedProprioceptionAdapter
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
            allow_tf32=bool(config.sonic_tensorrt_tf32),
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
        self.encoder_engine_path = getattr(self._enc, "engine_path", None)
        self.decoder_engine_path = getattr(self._dec, "engine_path", None)

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

        # The released decoder is 29-DoF; this robot exposes 23. Missing joints
        # are omitted only when applying the motor command. NVIDIA defines
        # last_action as the previous raw decoder output, not an estimate of
        # executed motion, so all 29 channels remain in policy history. Actual
        # non-motion of absent axes is already represented by q/dq feedback.
        self.default_physical = np.asarray(config.physical_default_angles, np.float64)
        self.default_il = self.robot_profile.physical_to_isaaclab(
            self.default_physical, fill=0.0
        )
        # Complete official action scale, first in motor/MuJoCo order and then
        # in decoder/IsaacLab order. Present entries come from the selected
        # config; the six absent axes use NVIDIA's released motor constants.
        full_action_scale_mj = np.zeros(29, np.float64)
        full_action_scale_mj[self.robot_profile.physical_motor_indices] = np.asarray(
            config.action_scale, np.float64
        )
        if self.robot_profile.is_reduced:
            wrist_4010_scale = 0.25 * 5.0 / (
                0.00425 * (10.0 * 2.0 * np.pi) ** 2
            )
            missing_scale = {
                13: 0.438577, 14: 0.438577,
                20: wrist_4010_scale, 21: wrist_4010_scale,
                27: wrist_4010_scale, 28: wrist_4010_scale,
            }
            for motor in self.robot_profile.missing_motor:
                full_action_scale_mj[motor] = missing_scale[int(motor)]
        full_action_scale_il = robot_g1.mj_to_il(full_action_scale_mj)
        self.reduced_proprioception = ReducedProprioceptionAdapter(
            strategy=getattr(config, "sonic_reduced_proprioception", "zero"),
            missing_idx=self.robot_profile.missing_isaaclab,
            default_il=self.default_il,
            action_scale_il=full_action_scale_il,
            control_dt=float(config.control_dt),
            settling_time_s=float(
                getattr(config, "sonic_shadow_settling_time_s", 0.12)
            ),
        )
        self.motor_action_adapter = MotorActionAdapter(
            action_scale=config.action_scale,
            control_dt=float(config.control_dt),
            scale=getattr(config, "sonic_motor_action_scale", None),
            abs_limit=getattr(config, "sonic_motor_action_abs_limit", None),
            ema=getattr(config, "sonic_motor_action_ema", None),
            target_rate_limit=getattr(
                config, "sonic_motor_target_rate_limit", None
            ),
        )
        # First twelve physical joints are the two six-axis legs for both G1
        # embodiments. Convert that physical selection to SONIC/IsaacLab order.
        leg_physical = np.arange(min(12, self.robot_profile.physical_dof))
        self.gait_reference_adapter = GaitReferenceAdapter(
            leg_idx=self.robot_profile.isaaclab_from_physical[leg_physical],
            leg_scale=getattr(config, "sonic_reference_leg_scale", 1.0),
            root_height_scale=getattr(
                config, "sonic_reference_root_height_scale", 1.0
            ),
            default_stand_q_il=self.default_il,
            joint_scale_il=self.robot_profile.physical_to_isaaclab(
                np.asarray(config.sonic_reference_joint_scale, np.float64),
                fill=1.0,
            ),
            joint_dvel_scale_il=(
                None if getattr(
                    config, "sonic_reference_joint_dvel_scale", None
                ) is None
                else self.robot_profile.physical_to_isaaclab(
                    np.asarray(
                        config.sonic_reference_joint_dvel_scale, np.float64
                    ),
                    fill=1.0,
                )
            ),
            joint_offset_il=(
                None if getattr(
                    config, "sonic_reference_joint_offset_rad", None
                ) is None
                else self.robot_profile.physical_to_isaaclab(
                    np.asarray(
                        config.sonic_reference_joint_offset_rad, np.float64
                    ),
                    fill=0.0,
                )
            ),
            joint_flex_scale_il=(
                None if getattr(
                    config, "sonic_reference_joint_flex_scale", None
                ) is None
                else self.robot_profile.physical_to_isaaclab(
                    np.asarray(
                        config.sonic_reference_joint_flex_scale, np.float64
                    ),
                    fill=1.0,
                )
            ),
        )
        self.landing_hold_adapter = LandingHoldAdapter(
            enabled=bool(getattr(config, "sonic_landing_hold", False)),
            mode=str(getattr(config, "sonic_landing_hold_mode", "pose")),
        )  # legacy attempt; default off
        self.action_history_source = str(
            getattr(config, "sonic_action_history_source", "raw")
        ).strip().lower()
        if self.action_history_source not in ("raw", "executed"):
            raise ValueError("sonic_action_history_source must be raw or executed")
        # Reference/history masks are explicit reduced-embodiment adaptations.
        # Official-parity execution leaves both native 29-D; the named
        # named research profiles enable masks under checked narrow contracts.
        self.hold_idx = self.robot_profile.missing_isaaclab
        adaptation_scope = str(
            getattr(config, "sonic_missing_adaptation_scope", "all")
        ).strip().lower()
        if adaptation_scope == "all":
            scoped_hold_idx = self.hold_idx
        elif adaptation_scope == "waist":
            scoped_hold_idx = np.intersect1d(self.hold_idx, [5, 8])
        elif adaptation_scope == "wrist":
            scoped_hold_idx = np.setdiff1d(self.hold_idx, [5, 8])
        else:
            raise ValueError(
                "sonic_missing_adaptation_scope must be all, waist, or wrist"
            )
        self.missing_adaptation_scope = adaptation_scope
        self.encoder_hold_idx = (
            scoped_hold_idx if config.sonic_mask_missing_reference else None
        )
        self.action_history_hold_idx = (
            scoped_hold_idx
            if getattr(config, "sonic_mask_missing_action_history", False)
            else None
        )
        self.adaptation_profile = str(
            getattr(config, "sonic_adaptation_profile", "official-parity")
        ).strip().lower()
        if self.adaptation_profile not in (
            "official-parity", "stable-slow-23", "robust-small-gait-23",
            "forward-stable-23",
        ):
            raise ValueError(
                "sonic_adaptation_profile must be official-parity, "
                "stable-slow-23, robust-small-gait-23, or forward-stable-23"
            )
        if (
            self.adaptation_profile in ("stable-slow-23", "robust-small-gait-23")
            and not allow_legacy_adaptation
        ):
            raise ValueError(
                f"{self.adaptation_profile} is a withdrawn legacy attempt and "
                "is blocked from live use. Current recipe: forward-stable-23 "
                "--slow-gait. Reproduction only: pass --legacy-reproduction"
            )
        if self.adaptation_profile == "forward-stable-23":
            release_profile = getattr(config, "sonic_release_profile", None)
            reference_joint_scale = np.asarray(
                config.sonic_reference_joint_scale, np.float64
            )
            def _bilateral_in_range(indices) -> bool:
                pair = reference_joint_scale[list(indices)]
                return bool(
                    np.allclose(pair, pair[0], atol=1e-12)
                    and 0.28 <= float(pair[0]) <= 1.00
                )

            other_scale = np.delete(
                reference_joint_scale, [0, 3, 4, 6, 9, 10]
            )
            qualified_lower_body_scale = (
                _bilateral_in_range((0, 6))
                and _bilateral_in_range((3, 9))
                and _bilateral_in_range((4, 10))
                and np.all(other_scale == 1.0)
            )
            direct_output = (
                np.all(self.motor_action_adapter.scale == 1.0)
                and np.all(self.motor_action_adapter.ema == 1.0)
                and np.all(np.isinf(self.motor_action_adapter.abs_limit))
                and np.all(np.isinf(
                    self.motor_action_adapter.target_rate_limit
                ))
            )
            contract_ok = (
                self.robot_profile.name == "g1_23dof"
                and release_profile in (
                    "low-latency", "original-release", "sonic-v1.1"
                )
                and self.missing_adaptation_scope == "all"
                and self.encoder_hold_idx is not None
                and self.action_history_hold_idx is not None
                and self.reduced_proprioception.strategy == "zero"
                and float(config.action_ema) == 1.0
                and float(config.sonic_reference_leg_scale) == 1.0
                and float(config.sonic_reference_root_height_scale) == 1.0
                and qualified_lower_body_scale
                and direct_output
                and self.action_history_source == "raw"
            )
            if not contract_ok:
                raise ValueError(
                    "forward-stable-23 is defined only for g1_23dof with a "
                    "released model profile, all six absent reference/history "
                    "axes masked, zero missing-axis proprioception, bilateral "
                    "hip-pitch/knee/ankle-pitch reference scales in "
                    "[0.28, 1.00] with every other reference channel unscaled, "
                    "raw action history, and fully direct output"
                )
        if self.adaptation_profile == "stable-slow-23":
            release_profile = getattr(config, "sonic_release_profile", None)
            expected_scale = np.ones(self.robot_profile.physical_dof)
            expected_ema = np.ones(self.robot_profile.physical_dof)
            expected_scale[[5, 11]] = 0.95
            expected_ema[[4, 10]] = 0.95
            contract_ok = (
                self.robot_profile.name == "g1_23dof"
                and release_profile == "original-release"
                and self.missing_adaptation_scope == "all"
                and self.encoder_hold_idx is not None
                and self.action_history_hold_idx is not None
                and self.reduced_proprioception.strategy == "zero"
                and float(config.action_ema) == 1.0
                and np.array_equal(
                    self.motor_action_adapter.scale, expected_scale
                )
                and np.array_equal(
                    self.motor_action_adapter.ema, expected_ema
                )
                and np.all(np.isinf(self.motor_action_adapter.abs_limit))
                and np.all(np.isinf(
                    self.motor_action_adapter.target_rate_limit
                ))
                and self.action_history_source == "raw"
            )
            if not contract_ok:
                raise ValueError(
                    "stable-slow-23 is qualified only for Original Release + "
                    "g1_23dof with both absent-axis masks, zero missing-axis "
                    "proprioception, all-axis scope, ankle-pitch EMA=0.95, "
                    "ankle-roll action scale=0.95, raw action history, and no "
                    "clipping or slew limit"
                )
        if self.adaptation_profile == "robust-small-gait-23":
            release_profile = getattr(config, "sonic_release_profile", None)
            contracts = {
                "low-latency": {
                    "scope": "wrist", "mask_reference": True,
                    "mask_history": False, "hip_scale": 0.80,
                },
                "original-release": {
                    "scope": "wrist", "mask_reference": True,
                    "mask_history": True, "hip_scale": 0.80,
                },
                "sonic-v1.1": {
                    "scope": "wrist", "mask_reference": True,
                    "mask_history": True, "hip_scale": 0.90,
                },
            }
            selected_contract = contracts.get(release_profile)
            expected_reference_scale = np.ones(self.robot_profile.physical_dof)
            if self.robot_profile.physical_dof >= 7:
                expected_reference_scale[[0, 6]] = (
                    selected_contract["hip_scale"]
                    if selected_contract is not None else np.nan
                )
            direct_output = (
                np.all(self.motor_action_adapter.scale == 1.0)
                and np.all(self.motor_action_adapter.ema == 1.0)
                and np.all(np.isinf(self.motor_action_adapter.abs_limit))
                and np.all(np.isinf(
                    self.motor_action_adapter.target_rate_limit
                ))
            )
            contract_ok = (
                self.robot_profile.name == "g1_23dof"
                and selected_contract is not None
                and self.missing_adaptation_scope == selected_contract["scope"]
                and (self.encoder_hold_idx is not None)
                    == selected_contract["mask_reference"]
                and (self.action_history_hold_idx is not None)
                    == selected_contract["mask_history"]
                and self.reduced_proprioception.strategy == "zero"
                and float(config.action_ema) == 1.0
                and float(config.sonic_reference_leg_scale) == 1.0
                and float(config.sonic_reference_root_height_scale) == 1.0
                and np.array_equal(
                    np.asarray(config.sonic_reference_joint_scale),
                    expected_reference_scale,
                )
                and direct_output
                and self.action_history_source == "raw"
            )
            if not contract_ok:
                raise ValueError(
                    "robust-small-gait-23 requires its model-specific g1_23dof "
                    "contract: wrist-only absent-reference masking, the "
                    "qualified wrist action-history setting and bilateral "
                    "hip-pitch reference scale, zero missing proprioception, "
                    "raw 29-D action history, and fully direct motor output"
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
        self.last_reference_q_il_adapted = np.zeros(n_dof, np.float32)
        self.last_reference_dq_il_adapted = np.zeros(n_dof, np.float32)
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
        print(
            "       reduced proprioception="
            f"{self.reduced_proprioception.strategy}, "
            f"missing adaptation scope={self.missing_adaptation_scope}, "
            f"adaptation profile={self.adaptation_profile}"
        )
        print(
            "       motor shaping="
            f"{'enabled' if self.motor_action_adapter.enabled else 'direct'}, "
            f"action history={self.action_history_source}"
        )
        print(
            "       gait reference="
            f"leg scale {self.gait_reference_adapter.leg_scale:g}, "
            "root-height scale "
            f"{self.gait_reference_adapter.root_height_scale:g}"
        )

    def reset(self) -> None:
        """Clear history; official StateLogger zero-pads unavailable past rows."""
        self._last_act_il.fill(0.0)
        for history in (
            self._h_angvel, self._h_jpos, self._h_jvel,
            self._h_lastact, self._h_grav,
        ):
            history.fill(0.0)
        self.action.fill(0.0)
        self.reduced_proprioception.reset()
        self.motor_action_adapter.reset()
        self.landing_hold_adapter.reset()

    def _encode(self, x):
        name_in, name_out = self._enc_io
        return self._enc.run([name_out], {name_in: x.astype(np.float32)})[0]

    def _decode(self, x):
        name_in, name_out = self._dec_io
        return self._dec.run([name_out], {name_in: x.astype(np.float32)})[0]

    def _proprio_il(self, dof_pos, dof_vel, reference):
        """Physical robot state -> official 29-D SONIC proprioception."""
        q = self.robot_profile.physical_to_isaaclab(dof_pos, fill=0.0)
        v = self.robot_profile.physical_to_isaaclab(dof_vel, fill=0.0)
        q, v = self.reduced_proprioception.fill(
            q,
            v,
            reference_q_il=np.asarray(reference["dof_fut"])[0, 0],
            reference_dq_il=np.asarray(reference["dvel_fut"])[0, 0],
        )
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
        raw = self.reference.window()
        g = self.gait_reference_adapter.apply(raw)
        measured_il = self.robot_profile.physical_to_isaaclab(
            np.asarray(dof_pos, np.float64).reshape(-1), fill=0.0
        )
        g = self.landing_hold_adapter.apply(
            g,
            measured_q_il=measured_il,
            raw_dof_fut=g.get("dof_fut"),
            foot_z=_ignored.get("foot_z"),
            foot_contact=_ignored.get("foot_contact"),
        )
        self.last_reference_q_il_adapted = np.asarray(
            g["dof_fut"][0, 0], np.float32
        ).copy()
        self.last_reference_dq_il_adapted = np.asarray(
            g["dvel_fut"][0, 0], np.float32
        ).copy()
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
            g,
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

        raw_act = self.robot_profile.isaaclab_to_physical(
            action_il[0]
        ).astype(np.float32)
        act = self.motor_action_adapter.apply(raw_act)
        # The legacy scalar EMA remains for backwards-compatible ablations.
        self.action = self.action_ema * act + (1.0 - self.action_ema) * self.action

        # Official SONIC feeds the raw decoder vector back as last_action.
        # ``executed`` is an explicit research ablation that substitutes the
        # shaped values only on physically present axes.
        history_il = np.asarray(action_il, np.float64).copy()
        if self.action_history_source == "executed":
            history_il[0, self.robot_profile.isaaclab_from_physical] = self.action
        self._last_act_il = (
            history_il
            if self.action_history_hold_idx is None
            else self._masked_action_history(history_il)
        )
        self.reduced_proprioception.advance(action_il[0])
        self.last_action_il_raw = action_il[0].astype(np.float32).copy()
        self.last_action_physical_raw = raw_act.copy()
        self.last_token = token[0].astype(np.float32).copy()
        self.last_encoder_obs = enc_obs[0].astype(np.float32).copy()
        self.last_decoder_obs = decoder_obs[0].astype(np.float32).copy()
        self.last_policy_ms = (time.perf_counter() - policy_start) * 1e3
        self.act_count += 1
        return self.action

    def _masked_action_history(self, action_il) -> np.ndarray:
        """Reduced-embodiment adaptation; official SONIC keeps all 29 values."""
        out = np.asarray(action_il, np.float64).copy()
        out[..., self.action_history_hold_idx] = 0.0
        return out

    def diagnostic_fields(self) -> dict[str, np.ndarray]:
        """Fixed-shape, per-control-step data for deployment episode logs."""
        missing_q, missing_dq = self.reduced_proprioception.diagnostics()
        return {
            "sonic_action_il_raw": self.last_action_il_raw.copy(),
            "sonic_action_physical_raw": self.last_action_physical_raw.copy(),
            "sonic_missing_action_raw": self.last_action_il_raw[self.hold_idx].copy(),
            "sonic_missing_proprio_q": missing_q,
            "sonic_missing_proprio_dq": missing_dq,
            "sonic_token": self.last_token.copy(),
            "sonic_encoder_obs": self.last_encoder_obs.copy(),
            "sonic_decoder_obs": self.last_decoder_obs.copy(),
            "sonic_reference_q_il_adapted": (
                self.last_reference_q_il_adapted.copy()
            ),
            "sonic_reference_dq_il_adapted": (
                self.last_reference_dq_il_adapted.copy()
            ),
            "encoder_inference_ms": np.float32(self.last_encoder_ms),
            "decoder_inference_ms": np.float32(self.last_decoder_ms),
            "policy_total_ms": np.float32(self.last_policy_ms),
            "sonic_encoder_mode": np.int32({"g1": 0, "teleop": 1, "smpl": 2}[self.mode]),
            "sonic_reference_gait_active": np.bool_(
                self.gait_reference_adapter.last_active
            ),
            "sonic_landing_hold": self.landing_hold_adapter.last_holding.copy(),
        }
