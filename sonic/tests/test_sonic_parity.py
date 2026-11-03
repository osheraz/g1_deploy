from pathlib import Path
import json
import struct
import sys
import threading
import tempfile
import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sonic.model import obs, robot_g1  # noqa: E402
from sonic.model.model_profile import MODEL_PROFILES, get_model_profile  # noqa: E402
from sonic.model.robot_profile import G1_23DOF, G1_29DOF  # noqa: E402
from sonic.model.planner_onnx import (  # noqa: E402
    PlannerMotion,
    SonicKinematicPlanner,
)
from sonic.model.hardware_contract import (  # noqa: E402
    require_hardware_config,
    verify_hardware_sessions,
)
from sonic.model.inference_backend import (  # noqa: E402
    DIRECT_TENSORRT_PROVIDER,
    ENGINE_CACHE_SCHEMA,
    _apply_builder_math_flags,
    _engine_cache_signature,
)
from core.robot.unitree_crc import check_crc_library  # noqa: E402
from core.robot.unitree_dds import configure_process_local_cyclonedds_log  # noqa: E402
from core.robot.mode import release_active_motion_mode
from sonic.pico_reference import LOWER_BODY_IL_IDX, PicoReferenceWindow  # noqa: E402
from sonic.sim import (  # noqa: E402
    _KEYPAD_TELEOP_KEYS,
    _MUJOCO_DOCUMENTED_VIEWER_KEYS,
)


class MujocoKeyboardBindingTest(unittest.TestCase):
    def test_numeric_keypad_controls_do_not_overlap_viewer_shortcuts(self):
        self.assertTrue(_KEYPAD_TELEOP_KEYS.isdisjoint(_MUJOCO_DOCUMENTED_VIEWER_KEYS))


class RecorderButtonOwnershipTest(unittest.TestCase):
    def test_automatic_only_recorder_does_not_consume_gamepad_x(self):
        from core.recording.episode import Recorder
        from core.robot.remote import KeyMap

        with tempfile.TemporaryDirectory() as tmp:
            recorder = Recorder(tmp, {"test": True}, button=None)
            buttons = [0] * 16
            buttons[KeyMap.X] = 1
            recorder.handle_buttons(buttons)
            self.assertFalse(recorder.recording)
            self.assertEqual(recorder.button_name, "automatic-only")


class UnitreeDdsConfigTest(unittest.TestCase):
    def test_shared_cyclonedds_log_is_replaced_with_process_local_path(self):
        module = SimpleNamespace(
            ChannelConfigHasInterface=("<Tracing><OutputFile>/tmp/cdds.LOG</OutputFile></Tracing>")
        )
        path = configure_process_local_cyclonedds_log(module)
        self.assertEqual(path.parent, Path("/tmp"))
        self.assertIn(str(path), module.ChannelConfigHasInterface)
        self.assertNotIn(">/tmp/cdds.LOG<", module.ChannelConfigHasInterface)
        self.assertEqual(configure_process_local_cyclonedds_log(module), path)

    def test_unitree_crc_requires_installed_sdk_binary(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lib").mkdir()
            for name in ("crc_amd64.so", "crc_aarch64.so"):
                (root / "lib" / name).write_text("version https://git-lfs.github.com/spec/v1\n")
            module = SimpleNamespace(__file__=str(root / "crc.py"))
            with self.assertRaisesRegex(RuntimeError, "git lfs pull"):
                check_crc_library(module)
            self.assertEqual(module.__file__, str(root / "crc.py"))


class UnitreeMotionModeHandoffTest(unittest.TestCase):
    class _Client:
        def __init__(self, modes, release_status=0):
            self.modes = deque(modes)
            self.release_status = release_status
            self.timeout = None
            self.initialized = False
            self.release_count = 0

        def SetTimeout(self, timeout):
            self.timeout = timeout

        def Init(self):
            self.initialized = True

        def CheckMode(self):
            return 0, self.modes.popleft()

        def ReleaseMode(self):
            self.release_count += 1
            return self.release_status, None

    def test_active_service_is_released_before_return(self):
        client = self._Client([{"form": "0", "name": "ai"}, {"form": "", "name": ""}])
        sleeps = []
        returned = release_active_motion_mode(lambda: client, sleep_fn=sleeps.append)
        self.assertIs(returned, client)
        self.assertEqual(client.timeout, 5.0)
        self.assertTrue(client.initialized)
        self.assertEqual(client.release_count, 1)
        self.assertEqual(sleeps, [5.0])

    def test_release_failure_is_fail_closed(self):
        client = self._Client([{"form": "0", "name": "ai"}], release_status=3102)
        with self.assertRaisesRegex(RuntimeError, "ReleaseMode failed"):
            release_active_motion_mode(lambda: client, sleep_fn=lambda _: None)

    def test_check_failure_is_fail_closed(self):
        client = self._Client([])
        client.CheckMode = lambda: (3104, None)
        with self.assertRaisesRegex(RuntimeError, "CheckMode failed"):
            release_active_motion_mode(lambda: client, sleep_fn=lambda _: None)


class DefaultPoseHoldTelemetryTest(unittest.TestCase):
    @staticmethod
    def _robot(age_s):
        from core.robot.remote import KeyMap
        from core.robot.robot_interface import RobotInterface

        robot = object.__new__(RobotInterface)
        robot.config = SimpleNamespace(
            control_dt=0.0,
            max_low_state_age_s=0.1,
            max_default_hold_low_state_age_s=0.5,
            default_angles=np.zeros(2),
            kps=np.ones(2),
            kds=np.ones(2),
        )
        robot.remote_controller = SimpleNamespace(button=[0] * 16)
        robot.low_cmd = SimpleNamespace(
            motor_cmd=[SimpleNamespace(q=0.0, qd=0.0, kp=0.0, kd=0.0, tau=0.0) for _ in range(2)]
        )
        robot.low_state_snapshot_with_age = lambda: (None, age_s)
        robot.send_cmd = lambda: robot.remote_controller.button.__setitem__(KeyMap.start, 1)
        return robot

    def test_brief_gap_keeps_unchanged_default_hold_alive(self):
        from core.robot.remote import KeyMap

        robot = self._robot(0.112)
        robot.default_pos_state(KeyMap.start, "Start")
        self.assertEqual(robot.remote_controller.button[KeyMap.start], 1)

    def test_sustained_gap_still_fails_closed(self):
        from core.robot.remote import KeyMap

        robot = self._robot(0.501)
        with self.assertRaisesRegex(RuntimeError, "continuously unavailable"):
            robot.default_pos_state(KeyMap.start, "Start")


class TensorRTContractTest(unittest.TestCase):
    class _Flags:
        TF32 = "tf32"
        FP16 = "fp16"

    class _TRT:
        BuilderFlag = None

    class _Config:
        def __init__(self):
            self.set_flags = []
            self.cleared_flags = []

        def set_flag(self, flag):
            self.set_flags.append(flag)

        def clear_flag(self, flag):
            self.cleared_flags.append(flag)

    def setUp(self):
        self._TRT.BuilderFlag = self._Flags

    def test_fp32_explicitly_clears_tf32_and_fp16(self):
        config = self._Config()
        builder = SimpleNamespace(platform_has_fast_fp16=True)
        _apply_builder_math_flags(config, builder, self._TRT, fp16=False, allow_tf32=False)
        self.assertEqual(config.set_flags, [])
        self.assertEqual(config.cleared_flags, [self._Flags.TF32, self._Flags.FP16])

    def test_math_mode_changes_engine_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.onnx"
            model.write_bytes(b"onnx")
            common = dict(
                trt_version="10.10.0",
                torch_cuda="12.8",
                gpu_name="RTX 4090",
                gpu_capability=(8, 9),
                fp16=False,
                input_shapes={"obs": (1, 4)},
            )
            strict = _engine_cache_signature(model, allow_tf32=False, **common)
            relaxed = _engine_cache_signature(model, allow_tf32=True, **common)
            self.assertEqual(strict["schema"], ENGINE_CACHE_SCHEMA)
            self.assertNotEqual(strict, relaxed)
            self.assertFalse(strict["math"]["allow_tf32"])

    def test_robot_facing_config_and_sessions_fail_closed(self):
        config = SimpleNamespace(
            policy_type="sonic",
            sonic_inference_backend="auto",
            sonic_planner_backend="auto",
            sonic_tensorrt_fp16=False,
            sonic_tensorrt_tf32=False,
        )
        with self.assertRaisesRegex(RuntimeError, "automatic"):
            require_hardware_config(config, "motion")

        config.sonic_inference_backend = "tensorrt"
        config.sonic_planner_backend = "tensorrt"
        require_hardware_config(config, "planner")
        config.sonic_tensorrt_tf32 = True
        with self.assertRaisesRegex(RuntimeError, "forbids TF32"):
            require_hardware_config(config, "planner")
        config.sonic_tensorrt_tf32 = False
        config.sonic_tensorrt_fp16 = True
        with self.assertRaisesRegex(RuntimeError, "requires the reviewed FP32"):
            require_hardware_config(config, "planner")
        config.sonic_tensorrt_fp16 = False
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "model.engine"
            engine.write_bytes(b"engine")
            planner = SimpleNamespace(providers=(DIRECT_TENSORRT_PROVIDER,), engine_path=engine)
            runner = SimpleNamespace(
                encoder_providers=(DIRECT_TENSORRT_PROVIDER,),
                decoder_providers=(DIRECT_TENSORRT_PROVIDER,),
                encoder_engine_path=engine,
                decoder_engine_path=engine,
                reference=SimpleNamespace(planner=planner),
            )
            paths = verify_hardware_sessions(runner, "planner")
            self.assertEqual(set(paths), {"encoder", "decoder", "planner"})

            runner.decoder_providers = ("CPUExecutionProvider",)
            with self.assertRaisesRegex(RuntimeError, "decoder provider"):
                verify_hardware_sessions(runner, "planner")


class EncoderMaskTest(unittest.TestCase):
    def setUp(self):
        self.q = np.ones((1, 10, 29))
        self.dq = np.full((1, 10, 29), 2.0)
        self.rootq = np.zeros((1, 10, 4))
        self.rootq[..., 0] = 1.0
        self.body_pos = np.ones((1, 14, 3))
        self.body_quat = np.zeros((1, 14, 4))
        self.body_quat[..., 0] = 1.0

    def build(self, mode, model_profile="default"):
        profile = get_model_profile(model_profile)
        sf = profile.smpl_future_frames
        kwargs = {}
        if mode == "smpl":
            kwargs = {
                "smpl_joints_fut": np.ones((1, sf, 24, 3)),
                "smpl_rootq_fut": self.rootq[:, :sf].copy(),
            }
        return obs.build_encoder_obs(
            mode,
            self.q,
            self.dq,
            self.rootq,
            np.ones((1, 10)),
            self.body_pos,
            self.body_quat,
            np.ones(1),
            np.array([[1.0, 0.0, 0.0, 0.0]]),
            np.ones((1, sf, 6)),
            model_profile=profile,
            **kwargs,
        )

    def test_g1_only_populates_official_terms(self):
        x = self.build("g1")
        self.assertFalse(x[:, 584:601].any())
        self.assertFalse(x[:, 661:].any())
        self.assertTrue(x[:, 4:584].any())
        self.assertTrue(x[:, 601:661].any())

    def test_teleop_only_populates_official_terms(self):
        x = self.build("teleop")
        self.assertFalse(x[:, 4:595].any())
        self.assertFalse(x[:, 601:661].any())
        self.assertFalse(x[:, 922:].any())
        self.assertTrue(x[:, 595:601].any())
        self.assertTrue(x[:, 661:922].any())

    def test_no_height_profiles_use_graph_specific_anchor_offsets(self):
        layouts = {
            "low_latency": {
                "order": "single_then_multi",
                "single": slice(584, 590),
                "multi": slice(590, 650),
            },
            "sonic_v1_1": {
                "order": "multi_then_single",
                "single": slice(644, 650),
                "multi": slice(584, 644),
            },
        }
        for name, expected in layouts.items():
            with self.subTest(profile=name, mode="g1"):
                profile = get_model_profile(name)
                self.assertEqual(profile.encoder_anchor_order, expected["order"])
                g1 = self.build("g1", name)
                self.assertFalse(g1[:, expected["single"]].any())
                self.assertTrue(g1[:, expected["multi"]].any())
            with self.subTest(profile=name, mode="teleop"):
                teleop = self.build("teleop", name)
                self.assertTrue(teleop[:, expected["single"]].any())
                self.assertFalse(teleop[:, expected["multi"]].any())

    def test_smpl_only_populates_official_terms(self):
        x = self.build("smpl")
        self.assertFalse(x[:, 4:922].any())
        self.assertTrue(x[:, 1702:].any())

    def test_all_released_model_shapes_and_23dof_wrist_mask(self):
        for name, profile in MODEL_PROFILES.items():
            sf = profile.smpl_future_frames
            x = obs.build_encoder_obs(
                "smpl",
                self.q,
                self.dq,
                self.rootq,
                np.ones((1, 10)),
                self.body_pos,
                self.body_quat,
                np.ones(1),
                np.array([[1.0, 0.0, 0.0, 0.0]]),
                np.ones((1, sf, 6)),
                hold_idx=G1_23DOF.missing_isaaclab,
                model_profile=profile,
                smpl_joints_fut=np.ones((1, sf, 24, 3)),
                smpl_rootq_fut=self.rootq[:, :sf],
            )
            self.assertEqual(x.shape, (1, profile.encoder_input_dim), name)
            np.testing.assert_array_equal(x[0, -sf * 6 :].reshape(sf, 6)[:, 2:], 0.0)


class PlannerMathTest(unittest.TestCase):
    def test_joint_order_round_trip(self):
        il = np.arange(29)
        np.testing.assert_array_equal(robot_g1.mj_to_il(robot_g1.il_to_mj(il)), il)
        np.testing.assert_array_equal(
            G1_29DOF.isaaclab_to_physical(G1_29DOF.physical_to_isaaclab(il)), il
        )
        physical23 = np.arange(23)
        projected = G1_23DOF.physical_to_isaaclab(physical23)
        np.testing.assert_array_equal(G1_23DOF.isaaclab_to_physical(projected), physical23)
        np.testing.assert_array_equal(projected[G1_23DOF.missing_isaaclab], 0)

    def test_reduced_planner_motion_remains_native_recurrent_context(self):
        motion = PlannerMotion(
            joint_pos_il=np.ones((5, 29), dtype=np.float64),
            joint_vel_il=np.full((5, 29), 2.0, dtype=np.float64),
            root_pos=np.zeros((5, 3), dtype=np.float64),
            root_quat_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (5, 1)),
        )
        ref = object.__new__(PicoReferenceWindow)
        ref.robot_profile = G1_23DOF
        # 23-DoF masking belongs at tracker observation/action boundaries. The
        # planner's own 29-DoF q/dq remains exactly what the model generated.
        np.testing.assert_array_equal(motion.joint_pos_il, 1.0)
        np.testing.assert_array_equal(motion.joint_vel_il, 2.0)

    def test_idle_hold_overrides_tracker_window_without_mutating_plan(self):
        dof = np.arange(4 * 29, dtype=np.float64).reshape(4, 29)
        dvel = dof + 1000.0
        rootq = np.tile([1.0, 0.0, 0.0, 0.0], (4, 1))
        rootq[:, 3] = np.arange(4) * 0.01
        plan = {
            "dof_fut": dof.copy(),
            "dvel_fut": dvel.copy(),
            "rootq_fut": rootq.copy(),
            "root_quat_cur": rootq[1].copy(),
            "rootz_fut": np.linspace(0.7, 0.8, 4),
            "rootz_cur": np.float64(0.75),
        }
        original = {key: np.asarray(value).copy() for key, value in plan.items()}
        ref = object.__new__(PicoReferenceWindow)
        ref._idle_holding = True
        ref._idle_hold_q_il = np.linspace(-0.2, 0.2, 29)
        ref._idle_hold_rootq = np.array([1.0, 0.0, 0.0, 0.0])
        ref._idle_hold_rootz = 0.77

        held = ref._apply_idle_hold_to_plan(plan)

        np.testing.assert_allclose(held["dof_fut"], np.broadcast_to(ref._idle_hold_q_il, dof.shape))
        np.testing.assert_array_equal(held["dvel_fut"], 0.0)
        np.testing.assert_allclose(
            held["rootq_fut"],
            np.broadcast_to(ref._idle_hold_rootq, rootq.shape),
        )
        np.testing.assert_allclose(held["rootz_fut"], 0.77)
        for key, value in original.items():
            np.testing.assert_array_equal(plan[key], value)

    def test_idle_centering_smoothly_targets_known_stable_stand(self):
        plan = {
            "dof_fut": np.zeros((4, 29), dtype=np.float64),
            "dvel_fut": np.zeros((4, 29), dtype=np.float64),
            "rootq_fut": np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)),
            "root_quat_cur": np.array([1.0, 0.0, 0.0, 0.0]),
            "rootz_fut": np.full(4, 0.75),
            "rootz_cur": np.float64(0.75),
        }
        original = {key: np.asarray(value).copy() for key, value in plan.items()}
        ref = object.__new__(PicoReferenceWindow)
        ref._idle_centering = True
        ref._idle_centering_step = 0
        ref._idle_centering_start_q_il = np.zeros(29)
        ref._stable_stand_q_il = np.ones(29)
        ref._idle_centering_rootq = np.array([1.0, 0.0, 0.0, 0.0])
        ref._idle_centering_start_rootz = 0.75
        ref._stable_stand_rootz = 0.79
        ref.model_profile = SimpleNamespace(g1_future_stride=5)

        centered = ref._apply_idle_hold_to_plan(plan)

        np.testing.assert_allclose(centered["dof_fut"][0], 0.0)
        self.assertTrue(np.all(np.diff(centered["dof_fut"][:, 0]) > 0.0))
        self.assertTrue(np.all(centered["dvel_fut"][:, 0] >= 0.0))
        self.assertGreater(centered["rootz_fut"][-1], centered["rootz_fut"][0])
        for key, value in original.items():
            np.testing.assert_array_equal(plan[key], value)

    def test_reference_mask_utility_is_not_used_for_official_action_history(self):
        decoded = np.arange(29, dtype=np.float64)[None]
        projected = G1_23DOF.mask_isaaclab_reference(decoded)
        np.testing.assert_array_equal(projected[:, G1_23DOF.missing_isaaclab], 0.0)
        present = np.setdiff1d(np.arange(29), G1_23DOF.missing_isaaclab)
        np.testing.assert_array_equal(projected[:, present], decoded[:, present])
        np.testing.assert_array_equal(G1_29DOF.mask_isaaclab_reference(decoded), decoded)
        # SONIC's official last_action is raw decoder output. The mask above is
        # a reference/projection utility, not the default history contract.
        np.testing.assert_array_equal(decoded, np.arange(29)[None])

    def test_reduced_config_preserves_reference_and_raw_action_history(self):
        from core.config import Config

        config = Config("g1_23dof_sonic.yaml")
        self.assertFalse(config.sonic_mask_missing_reference)
        self.assertFalse(config.sonic_mask_missing_action_history)

    def test_resample_context_and_blend(self):
        qpos = np.zeros((30, 36), dtype=np.float64)
        qpos[:, 2] = 0.788740
        qpos[:, 3] = 1.0
        qpos[:, 7:] = np.arange(29)
        motion = SonicKinematicPlanner._resample_50hz(qpos)
        self.assertEqual(motion.timesteps, 50)
        self.assertTrue(np.isfinite(motion.joint_vel_il).all())
        context = SonicKinematicPlanner._context_from_motion(motion, 2)
        self.assertEqual(context.shape, (1, 4, 36))

        planner = object.__new__(SonicKinematicPlanner)
        planner.current_motion = motion
        planner.current_frame = 3
        planner.install_replan(motion, 5)
        self.assertEqual(planner.current_frame, 0)
        self.assertEqual(planner.current_motion.timesteps, 52)

    def test_official_idle_horizon_readaptation(self):
        motion = PlannerMotion(
            joint_pos_il=np.zeros((2, 29), dtype=np.float64),
            joint_vel_il=np.zeros((2, 29), dtype=np.float64),
            root_pos=np.zeros((2, 3), dtype=np.float64),
            root_quat_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        )
        measured_il = np.zeros(29)
        measured_il[LOWER_BODY_IL_IDX] = 0.2
        robot = type("Robot", (), {})()
        robot.qj = G1_23DOF.isaaclab_to_physical(measured_il)
        planner = type("Planner", (), {})()
        planner.current_motion = motion
        planner.current_frame = 1

        ref = object.__new__(PicoReferenceWindow)
        ref.robot = robot
        ref.planner = planner
        ref.idle_hold = False
        ref._last_cmd = (0, {})
        ref._idle_readapt_original = np.zeros(29)
        ref._idle_readapt_stored = False
        ref._idle_readapt_state = "idle"
        ref.robot_profile = G1_23DOF
        ref._idle_readapt_at_horizon()
        np.testing.assert_allclose(motion.joint_pos_il[-1, LOWER_BODY_IL_IDX], 0.004, atol=1e-12)
        self.assertEqual(ref._idle_readapt_state, "adapting")


class PicoSemanticsTest(unittest.TestCase):
    def test_ordinary_locomotion_replans_at_planner_rate(self):
        """A constant straight command must not fall back to the old 1 Hz path."""
        ref = object.__new__(PicoReferenceWindow)
        ref.default_replan_period_s = 0.1
        ref._replan_interval_s = 0.0
        ref.gated = True
        nav = {
            "movement": np.array([1.0, 0.0, 0.0]),
            "facing": np.array([1.0, 0.0, 0.0]),
            "speed": 0.4,
            "height": -1.0,
        }
        ref._last_cmd = (
            1,
            {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in nav.items()
            },
        )
        self.assertTrue(ref._should_replan(1, nav))

    def test_idle_small_turn_keeps_last_installed_facing_until_replan(self):
        ref = object.__new__(PicoReferenceWindow)
        ref.initialized = True
        ref._fatal_error = None
        ref._ticks = 5
        ref._last_planner_tick = 0
        ref.idle_hold = True
        ref._last_cmd = (
            0,
            {
                "movement": np.zeros(3),
                "facing": np.array([1.0, 0.0, 0.0]),
                "speed": -1.0,
                "height": -1.0,
            },
        )
        small_yaw = np.radians(2.0)
        nav = {
            "movement": np.zeros(3),
            "facing": np.array([np.cos(small_yaw), np.sin(small_yaw), 0.0]),
            "speed": -1.0,
            "height": -1.0,
        }
        ref._current_command = lambda: (0, nav)
        ref._lock = threading.Lock()
        ref.planner = SimpleNamespace(
            current_frame=2,
            current_motion=SimpleNamespace(timesteps=20),
        )
        ref._idle_holding = True
        ref._idle_centering = False
        ref._maybe_replan()
        np.testing.assert_array_equal(ref._last_cmd[1]["facing"], [1.0, 0.0, 0.0])

    def test_turn_in_place_is_not_marked_as_idle_stop_transition(self):
        ref = object.__new__(PicoReferenceWindow)
        ref.initialized = True
        ref._fatal_error = None
        ref._ticks = 5
        ref._last_planner_tick = 0
        ref.idle_hold = True
        ref._last_cmd = (
            0,
            {
                "movement": np.zeros(3),
                "facing": np.array([1.0, 0.0, 0.0]),
                "speed": -1.0,
                "height": -1.0,
            },
        )
        nav = {
            "movement": np.zeros(3),
            "facing": np.array([0.98, 0.2, 0.0]),
            "speed": -1.0,
            "height": -1.0,
        }
        ref._current_command = lambda: (0, nav)
        ref._should_replan = lambda mode, command: True
        ref._lock = threading.Lock()
        ref.planner = SimpleNamespace(
            current_frame=2,
            current_motion=SimpleNamespace(timesteps=20),
            prepare_replan=lambda: (4, np.zeros((1, 4, 36))),
            infer_replan=lambda context, **command: object(),
            install_replan=lambda generated, frame: None,
        )
        ref._idle_holding = True
        ref._idle_settling = False
        ref._idle_centering = False
        ref._turn_plan_active = False
        ref._idle_readapt_stored = False
        ref.replan_count = 0
        ref._maybe_replan()
        self.assertTrue(ref._turn_plan_active)
        self.assertFalse(ref._idle_settling)
        self.assertFalse(ref._idle_holding)
        self.assertEqual(ref.replan_count, 1)

    def test_stand_capture_saves_support_qualified_measured_pose(self):
        motion = PlannerMotion(
            joint_pos_il=np.zeros((2, 29)),
            joint_vel_il=np.zeros((2, 29)),
            root_pos=np.array([[0.0, 0.0, 0.78], [0.0, 0.0, 0.78]]),
            root_quat_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        )
        physical = np.linspace(-0.2, 0.2, 23)
        support = {
            "left_supported": True,
            "right_supported": True,
            "max_corner_z_m": 0.005,
            "max_foot_corner_range_m": 0.001,
            "com_support_margin_m": 0.08,
            "base_xy": np.zeros(2),
            "base_planar_speed_mps": 0.0,
        }
        robot = SimpleNamespace(
            qj=physical.copy(),
            dqj=np.zeros(23),
            base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            root_z=0.79,
            stand_support_diagnostics=lambda: support,
        )
        client = SimpleNamespace(idle_capture_stationary=lambda: True)
        ref = object.__new__(PicoReferenceWindow)
        ref.idle_hold = True
        ref.robot = robot
        ref.client = client
        ref.robot_profile = G1_23DOF
        ref.planner = SimpleNamespace(current_motion=motion, current_frame=0)
        ref._stand_capture_requested = False
        ref._stand_capture_samples = deque(maxlen=50)
        ref._stand_capture_wait_ticks = 0
        ref.stand_capture_count = 0
        ref._stable_stand_rootz = 0.78
        ref._idle_holding = True
        ref._idle_hold_q_il = np.linspace(-0.1, 0.1, 29)
        ref._idle_hold_rootz = 0.78
        ref._turn_plan_active = False
        self.assertTrue(ref.request_stable_stand_capture("test"))
        for _ in range(50):
            ref._update_stable_stand_capture()
        self.assertEqual(ref.stand_capture_count, 1)
        self.assertFalse(ref._stand_capture_requested)
        expected = G1_23DOF.physical_to_isaaclab(physical, fill=0.0)
        np.testing.assert_allclose(ref._stable_stand_q_il, expected)
        np.testing.assert_allclose(ref._idle_hold_q_il, expected)
        self.assertAlmostEqual(ref._stable_stand_rootz, 0.79)
        self.assertAlmostEqual(ref._idle_hold_rootz, 0.79)

    def test_stand_capture_accepts_stable_physical_pose_without_contact_fields(self):
        robot = SimpleNamespace(
            qj=np.zeros(23),
            dqj=np.zeros(23),
            base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        ref = object.__new__(PicoReferenceWindow)
        ref.idle_hold = True
        ref.robot = robot
        ref.client = SimpleNamespace(idle_capture_stationary=lambda: True)
        ref.robot_profile = G1_23DOF
        ref._stand_capture_requested = False
        ref._stand_capture_samples = deque(maxlen=50)
        ref._stand_capture_wait_ticks = 0
        ref.stand_capture_count = 0
        ref._stable_stand_rootz = 0.78
        ref._idle_holding = True
        ref._idle_hold_q_il = np.zeros(29)
        ref._idle_hold_rootz = 0.78
        self.assertTrue(ref.request_stable_stand_capture("test"))
        for _ in range(50):
            ref._update_stable_stand_capture()
        self.assertFalse(ref._stand_capture_requested)
        self.assertEqual(ref.stand_capture_count, 1)
        np.testing.assert_allclose(ref._stable_stand_q_il, np.zeros(29))
        self.assertAlmostEqual(ref._stable_stand_rootz, 0.78)

    def test_stand_capture_waits_if_physical_pose_is_moving(self):
        robot = SimpleNamespace(
            qj=np.zeros(23),
            dqj=np.full(23, 0.2),
            base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        ref = object.__new__(PicoReferenceWindow)
        ref.idle_hold = True
        ref.robot = robot
        ref.client = SimpleNamespace(idle_capture_stationary=lambda: True)
        ref.robot_profile = G1_23DOF
        ref._stand_capture_requested = False
        ref._stand_capture_samples = deque(maxlen=50)
        ref._stand_capture_wait_ticks = 0
        ref.stand_capture_count = 0
        ref._idle_holding = True
        ref._idle_hold_q_il = np.zeros(29)
        ref._idle_hold_rootz = 0.78
        self.assertTrue(ref.request_stable_stand_capture("test"))
        for _ in range(50):
            ref._update_stable_stand_capture()
        self.assertTrue(ref._stand_capture_requested)
        self.assertEqual(ref.stand_capture_count, 0)

    def test_forced_planner_profile_stays_g1_when_vr_fields_arrive(self):
        ref = object.__new__(PicoReferenceWindow)
        ref.force_g1 = True
        ref.force_smpl = False
        ref._vr_active = False
        ref._smpl_active = False
        ref._vr_input_present = False
        ref.client = type(
            "Client",
            (),
            {
                "age_s": lambda self: 0.01,
                "vr_3point": lambda self: (
                    np.zeros((3, 3)),
                    np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
                ),
            },
        )()
        ref._update_encoder_mode()
        self.assertTrue(ref._vr_input_present)
        self.assertFalse(ref._vr_active)
        self.assertEqual(ref.sonic_mode, "g1")

    def test_explicit_smpl_profile_reports_smpl_before_first_window(self):
        ref = object.__new__(PicoReferenceWindow)
        ref.force_g1 = False
        ref.force_smpl = True
        ref._smpl_active = False
        ref._vr_active = False
        self.assertEqual(ref.sonic_mode, "smpl")

    def test_unitree_joystick_standin_matches_planner_g1_semantics(self):
        try:
            from sonic.inputs.joystick_teleop import JoystickTeleop
            from sonic.inputs.pico_client import LocomotionMode
        except ModuleNotFoundError as exc:
            self.skipTest(f"deployment transport dependency is unavailable: {exc}")
        joystick = JoystickTeleop(locomotion_mode=LocomotionMode.RUN)
        joystick.set_axes(lx=0.0, ly=1.0, rx=1.0)
        joystick.tick(0.1)
        nav = joystick.nav_command()
        self.assertEqual(joystick.mode(), int(LocomotionMode.RUN))
        self.assertIsNone(joystick.vr_3point())
        self.assertEqual(joystick.navigation_frame, "world")
        self.assertAlmostEqual(nav["speed"], 4.5)
        np.testing.assert_allclose(nav["facing"], [np.cos(-0.15), np.sin(-0.15), 0.0], atol=1e-7)
        np.testing.assert_allclose(nav["movement"], nav["facing"], atol=1e-7)

    def test_unitree_joystick_yaw_rate_follows_gait_scale(self):
        try:
            from sonic.inputs.joystick_teleop import JoystickTeleop, YAW_GAIN
            from sonic.inputs.pico_client import LocomotionMode
        except ModuleNotFoundError as exc:
            self.skipTest(f"deployment transport dependency is unavailable: {exc}")
        joystick = JoystickTeleop(locomotion_mode=LocomotionMode.SLOW_WALK, yaw_command_scale=0.40)
        joystick.set_axes(lx=0.0, ly=1.0, rx=1.0)
        joystick.tick(0.1)
        nav = joystick.nav_command()
        expected = -YAW_GAIN * 0.40 * 0.1
        np.testing.assert_allclose(
            nav["facing"], [np.cos(expected), np.sin(expected), 0.0], atol=1e-7
        )

    def test_native_gamepad_yaw_rate_follows_gait_scale(self):
        try:
            from sonic.inputs.joystick_teleop import GamepadManagerTeleop
        except ModuleNotFoundError as exc:
            self.skipTest(f"deployment transport dependency is unavailable: {exc}")
        joystick = GamepadManagerTeleop()
        joystick.control_started = True
        joystick.yaw_command_scale = 0.40
        joystick.set_axes(lx=0.0, ly=1.0, rx=1.0)
        joystick.tick(0.1)
        expected = -joystick.MANAGER_YAW_GAIN * 0.40 * 0.1
        np.testing.assert_allclose(
            joystick.nav_command()["facing"],
            [np.cos(expected), np.sin(expected), 0.0],
            atol=1e-7,
        )

    def test_native_gamepad_inplace_yaw_can_exceed_walk_turn_scale(self):
        try:
            from sonic.inputs.joystick_teleop import GamepadManagerTeleop
        except ModuleNotFoundError as exc:
            self.skipTest(f"deployment transport dependency is unavailable: {exc}")
        joystick = GamepadManagerTeleop()
        joystick.control_started = True
        joystick.yaw_command_scale = 0.15
        joystick.yaw_inplace_scale = 0.35
        joystick.set_axes(lx=0.0, ly=0.0, rx=1.0)
        joystick.tick(0.1)
        expected = -joystick.MANAGER_YAW_GAIN * 0.35 * 0.1
        np.testing.assert_allclose(
            joystick.nav_command()["facing"],
            [np.cos(expected), np.sin(expected), 0.0],
            atol=1e-7,
        )
        joystick.yaw_rad = 0.0
        joystick.set_axes(lx=0.0, ly=1.0, rx=1.0)
        joystick.tick(0.1)
        expected = -joystick.MANAGER_YAW_GAIN * 0.15 * 0.1
        np.testing.assert_allclose(
            joystick.nav_command()["facing"],
            [np.cos(expected), np.sin(expected), 0.0],
            atol=1e-7,
        )

    def test_unitree_joystick_dpad_selects_modes_on_rising_edges(self):
        try:
            from sonic.inputs.joystick_teleop import JoystickTeleop
            from sonic.inputs.pico_client import LocomotionMode
            from core.robot.remote import KeyMap
        except ModuleNotFoundError as exc:
            self.skipTest(f"deployment transport dependency is unavailable: {exc}")
        remote = SimpleNamespace(lx=0.0, ly=0.0, rx=0.0, button=[0] * 16)
        joystick = JoystickTeleop(locomotion_mode=LocomotionMode.WALK)

        remote.button[KeyMap.up] = 1
        joystick.update_remote(remote)
        self.assertEqual(joystick.locomotion_mode, LocomotionMode.SLOW_WALK)
        self.assertEqual(joystick.mode(), LocomotionMode.IDLE)

        remote.button[KeyMap.up] = 0
        joystick.update_remote(remote)
        remote.button[KeyMap.down] = 1
        remote.ly = 1.0
        joystick.update_remote(remote)
        self.assertEqual(joystick.mode(), LocomotionMode.RUN)
        self.assertAlmostEqual(joystick.nav_command()["speed"], 4.5)

    def test_native_gamepad_manager_standing_fsm_and_system_buttons(self):
        try:
            from sonic.inputs.joystick_teleop import GamepadManagerTeleop
            from sonic.inputs.pico_client import LocomotionMode
            from core.robot.remote import KeyMap
        except ModuleNotFoundError as exc:
            self.skipTest(f"deployment transport dependency is unavailable: {exc}")

        remote = SimpleNamespace(lx=0.0, ly=0.0, rx=0.0, ry=0.0, button=[0] * 16)
        joystick = GamepadManagerTeleop()
        self.assertFalse(joystick.control_started)
        self.assertEqual(joystick.mode(), LocomotionMode.IDLE)
        joystick.tick(0.2)
        self.assertEqual(joystick.latest_seq, 0)

        remote.button[KeyMap.start] = 1
        joystick.update_remote(remote)
        self.assertTrue(joystick.control_started)
        joystick.tick(0.02)
        self.assertEqual(joystick.latest_seq, 1)
        remote.button[KeyMap.start] = 0
        joystick.update_remote(remote)

        remote.button[KeyMap.F1] = 1
        remote.button[KeyMap.F3] = 1
        joystick.update_remote(remote)
        self.assertTrue(joystick.consume_idle_capture_request())
        joystick.update_remote(remote)
        self.assertFalse(joystick.consume_idle_capture_request())
        remote.button[KeyMap.F1] = 0
        remote.button[KeyMap.F3] = 0
        joystick.update_remote(remote)

        remote.ly = 1.0
        remote.button[KeyMap.X] = 1
        joystick.update_remote(remote)
        self.assertEqual(joystick.mode(), LocomotionMode.WALK)
        self.assertEqual(joystick.nav_command()["speed"], -1.0)

        # Holding X is not another transition; only a new rising edge advances.
        joystick.update_remote(remote)
        self.assertEqual(joystick.mode(), LocomotionMode.WALK)
        remote.button[KeyMap.X] = 0
        joystick.update_remote(remote)
        remote.button[KeyMap.X] = 1
        joystick.update_remote(remote)
        self.assertEqual(joystick.mode(), LocomotionMode.RUN)
        self.assertEqual(joystick.nav_command()["speed"], 1.5)

        remote.button[KeyMap.X] = 0
        remote.button[KeyMap.Y] = 1
        joystick.update_remote(remote)
        self.assertEqual(joystick.mode(), LocomotionMode.WALK)
        remote.button[KeyMap.Y] = 0
        remote.button[KeyMap.B] = 1
        joystick.update_remote(remote)
        self.assertEqual(joystick.mode(), LocomotionMode.SLOW_WALK)
        self.assertEqual(joystick.nav_command()["speed"], 0.4)

        remote.button[KeyMap.B] = 0
        remote.button[KeyMap.A] = 1
        joystick.update_remote(remote)
        self.assertEqual(joystick.mode(), LocomotionMode.IDLE)
        self.assertFalse(joystick.stop_requested)
        remote.button[KeyMap.A] = 0
        remote.button[KeyMap.select] = 1
        joystick.update_remote(remote)
        self.assertTrue(joystick.stop_requested)

    def test_native_gamepad_manager_motion_set_selection_and_cycle(self):
        try:
            from sonic.inputs.joystick_teleop import GamepadManagerTeleop
            from sonic.inputs.pico_client import LocomotionMode
            from core.robot.remote import KeyMap
        except ModuleNotFoundError as exc:
            self.skipTest(f"deployment transport dependency is unavailable: {exc}")

        remote = SimpleNamespace(lx=0.0, ly=0.0, rx=0.0, ry=0.0, button=[0] * 16)
        joystick = GamepadManagerTeleop()
        remote.button[KeyMap.start] = 1
        joystick.update_remote(remote)
        remote.button[KeyMap.start] = 0
        remote.button[KeyMap.right] = 1
        joystick.update_remote(remote)
        self.assertEqual(joystick.motion_set_name, "STYLED_WALKING")
        self.assertEqual(joystick.locomotion_mode, LocomotionMode.LEDGE_WALKING)

        remote.button[KeyMap.right] = 0
        remote.button[KeyMap.X] = 1
        joystick.update_remote(remote)
        self.assertEqual(joystick.locomotion_mode, LocomotionMode.OBJECT_CARRYING)

        remote.button[KeyMap.X] = 0
        remote.button[KeyMap.up] = 1
        joystick.update_remote(remote)
        self.assertEqual(joystick.motion_set_name, "STANDING")
        self.assertEqual(joystick.locomotion_mode, LocomotionMode.SLOW_WALK)


class ReleasedModelSelectionTest(unittest.TestCase):
    def test_encoder_decoder_profiles_are_atomic(self):
        from core.config import Config
        from sonic.model.deployment_profile import resolve_python_model_profile

        config = Config("g1_23dof_sonic.yaml")
        expected_variants = {
            "original-release": "default",
            "low-latency": "low_latency",
            "sonic-v1.1": "sonic_v1_1",
        }
        for profile, variant in expected_variants.items():
            resolved = resolve_python_model_profile(profile)
            config.select_sonic_model_profile(profile)
            self.assertEqual(config.sonic_model_variant, variant)
            self.assertEqual(Path(config.sonic_encoder).resolve(), resolved["encoder"])
            self.assertEqual(Path(config.sonic_decoder).resolve(), resolved["decoder"])
            self.assertEqual(
                Path(config.sonic_observation_config).resolve(),
                resolved["observation_config"],
            )

    def test_live_unitree_type_must_match_selected_python_topology(self):
        for mode in (1, 4):
            self.assertTrue(G1_23DOF.matches_mode_machine(mode))
            self.assertFalse(G1_29DOF.matches_mode_machine(mode))
        for mode in (2, 5):
            self.assertTrue(G1_29DOF.matches_mode_machine(mode))
            self.assertFalse(G1_23DOF.matches_mode_machine(mode))
        for mode in (0, 3, 6, 255):
            self.assertFalse(G1_23DOF.matches_mode_machine(mode))
            self.assertFalse(G1_29DOF.matches_mode_machine(mode))
        with self.assertRaisesRegex(RuntimeError, "locked-waist"):
            G1_23DOF.require_mode_machine(3)

    def test_official_planner_wire_message_and_orientation_fallback(self):
        try:
            from sonic.inputs.pico_client import (
                DEFAULT_VR_ORIENTATION,
                HEADER_SIZE,
                PicoClient,
                decode_planner_message,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"deployment transport dependency is unavailable: {exc}")
        fields = [
            {"name": "mode", "dtype": "i32", "shape": [1]},
            {"name": "movement", "dtype": "f32", "shape": [3]},
            {"name": "facing", "dtype": "f32", "shape": [3]},
            {"name": "speed", "dtype": "f32", "shape": [1]},
            {"name": "height", "dtype": "f32", "shape": [1]},
            {"name": "vr_position", "dtype": "f32", "shape": [9]},
        ]
        header = (
            json.dumps(
                {
                    "v": 1,
                    "endian": "le",
                    "count": 1,
                    "fields": fields,
                },
                separators=(",", ":"),
            )
            .encode()
            .ljust(HEADER_SIZE, b"\0")
        )
        payload = b"".join(
            (
                struct.pack("<i", 1),
                struct.pack("<fff", 1.0, 0.0, 0.0),
                struct.pack("<fff", 1.0, 0.0, 0.0),
                struct.pack("<f", 0.2),
                struct.pack("<f", -1.0),
                struct.pack("<fffffffff", *np.arange(9, dtype=np.float32)),
            )
        )
        decoded = decode_planner_message(b"planner" + header + payload)

        client = object.__new__(PicoClient)
        client._lock = threading.Lock()
        client.latest = decoded
        client._latest_by_topic = {b"planner": decoded}
        pos, quat = client.vr_3point()
        np.testing.assert_array_equal(pos, np.arange(9).reshape(3, 3))
        np.testing.assert_allclose(quat, DEFAULT_VR_ORIENTATION)

    def test_one_second_stale_command_becomes_idle(self):
        ref = object.__new__(PicoReferenceWindow)
        ref.client = type("Client", (), {"age_s": lambda self: 1.01})()
        ref._stale = False
        ref._stale_logged = True
        ref._last_world_nav = {
            "movement": np.array([1.0, 0.0, 0.0]),
            "facing": np.array([0.0, 1.0, 0.0]),
            "speed": 0.2,
            "height": -1.0,
        }
        mode, nav = ref._current_command()
        self.assertEqual(mode, 0)
        np.testing.assert_array_equal(nav["movement"], np.zeros(3))
        np.testing.assert_array_equal(nav["facing"], [0.0, 1.0, 0.0])
        self.assertTrue(ref._stale)

    def test_real_pico_world_command_is_not_rotated(self):
        ref = object.__new__(PicoReferenceWindow)
        ref.client = type("Client", (), {"navigation_frame": "world"})()
        nav = {
            "movement": np.array([0.0, 1.0, 0.0]),
            "facing": np.array([0.0, 1.0, 0.0]),
            "speed": 0.2,
            "height": -1.0,
        }
        world = ref._to_planner_world(nav)
        np.testing.assert_allclose(world["movement"], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(world["facing"], [0.0, 1.0, 0.0])

    def test_only_local_standin_is_rotated_once(self):
        ref = object.__new__(PicoReferenceWindow)
        ref.client = type("Client", (), {"navigation_frame": "local"})()
        ref._lock = threading.Lock()
        ref.planner = type("Planner", (), {})()
        ref.planner.current_frame = 0
        ref.planner.current_motion = PlannerMotion(
            joint_pos_il=np.zeros((1, 29)),
            joint_vel_il=np.zeros((1, 29)),
            root_pos=np.zeros((1, 3)),
            root_quat_wxyz=np.array([[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]]),
        )
        nav = {
            "movement": np.array([1.0, 0.0, 0.0]),
            "facing": np.array([1.0, 0.0, 0.0]),
            "speed": 0.2,
            "height": -1.0,
        }
        world = ref._to_planner_world(nav)
        np.testing.assert_allclose(world["movement"], [0.0, 1.0, 0.0], atol=1e-7)


class PreparedClipTest(unittest.TestCase):
    def test_neutral_clip_matches_configured_present_pose(self):
        clip = np.load(ROOT / "sonic/assets/motions/motion_neutral_default.npz")
        expected = np.array(
            [
                -0.312,
                0.0,
                0.0,
                0.669,
                -0.363,
                0.0,
                -0.312,
                0.0,
                0.0,
                0.669,
                -0.363,
                0.0,
                0.0,
                0.2,
                0.2,
                0.0,
                0.6,
                0.0,
                0.2,
                -0.2,
                0.0,
                0.6,
                0.0,
            ]
        )
        np.testing.assert_allclose(
            G1_23DOF.isaaclab_to_physical(clip["dof_il"][0]), expected, atol=1e-7
        )
        self.assertEqual(float(np.abs(clip["dof_vel_il"]).max()), 0.0)

    def test_feedback_expands_23_joints_into_official_mujoco_slots(self):
        try:
            from sonic.sensing.robot_feedback import RobotFeedbackPublisher
        except ModuleNotFoundError as exc:
            self.skipTest(f"deployment transport dependency is unavailable: {exc}")
        q23 = np.arange(23, dtype=np.float64)
        publisher = object.__new__(RobotFeedbackPublisher)
        publisher.robot_profile = G1_23DOF
        q29 = publisher.expand_to_mujoco(q23)
        present_mj = np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                15,
                16,
                17,
                18,
                19,
                22,
                23,
                24,
                25,
                26,
            ]
        )
        missing_mj = np.array([13, 14, 20, 21, 27, 28])
        np.testing.assert_array_equal(q29[present_mj], q23)
        np.testing.assert_array_equal(q29[missing_mj], 0.0)


class EpisodeAnalysisTest(unittest.TestCase):
    def test_diagnostic_report_accepts_recorded_schema(self):
        from sonic.scripts.analysis.analyze_sonic_episode import report

        with tempfile.TemporaryDirectory() as tmp:
            ep = Path(tmp)
            q = np.tile([1.0, 0.0, 0.0, 0.0], (4, 1))
            np.savez(
                ep / "state.npz",
                loop_period_s=np.full(4, 0.02),
                control_compute_s=np.full(4, 0.002),
                low_state_age_s=np.full(4, 0.003),
                tracking_error_q_23=np.zeros((4, 23)),
                imu_quat_wxyz=q,
                command_saturated_23=np.zeros((4, 23), bool),
                sonic_missing_action_raw=np.zeros((4, 6)),
            )
            (ep / "meta.json").write_text(
                json.dumps(
                    {
                        "control_dt": 0.02,
                        "sonic_missing_action_order": list(G1_23DOF.missing_names),
                    }
                )
            )
            output = report(ep)
            self.assertIn("target 50.0 Hz", output)
            self.assertIn("absolute pelvis pitch", output)
            self.assertIn("waist_pitch", output)


if __name__ == "__main__":
    unittest.main()
