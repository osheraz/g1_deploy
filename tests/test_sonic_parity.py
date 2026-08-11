from pathlib import Path
import json
import struct
import sys
import threading
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy.sonic import obs, robot_g1  # noqa: E402
from policy.sonic.model_profile import MODEL_PROFILES  # noqa: E402
from policy.sonic.robot_profile import G1_23DOF, G1_29DOF  # noqa: E402
from policy.sonic.planner_onnx import (  # noqa: E402
    PlannerMotion,
    SonicKinematicPlanner,
)
from policy.pico_reference import LOWER_BODY_IL_IDX, PicoReferenceWindow  # noqa: E402


class EncoderMaskTest(unittest.TestCase):
    def setUp(self):
        self.q = np.ones((1, 10, 29))
        self.dq = np.full((1, 10, 29), 2.0)
        self.rootq = np.zeros((1, 10, 4))
        self.rootq[..., 0] = 1.0
        self.body_pos = np.ones((1, 14, 3))
        self.body_quat = np.zeros((1, 14, 4))
        self.body_quat[..., 0] = 1.0

    def build(self, mode):
        kwargs = {}
        if mode == "smpl":
            kwargs = {
                "smpl_joints_fut": np.ones((1, 10, 24, 3)),
                "smpl_rootq_fut": self.rootq.copy(),
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
            np.ones((1, 10, 6)),
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
            np.testing.assert_array_equal(x[0, -sf * 6:].reshape(sf, 6)[:, 2:], 0.0)


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
        ref._last_cmd = (0, {})
        ref._idle_readapt_original = np.zeros(29)
        ref._idle_readapt_stored = False
        ref._idle_readapt_state = "idle"
        ref.robot_profile = G1_23DOF
        ref._idle_readapt_at_horizon()
        np.testing.assert_allclose(
            motion.joint_pos_il[-1, LOWER_BODY_IL_IDX], 0.004, atol=1e-12
        )
        self.assertEqual(ref._idle_readapt_state, "adapting")


class PicoSemanticsTest(unittest.TestCase):
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
            from common.joystick_teleop import JoystickTeleop
            from common.pico_client import LocomotionMode
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
        np.testing.assert_allclose(
            nav["facing"], [np.cos(-0.15), np.sin(-0.15), 0.0], atol=1e-7
        )
        np.testing.assert_allclose(nav["movement"], nav["facing"], atol=1e-7)

    def test_official_planner_wire_message_and_orientation_fallback(self):
        try:
            from common.pico_client import (
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
        header = json.dumps({
            "v": 1, "endian": "le", "count": 1, "fields": fields,
        }, separators=(",", ":")).encode().ljust(HEADER_SIZE, b"\0")
        payload = b"".join((
            struct.pack("<i", 1),
            struct.pack("<fff", 1.0, 0.0, 0.0),
            struct.pack("<fff", 1.0, 0.0, 0.0),
            struct.pack("<f", 0.2),
            struct.pack("<f", -1.0),
            struct.pack("<fffffffff", *np.arange(9, dtype=np.float32)),
        ))
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
        clip = np.load(ROOT / "policy/sonic/motion_neutral_default.npz")
        expected = np.array([
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
            0.0, 0.2, 0.2, 0.0, 0.6, 0.0,
            0.2, -0.2, 0.0, 0.6, 0.0,
        ])
        np.testing.assert_allclose(
            G1_23DOF.isaaclab_to_physical(clip["dof_il"][0]), expected, atol=1e-7
        )
        self.assertEqual(float(np.abs(clip["dof_vel_il"]).max()), 0.0)

    def test_feedback_expands_23_joints_into_official_mujoco_slots(self):
        try:
            from common.robot_feedback import RobotFeedbackPublisher
        except ModuleNotFoundError as exc:
            self.skipTest(f"deployment transport dependency is unavailable: {exc}")
        q23 = np.arange(23, dtype=np.float64)
        publisher = object.__new__(RobotFeedbackPublisher)
        publisher.robot_profile = G1_23DOF
        q29 = publisher.expand_to_mujoco(q23)
        present_mj = np.array([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            15, 16, 17, 18, 19, 22, 23, 24, 25, 26,
        ])
        missing_mj = np.array([13, 14, 20, 21, 27, 28])
        np.testing.assert_array_equal(q29[present_mj], q23)
        np.testing.assert_array_equal(q29[missing_mj], 0.0)


class EpisodeAnalysisTest(unittest.TestCase):
    def test_diagnostic_report_accepts_recorded_schema(self):
        from scripts.analyze_sonic_episode import report

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
            (ep / "meta.json").write_text(json.dumps({
                "control_dt": 0.02,
                "sonic_missing_action_order": list(G1_23DOF.missing_names),
            }))
            output = report(ep)
            self.assertIn("target 50.0 Hz", output)
            self.assertIn("absolute pelvis pitch", output)
            self.assertIn("waist_pitch", output)


if __name__ == "__main__":
    unittest.main()
