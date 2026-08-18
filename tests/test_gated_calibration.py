import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from calibration.gated_protocol import (
    DT, DYNAMIC_MAX_TARGET_RATE_RAD_S, FULL_SCREENING_AMPLITUDE,
    MAX_TARGET_RATE_RAD_S,
    build_review_protocol,
    load_review_protocol,
    save_review_protocol,
)
from calibration.mujoco_plant import SuspendedMujocoPlant
from calibration.protocol import PRESENT_23


REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / "g1_deploy"
ABSENT_23 = np.setdiff1d(np.arange(29), PRESENT_23)


class ReviewGatedProtocolTest(unittest.TestCase):
    def test_pilot_is_bilateral_right_first_and_round_trips(self):
        protocol = build_review_protocol("pilot")
        self.assertEqual(len(protocol.blocks), 20)
        self.assertAlmostEqual(protocol.motion_duration_s, 206.12)
        self.assertTrue(all(block.duration_s < 10.0 for block in protocol.blocks[1:13]))
        self.assertEqual(
            [block.name for block in protocol.blocks[1:13]],
            [
                "single_right_hip_pitch", "single_left_hip_pitch",
                "single_right_hip_roll", "single_left_hip_roll",
                "single_right_hip_yaw", "single_left_hip_yaw",
                "single_right_knee", "single_left_knee",
                "single_right_ankle_pitch", "single_left_ankle_pitch",
                "single_right_ankle_roll", "single_left_ankle_roll",
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_review_protocol(protocol, directory)
            loaded = load_review_protocol(path)
            manifest = json.loads(
                (Path(directory) / "protocol_manifest.json").read_text()
            )
        self.assertEqual(loaded.name, protocol.name)
        self.assertEqual(len(loaded.blocks), len(protocol.blocks))
        for expected, actual in zip(protocol.blocks, loaded.blocks):
            np.testing.assert_allclose(actual.offsets, expected.offsets, atol=1e-11)
            np.testing.assert_allclose(actual.local_time_s, expected.local_time_s)
            np.testing.assert_array_equal(actual.phase, expected.phase)
        self.assertEqual(manifest["embodiment"], "g1_23dof")
        self.assertIn("accelerometer", manifest["physical_sensor_log"]["imu"])
        self.assertIn(
            "foot contact force",
            manifest["physical_sensor_log"]["not_available_in_hg_lowstate"],
        )

    def test_screening_covers_exactly_the_physical_23_and_is_bounded(self):
        protocol = build_review_protocol("screening")
        self.assertEqual(len(protocol.blocks), 45)
        self.assertAlmostEqual(protocol.motion_duration_s, 1649.28)
        active = set()
        for block in protocol.blocks:
            block.validate()
            active.update(block.active_motors)
            rate_limit = (
                DYNAMIC_MAX_TARGET_RATE_RAD_S
                if block.kind.startswith("dynamic_")
                else MAX_TARGET_RATE_RAD_S
            )
            self.assertLessEqual(
                np.max(np.abs(np.diff(block.offsets, axis=0))) / DT,
                rate_limit + 1e-9,
            )
            np.testing.assert_allclose(block.offsets[[0, -1]], 0.0, atol=1e-12)
            np.testing.assert_allclose(block.offsets[:, ABSENT_23], 0.0, atol=1e-12)
            for motor in PRESENT_23:
                self.assertLessEqual(
                    np.max(np.abs(block.offsets[:, motor])),
                    FULL_SCREENING_AMPLITUDE[int(motor)] + 1e-9,
                )
        self.assertEqual(active, set(PRESENT_23.tolist()))
        blocks = {block.name: block for block in protocol.blocks}
        self.assertAlmostEqual(
            blocks["single_right_knee"].peak_offsets[9], FULL_SCREENING_AMPLITUDE[9]
        )
        self.assertAlmostEqual(
            blocks["single_right_ankle_pitch"].peak_offsets[10],
            FULL_SCREENING_AMPLITUDE[10],
        )
        self.assertAlmostEqual(
            blocks["group_bilateral_hip_roll"].peak_offsets[1], 0.144
        )
        dynamic = [block for block in protocol.blocks if block.kind.startswith("dynamic_")]
        self.assertEqual(len(dynamic), 14)
        self.assertGreater(
            max(np.max(np.abs(np.diff(block.offsets, axis=0))) / DT for block in dynamic),
            1.9,
        )
        self.assertTrue(any("hz_2.20" in phase for phase in blocks[
            "dynamic_left_knee"
        ].phase))

    def test_v3_review_log_fitter_uses_captured_neutral(self):
        fields = ["schema", "phase", "active_motor", "direction", "block_kind"]
        fields += [
            f"{prefix}_{motor}"
            for prefix in (
                "target_model", "command_hardware", "q_measured",
                "captured_neutral",
            )
            for motor in range(29)
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            state = directory / "state.csv"
            output = directory / "profile.json"
            with state.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for direction, command in ((-1, -0.05), (1, 0.05)):
                    for _ in range(15):
                        row = {
                            "schema": "e_skin.suspended_calibration_log.v3",
                            "phase": "dwell_test",
                            "active_motor": -1,
                            "direction": 0,
                            "block_kind": "single_joint",
                        }
                        for motor in range(29):
                            value = command if motor == 4 else 0.0
                            row[f"target_model_{motor}"] = value
                            row[f"command_hardware_{motor}"] = value
                            row[f"q_measured_{motor}"] = (
                                value - direction * 0.005 if motor == 4 else 0.0
                            )
                            row[f"captured_neutral_{motor}"] = 0.0
                        writer.writerow(row)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(DEPLOY / "scripts/fit_suspended_calibration.py"),
                    str(state),
                    "--coordinate-basis", "encoder_only",
                    "--output", str(output),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            profile = json.loads(output.read_text())
        self.assertIn("joints=1", completed.stdout)
        self.assertIn("4", profile["joints"])
        self.assertFalse(profile["approved_for_deployment"])


class TwoShoulderFixtureTest(unittest.TestCase):
    def test_compliant_shoulders_preserve_tilt_and_keep_feet_clear(self):
        angle = np.radians(5.0) / 2.0
        plant = SuspendedMujocoPlant(
            mjcf=REPO / "sonic_cpp/g1/scene_23dof.xml",
            config=DEPLOY / "configs/g1_23dof_sonic.yaml",
            suspension="two_shoulder",
            root_quat_wxyz=np.array([np.cos(angle), np.sin(angle), 0.0, 0.0]),
        )
        state = None
        max_force = 0.0
        for _ in range(500):
            state = plant.step(plant.neutral)
            max_force = max(
                max_force,
                float(np.max(np.linalg.norm(state["suspension_forces"], axis=1))),
            )
            self.assertEqual(int(state["contact_count"]), 0)
        assert state is not None
        for key in (
            "q_measured", "dq_measured", "tau_est", "root_position",
            "root_quaternion", "root_linear_velocity", "root_angular_velocity",
            "imu_quaternion", "imu_gyroscope", "suspension_forces",
            "foot_link_height",
        ):
            self.assertTrue(np.isfinite(state[key]).all(), key)
        self.assertGreater(float(np.min(state["foot_link_height"])), 0.40)
        self.assertLess(float(np.linalg.norm(state["root_angular_velocity"])), 0.02)
        self.assertGreater(max_force, 100.0)
        self.assertLess(max_force, 600.0)
        # A two-point shoulder fixture must not silently overwrite the sensed
        # hanging attitude with the level pelvis assumption used by the legacy
        # fixed-root test plant.
        self.assertGreater(abs(float(state["root_quaternion"][1])), 0.01)


if __name__ == "__main__":
    unittest.main()
