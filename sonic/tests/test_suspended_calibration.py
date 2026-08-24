import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sonic.calibration.joint_mapping import (
    JointCalibrationProfile, JointCurve, monotone_inverse_curve,
)
from sonic.calibration.protocol import PRESENT_23, build_plan, load_plan, save_plan


class SuspendedProtocolTest(unittest.TestCase):
    def test_screening_plan_is_one_joint_at_a_time_and_round_trips(self):
        plan = build_plan(protocol="screening", scope="legs")
        plan.validate()
        self.assertTrue(np.all(np.sum(np.abs(plan.offsets) > 1e-9, axis=1) <= 1))
        self.assertAlmostEqual(float(plan.offsets[0].max()), 0.0)
        self.assertAlmostEqual(float(plan.offsets[-1].max()), 0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = save_plan(plan, Path(directory) / "plan.csv")
            loaded = load_plan(path)
        np.testing.assert_allclose(loaded.offsets, plan.offsets, atol=1e-9)
        np.testing.assert_array_equal(loaded.active_motor, plan.active_motor)

    def test_all_scope_only_uses_present_motors(self):
        plan = build_plan(protocol="screening", scope="all")
        active = set(plan.active_motor[plan.active_motor >= 0].tolist())
        self.assertEqual(active, set(PRESENT_23.tolist()))


class JointMappingTest(unittest.TestCase):
    def test_monotone_inverse_rejects_noise_without_reversing(self):
        command = np.array([-1.0, -.5, 0.0, .5, 1.0])
        measured = np.array([-.8, -.2, -.25, .4, .75])
        knots, inverse = monotone_inverse_curve(command, measured)
        self.assertTrue(np.all(np.diff(knots) > 0))
        self.assertTrue(np.all(np.diff(inverse) >= 0))

    def test_direction_map_is_bounded_and_profile_round_trips(self):
        curve = JointCurve(
            [-1.0, 0.0, 1.0], [-1.1, .05, 1.1], [-1.0, -.05, 1.0], .08
        )
        self.assertAlmostEqual(curve.apply(1.0, 1), 1.08)
        self.assertAlmostEqual(curve.apply(-1.0, -1), -1.0)
        # Outside the measured domain, retain only the bounded endpoint
        # residual; never clamp the absolute target to the endpoint.
        self.assertAlmostEqual(curve.apply(2.0, 1), 2.08)
        profile = JointCalibrationProfile(
            "g1_23dof", "mujoco_truth", False, {4: curve}, {"test": True}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = profile.save(Path(directory) / "profile.json")
            loaded = JointCalibrationProfile.load(path)
        q = np.zeros(29)
        q[4] = 1.0
        self.assertAlmostEqual(loaded.apply_29(q, np.ones(29))[4], 1.08)

    def test_encoder_only_profile_cannot_self_approve(self):
        curve = JointCurve([-1, 1], [-1, 1], [-1, 1], .1)
        with self.assertRaisesRegex(ValueError, "encoder-only"):
            JointCalibrationProfile("g1_23dof", "encoder_only", True, {4: curve}, {})


if __name__ == "__main__":
    unittest.main()
