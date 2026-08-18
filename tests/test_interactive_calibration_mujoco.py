import unittest
from pathlib import Path

import mujoco
import numpy as np

from calibration.interactive import (
    GROUP_SELECTIONS,
    InteractiveCalibrationController,
    InteractiveCommand,
    SINGLE_SELECTIONS,
)
from calibration.mujoco_plant import SuspendedMujocoPlant
from calibration.protocol import PRESENT_23, SCREENING_AMPLITUDE


REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / "g1_deploy"


class InteractiveCalibrationMujocoTest(unittest.TestCase):
    def setUp(self):
        self.plant = SuspendedMujocoPlant(
            mjcf=REPO / "sonic_cpp/g1/scene_23dof.xml",
            config=DEPLOY / "configs/g1_23dof_sonic.yaml",
        )

    def advance(self, control, count):
        peak_offset = 0.0
        peak_tau = 0.0
        for _ in range(count):
            offset = control.step()
            state = self.plant.step(self.plant.neutral + offset)
            self.assertTrue(np.isfinite(state["q_measured"]).all())
            self.assertTrue(np.isfinite(state["dq_measured"]).all())
            self.assertTrue(np.isfinite(state["tau_est"]).all())
            peak_offset = max(peak_offset, float(np.max(np.abs(offset))))
            peak_tau = max(peak_tau, float(np.max(np.abs(state["tau_est"]))))
            self.assertEqual(self.plant.data.ncon, 0)
        return peak_offset, peak_tau

    def test_every_single_and_group_selection_remains_bounded_and_clear(self):
        control = InteractiveCalibrationController(max_envelope="screening")
        control.apply(InteractiveCommand.CAPTURE_NEUTRAL)
        screening_limit = max(SCREENING_AMPLITUDE.values())
        absent = np.setdiff1d(np.arange(29), PRESENT_23)

        visited = []
        for index in range(len(SINGLE_SELECTIONS)):
            if index:
                control.apply(InteractiveCommand.NEXT)
            visited.append(control.selection.name)
            control.apply(InteractiveCommand.EXECUTE)
            peak_offset, peak_tau = self.advance(control, 160)
            self.assertGreater(peak_offset, 0.0)
            self.assertLessEqual(peak_offset, screening_limit + 1e-10)
            self.assertGreater(peak_tau, 0.0)
            self.assertTrue(np.allclose(control.current_offset[absent], 0.0))
            control.apply(InteractiveCommand.NEUTRAL)
            self.advance(control, 160)
            self.assertLess(np.max(np.abs(control.current_offset)), 1e-9)
        self.assertEqual(visited, [item.name for item in SINGLE_SELECTIONS])

        control.apply(InteractiveCommand.GROUP_BANK)
        control.apply(InteractiveCommand.LOOP_MODE)
        visited = []
        for index in range(len(GROUP_SELECTIONS)):
            if index:
                control.apply(InteractiveCommand.NEXT)
            visited.append(control.selection.name)
            control.apply(InteractiveCommand.EXECUTE)
            peak_offset, peak_tau = self.advance(control, 100)
            self.assertGreater(peak_offset, 0.0)
            self.assertLessEqual(peak_offset, screening_limit + 1e-10)
            self.assertGreater(peak_tau, 0.0)
            self.assertTrue(np.allclose(control.current_offset[absent], 0.0))
            control.apply(InteractiveCommand.EXECUTE)
            self.advance(control, 160)
            self.assertLess(np.max(np.abs(control.current_offset)), 1e-9)
        self.assertEqual(visited, [item.name for item in GROUP_SELECTIONS])

        np.testing.assert_allclose(self.plant.data.qpos[:3], [0.0, 0.0, 1.2])
        np.testing.assert_allclose(self.plant.data.qpos[3:7], [1.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
