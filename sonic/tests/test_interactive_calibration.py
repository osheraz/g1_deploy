import unittest

import numpy as np

from sonic.calibration.interactive import (
    GROUP_SELECTIONS,
    InteractiveCalibrationController,
    InteractiveCommand,
    SINGLE_SELECTIONS,
)
from sonic.calibration.protocol import FULL_AMPLITUDE, PRESENT_23, SCREENING_AMPLITUDE


class InteractiveCalibrationTest(unittest.TestCase):
    def test_banks_only_reference_present_motors(self):
        self.assertEqual(len(SINGLE_SELECTIONS), 23)
        for selection in (*SINGLE_SELECTIONS, *GROUP_SELECTIONS):
            self.assertTrue(set(selection.motors).issubset(set(PRESENT_23.tolist())))

    def test_screening_controller_never_exceeds_scaled_screening_envelope(self):
        control = InteractiveCalibrationController(max_envelope="screening")
        control.apply(InteractiveCommand.CAPTURE_NEUTRAL)
        control.apply(InteractiveCommand.SCALE_UP)
        control.apply(InteractiveCommand.SCALE_UP)
        control.apply(InteractiveCommand.EXECUTE)
        samples = np.asarray([control.step() for _ in range(300)])
        for motor in PRESENT_23:
            self.assertLessEqual(
                np.max(np.abs(samples[:, motor])),
                SCREENING_AMPLITUDE[int(motor)] + 1e-10,
            )

    def test_extended_range_is_only_available_when_explicit(self):
        screening = InteractiveCalibrationController(max_envelope="screening")
        screening.apply(InteractiveCommand.CAPTURE_NEUTRAL)
        screening.apply(InteractiveCommand.RANGE_UP)
        self.assertEqual(screening.range_name, "screening")
        full = InteractiveCalibrationController(max_envelope="full")
        full.apply(InteractiveCommand.CAPTURE_NEUTRAL)
        full.apply(InteractiveCommand.RANGE_UP)
        self.assertEqual(full.range_name, "extended")
        full.scale = 1.0
        amplitude = full.selection_amplitude()
        self.assertAlmostEqual(amplitude[0], FULL_AMPLITUDE[0])

    def test_selection_change_returns_neutral_before_switching(self):
        control = InteractiveCalibrationController()
        control.apply(InteractiveCommand.CAPTURE_NEUTRAL)
        control.apply(InteractiveCommand.EXECUTE)
        for _ in range(200):
            control.step()
        self.assertGreater(np.max(np.abs(control.current_offset)), 0.0)
        original = control.selection.name
        control.apply(InteractiveCommand.NEXT)
        self.assertTrue(control.status_dict()["pending_selection"])
        for _ in range(300):
            control.step()
        self.assertFalse(control.status_dict()["pending_selection"])
        self.assertNotEqual(control.selection.name, original)
        self.assertLess(np.max(np.abs(control.current_offset)), 1e-9)

    def test_new_selection_starts_with_positive_static_phase(self):
        control = InteractiveCalibrationController()
        control.apply(InteractiveCommand.CAPTURE_NEUTRAL)
        control.apply(InteractiveCommand.EXECUTE)
        for _ in range(200):
            control.step()
        control.apply(InteractiveCommand.NEUTRAL)
        for _ in range(200):
            control.step()
        control.apply(InteractiveCommand.NEXT)
        control.apply(InteractiveCommand.EXECUTE)
        for _ in range(200):
            control.step()
        motor = control.selection.motors[0]
        self.assertGreater(control.current_offset[motor], 0.0)

    def test_loop_starts_at_zero_and_abort_returns_neutral(self):
        control = InteractiveCalibrationController()
        control.apply(InteractiveCommand.CAPTURE_NEUTRAL)
        control.apply(InteractiveCommand.LOOP_MODE)
        control.apply(InteractiveCommand.EXECUTE)
        trajectory = np.asarray([control.step() for _ in range(500)])
        self.assertGreater(np.max(np.abs(trajectory)), 0.0)
        control.apply(InteractiveCommand.ABORT)
        for _ in range(300):
            control.step()
        self.assertTrue(control.aborted)
        self.assertLess(np.max(np.abs(control.current_offset)), 1e-9)

    def test_motion_is_locked_until_neutral_is_captured(self):
        control = InteractiveCalibrationController()
        control.apply(InteractiveCommand.EXECUTE)
        for _ in range(200):
            control.step()
        self.assertFalse(control.captured)
        self.assertEqual(np.max(np.abs(control.current_offset)), 0.0)
        control.apply(InteractiveCommand.CAPTURE_NEUTRAL)
        self.assertTrue(control.captured)
        control.apply(InteractiveCommand.EXECUTE)
        for _ in range(200):
            control.step()
        self.assertGreater(np.max(np.abs(control.current_offset)), 0.0)


if __name__ == "__main__":
    unittest.main()
