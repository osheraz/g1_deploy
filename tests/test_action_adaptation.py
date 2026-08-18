import unittest

import numpy as np

from policy.sonic.action_adaptation import MotorActionAdapter


class MotorActionAdapterTest(unittest.TestCase):
    def test_neutral_path_is_exact(self):
        adapter = MotorActionAdapter(action_scale=[0.5, 0.25], control_dt=0.02)
        raw = np.array([2.0, -3.0], np.float32)
        np.testing.assert_array_equal(adapter.apply(raw), raw)
        self.assertFalse(adapter.enabled)

    def test_scale_clip_ema_then_rate_limit(self):
        adapter = MotorActionAdapter(
            action_scale=[0.5, 0.25],
            control_dt=0.02,
            scale=[0.5, 2.0],
            abs_limit=[0.8, 3.0],
            ema=[1.0, 0.5],
            target_rate_limit=[10.0, 5.0],
        )
        # scale/clip -> [0.8, 3], EMA -> [0.8, 1.5], rate limits in
        # normalized units -> [0.4, 0.4].
        np.testing.assert_allclose(adapter.apply([2.0, 2.0]), [0.4, 0.4])
        np.testing.assert_allclose(adapter.apply([2.0, 2.0]), [0.8, 0.8])

    def test_reset_clears_causal_state(self):
        adapter = MotorActionAdapter(
            action_scale=[1.0], control_dt=0.1, ema=[0.5]
        )
        np.testing.assert_allclose(adapter.apply([1.0]), [0.5])
        adapter.reset()
        np.testing.assert_allclose(adapter.apply([1.0]), [0.5])

    def test_invalid_parameters_fail(self):
        with self.assertRaises(ValueError):
            MotorActionAdapter(action_scale=[0.5], control_dt=0.02, ema=[0.0])
        with self.assertRaises(ValueError):
            MotorActionAdapter(
                action_scale=[0.5], control_dt=0.02, target_rate_limit=[-1.0]
            )


if __name__ == "__main__":
    unittest.main()
