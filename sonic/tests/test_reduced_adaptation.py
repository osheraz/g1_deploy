import unittest
from pathlib import Path

import numpy as np

from core.config import Config
from sonic.model.reduced_adaptation import ReducedProprioceptionAdapter


class ReducedProprioceptionAdapterTest(unittest.TestCase):
    def _adapter(self, strategy="zero", settling=0.12):
        return ReducedProprioceptionAdapter(
            strategy=strategy,
            missing_idx=[5, 8],
            default_il=np.zeros(29),
            action_scale_il=np.ones(29) * 0.5,
            control_dt=0.02,
            settling_time_s=settling,
        )

    def test_zero_preserves_unavailable_zero_measurement(self):
        adapter = self._adapter("zero")
        q, dq = adapter.fill(
            np.ones((1, 29)),
            np.ones((1, 29)),
            reference_q_il=np.ones(29) * 3,
            reference_dq_il=np.ones(29) * 4,
        )
        np.testing.assert_array_equal(q[0, [5, 8]], [1, 1])
        np.testing.assert_array_equal(dq[0, [5, 8]], [1, 1])

    def test_reference_fills_only_missing_axes(self):
        adapter = self._adapter("reference")
        q, dq = adapter.fill(
            np.zeros((1, 29)),
            np.zeros((1, 29)),
            reference_q_il=np.arange(29),
            reference_dq_il=-np.arange(29),
        )
        np.testing.assert_array_equal(q[0, [5, 8]], [5, 8])
        np.testing.assert_array_equal(dq[0, [5, 8]], [-5, -8])
        self.assertEqual(q[0, 4], 0)

    def test_shadow_is_causal_and_converges(self):
        adapter = self._adapter("shadow", settling=0.12)
        action = np.zeros(29)
        action[[5, 8]] = [1.0, -1.0]
        q0, _ = adapter.diagnostics()
        np.testing.assert_array_equal(q0, [0, 0])
        for _ in range(50):
            adapter.advance(action)
        q, dq = adapter.diagnostics()
        np.testing.assert_allclose(q, [0.5, -0.5], atol=1e-4)
        np.testing.assert_allclose(dq, [0, 0], atol=2e-3)


class AdaptationConfigTest(unittest.TestCase):
    def test_official_config_remains_parity(self):
        cfg = Config("g1_23dof_sonic.yaml")
        self.assertEqual(cfg.sonic_adaptation_profile, "official-parity")
        self.assertFalse(cfg.sonic_mask_missing_reference)
        self.assertFalse(cfg.sonic_mask_missing_action_history)
        self.assertEqual(cfg.action_ema, 1.0)
        self.assertEqual(cfg.sonic_planner_random_seed, 1234)


    def test_forward_stable_profile_is_all_six_masks_and_direct(self):
        profile = (
            Path(__file__).resolve().parents[2]
            / "configs/adaptations/g1_23dof_forward_stable.yaml"
        )
        cfg = Config(profile)
        self.assertEqual(cfg.sonic_adaptation_profile, "forward-stable-23")
        self.assertTrue(cfg.sonic_mask_missing_reference)
        self.assertTrue(cfg.sonic_mask_missing_action_history)
        self.assertEqual(cfg.sonic_missing_adaptation_scope, "all")
        self.assertEqual(cfg.sonic_reduced_proprioception, "zero")
        self.assertEqual(cfg.action_ema, 1.0)
        np.testing.assert_array_equal(cfg.sonic_reference_joint_scale, np.ones(23))
        np.testing.assert_array_equal(cfg.sonic_motor_action_scale, np.ones(23))
        np.testing.assert_array_equal(cfg.sonic_motor_action_ema, np.ones(23))
        self.assertTrue(np.all(np.isinf(cfg.sonic_motor_action_abs_limit)))
        self.assertTrue(np.all(np.isinf(cfg.sonic_motor_target_rate_limit)))
        self.assertEqual(cfg.sonic_action_history_source, "raw")


if __name__ == "__main__":
    unittest.main()
