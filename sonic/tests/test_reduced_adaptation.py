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


class StableSlowProfileConfigTest(unittest.TestCase):
    def test_official_config_remains_parity(self):
        cfg = Config("g1_23dof_sonic.yaml")
        self.assertEqual(cfg.sonic_adaptation_profile, "official-parity")
        self.assertFalse(cfg.sonic_mask_missing_reference)
        self.assertFalse(cfg.sonic_mask_missing_action_history)
        self.assertEqual(cfg.action_ema, 1.0)
        self.assertEqual(cfg.sonic_planner_random_seed, 1234)

    def test_named_profile_has_qualified_minimal_motor_shaping(self):
        profile = (
            Path(__file__).resolve().parents[2]
            / "configs/adaptations/original_release_23dof_stable_slow.yaml"
        )
        cfg = Config(profile)
        self.assertEqual(cfg.sonic_adaptation_profile, "stable-slow-23")
        self.assertTrue(cfg.sonic_mask_missing_reference)
        self.assertTrue(cfg.sonic_mask_missing_action_history)
        self.assertEqual(cfg.sonic_missing_adaptation_scope, "all")
        self.assertEqual(cfg.sonic_reduced_proprioception, "zero")
        self.assertEqual(cfg.action_ema, 1.0)
        expected_scale = np.ones(23)
        expected_ema = np.ones(23)
        expected_scale[[5, 11]] = 0.95
        expected_ema[[4, 10]] = 0.95
        np.testing.assert_array_equal(cfg.sonic_motor_action_scale, expected_scale)
        np.testing.assert_array_equal(cfg.sonic_motor_action_ema, expected_ema)
        self.assertEqual(cfg.sonic_action_history_source, "raw")

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

    def test_forward_stable_alias_yaml_keeps_shared_contract(self):
        config_root = Path(__file__).resolve().parents[2] / "configs/adaptations"
        shared = Config(config_root / "g1_23dof_forward_stable.yaml")
        alias = Config(config_root / "original_release_23dof_forward_stable.yaml")
        self.assertEqual(
            alias.sonic_adaptation_profile, shared.sonic_adaptation_profile
        )
        self.assertEqual(
            alias.sonic_missing_adaptation_scope,
            shared.sonic_missing_adaptation_scope,
        )
        np.testing.assert_array_equal(
            alias.sonic_reference_joint_scale, shared.sonic_reference_joint_scale
        )

    def test_robust_small_gait_profile_is_reference_only(self):
        profile = (
            Path(__file__).resolve().parents[2]
            / "configs/adaptations/original_release_23dof_robust_small_gait.yaml"
        )
        cfg = Config(profile)
        self.assertEqual(cfg.sonic_adaptation_profile, "robust-small-gait-23")
        self.assertTrue(cfg.sonic_mask_missing_reference)
        self.assertTrue(cfg.sonic_mask_missing_action_history)
        self.assertEqual(cfg.sonic_missing_adaptation_scope, "wrist")
        expected = np.ones(23)
        expected[[0, 6]] = 0.80
        np.testing.assert_array_equal(cfg.sonic_reference_joint_scale, expected)
        np.testing.assert_array_equal(cfg.sonic_motor_action_scale, np.ones(23))
        np.testing.assert_array_equal(cfg.sonic_motor_action_ema, np.ones(23))
        self.assertTrue(np.all(np.isinf(cfg.sonic_motor_action_abs_limit)))
        self.assertTrue(np.all(np.isinf(cfg.sonic_motor_target_rate_limit)))
        self.assertEqual(cfg.sonic_action_history_source, "raw")

    def test_model_tuned_robust_profiles_are_atomic_and_direct(self):
        config_root = Path(__file__).resolve().parents[2] / "configs/adaptations"
        cases = (
            ("low_latency_23dof_robust_small_gait.yaml", 0.80, False),
            ("original_release_23dof_robust_small_gait.yaml", 0.80, True),
            ("sonic_v1_1_23dof_robust_small_gait.yaml", 0.90, True),
        )
        for filename, hip_scale, mask_history in cases:
            with self.subTest(filename=filename):
                cfg = Config(config_root / filename)
                self.assertEqual(cfg.sonic_adaptation_profile, "robust-small-gait-23")
                self.assertEqual(cfg.sonic_missing_adaptation_scope, "wrist")
                self.assertTrue(cfg.sonic_mask_missing_reference)
                self.assertEqual(cfg.sonic_mask_missing_action_history, mask_history)
                expected = np.ones(23)
                expected[[0, 6]] = hip_scale
                np.testing.assert_array_equal(cfg.sonic_reference_joint_scale, expected)
                np.testing.assert_array_equal(cfg.sonic_motor_action_scale, np.ones(23))
                np.testing.assert_array_equal(cfg.sonic_motor_action_ema, np.ones(23))


if __name__ == "__main__":
    unittest.main()
