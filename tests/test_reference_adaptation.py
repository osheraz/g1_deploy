import unittest

import numpy as np

from policy.sonic.reference_adaptation import GaitReferenceAdapter


class GaitReferenceAdapterTest(unittest.TestCase):
    def _reference(self, *, active=True):
        q = np.arange(58, dtype=np.float32).reshape(1, 2, 29) / 10.0
        dq = np.ones_like(q) * 2.0
        return {
            "dof_fut": q,
            "dvel_fut": dq,
            "rootz_fut": np.array([[0.7, 0.9]], np.float32),
            "rootz_cur": np.array([0.8], np.float32),
            "stable_stand_q_il": np.arange(29, dtype=np.float32) / 20.0,
            "stable_stand_root_z": np.float32(0.75),
            "reference_gait_active": active,
        }

    def test_active_gait_scales_only_selected_reference_channels(self):
        source = self._reference()
        original_q = source["dof_fut"].copy()
        adapter = GaitReferenceAdapter(
            leg_idx=[0, 3, 6],
            leg_scale=0.5,
            root_height_scale=0.5,
            default_stand_q_il=np.zeros(29),
        )
        out = adapter.apply(source)
        stand = source["stable_stand_q_il"][[0, 3, 6]]
        expected = stand + 0.5 * (original_q[..., [0, 3, 6]] - stand)
        np.testing.assert_allclose(out["dof_fut"][..., [0, 3, 6]], expected)
        np.testing.assert_allclose(out["dvel_fut"][..., [0, 3, 6]], 1.0)
        np.testing.assert_array_equal(out["dof_fut"][..., 1], original_q[..., 1])
        np.testing.assert_allclose(out["rootz_fut"], [[0.725, 0.825]])
        np.testing.assert_allclose(out["rootz_cur"], [0.775])
        np.testing.assert_array_equal(source["dof_fut"], original_q)
        self.assertTrue(adapter.last_active)

    def test_idle_reference_is_bitwise_unchanged(self):
        source = self._reference(active=False)
        adapter = GaitReferenceAdapter(
            leg_idx=[0, 3, 6],
            leg_scale=0.5,
            default_stand_q_il=np.zeros(29),
        )
        self.assertIs(adapter.apply(source), source)
        self.assertFalse(adapter.last_active)

    def test_per_joint_scale_preserves_unselected_weight_shift_axes(self):
        source = self._reference()
        original_q = source["dof_fut"].copy()
        scales = np.ones(29)
        scales[[0, 3, 6]] = [0.5, 0.7, 0.9]
        adapter = GaitReferenceAdapter(
            leg_idx=[0, 1, 3, 4, 6, 7],
            default_stand_q_il=np.zeros(29),
            joint_scale_il=scales,
        )
        out = adapter.apply(source)
        stand = source["stable_stand_q_il"]
        for index in (0, 3, 6):
            np.testing.assert_allclose(
                out["dof_fut"][..., index],
                stand[index] + scales[index] * (
                    original_q[..., index] - stand[index]
                ),
            )
        for index in (1, 4, 7):
            np.testing.assert_array_equal(
                out["dof_fut"][..., index], original_q[..., index]
            )

    def test_neutral_adapter_is_bitwise_pass_through(self):
        source = self._reference(active=True)
        adapter = GaitReferenceAdapter(
            leg_idx=np.arange(12),
            default_stand_q_il=np.zeros(29),
        )
        self.assertIs(adapter.apply(source), source)
        self.assertFalse(adapter.last_active)


if __name__ == "__main__":
    unittest.main()
