import unittest

import numpy as np

from sonic.model.reference_adaptation import (
    GaitReferenceAdapter,
    LandingHoldAdapter,
    LEFT_SAGITTAL_IL,
    SAGITTAL_PHYSICAL_IDX,
    SLOW_GAIT_KNEE_OFFSET_DEG,
    SLOW_GAIT_SAGITTAL_SCALE,
    SLOW_GAIT_YAW_COMMAND_SCALE,
    SLOW_GAIT_YAW_INPLACE_SCALE,
    expand_slow_gait_recipe,
    gait_yaw_command_scale,
)
from sonic.model.robot_profile import G1_23DOF


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

    def test_gait_offset_is_added_after_position_scale(self):
        source = self._reference()
        scales = np.ones(29)
        offset = np.zeros(29)
        scales[3] = 0.30
        offset[3] = 0.10
        adapter = GaitReferenceAdapter(
            leg_idx=[3],
            default_stand_q_il=np.zeros(29),
            joint_scale_il=scales,
            joint_offset_il=offset,
        )
        out = adapter.apply(source)
        stand = source["stable_stand_q_il"][3]
        expected = stand + 0.30 * (source["dof_fut"][..., 3] - stand) + 0.10
        np.testing.assert_allclose(out["dof_fut"][..., 3], expected, atol=1e-6)

    def test_dvel_scale_can_stay_larger_than_position_scale(self):
        source = self._reference()
        q_scale = np.ones(29)
        dq_scale = np.ones(29)
        q_scale[[0, 3, 6]] = 0.30
        dq_scale[[0, 3, 6]] = 0.70
        adapter = GaitReferenceAdapter(
            leg_idx=[0, 3, 6],
            default_stand_q_il=np.zeros(29),
            joint_scale_il=q_scale,
            joint_dvel_scale_il=dq_scale,
        )
        out = adapter.apply(source)
        stand = source["stable_stand_q_il"][[0, 3, 6]]
        expected = stand + 0.30 * (source["dof_fut"][..., [0, 3, 6]] - stand)
        np.testing.assert_allclose(out["dof_fut"][..., [0, 3, 6]], expected)
        np.testing.assert_allclose(out["dvel_fut"][..., [0, 3, 6]], 1.4)
        np.testing.assert_array_equal(out["dvel_fut"][..., 1], 2.0)

    def test_flexion_scale_preserves_only_the_flexed_side_of_stand(self):
        source = self._reference()
        source["dof_fut"] = source["dof_fut"].copy()
        source["dof_fut"][..., 9] = np.array([0.10, 0.80], np.float32)
        source["stable_stand_q_il"] = source["stable_stand_q_il"].copy()
        source["stable_stand_q_il"][9] = 0.40
        scales = np.ones(29)
        flex = np.ones(29)
        scales[9] = 0.28
        flex[9] = 0.60
        adapter = GaitReferenceAdapter(
            leg_idx=[9],
            default_stand_q_il=np.zeros(29),
            joint_scale_il=scales,
            joint_flex_scale_il=flex,
        )
        out = adapter.apply(source)
        np.testing.assert_allclose(
            out["dof_fut"][..., 9],
            [[0.40 + 0.28 * (0.10 - 0.40), 0.40 + 0.60 * (0.80 - 0.40)]],
        )
        np.testing.assert_allclose(out["dvel_fut"][..., 9], [[0.56, 1.2]])

    def test_slow_gait_recipe_matches_cpp_and_allows_yaw_overrides(self):
        self.assertEqual(SLOW_GAIT_SAGITTAL_SCALE, 0.28)
        self.assertEqual(SLOW_GAIT_KNEE_OFFSET_DEG, 8.0)
        self.assertEqual(SLOW_GAIT_YAW_COMMAND_SCALE, 0.15)
        self.assertEqual(SLOW_GAIT_YAW_INPLACE_SCALE, 0.35)
        scales, offsets, walk_yaw, inplace_yaw = expand_slow_gait_recipe()
        self.assertEqual(scales, ["sagittal=0.28"])
        self.assertEqual(offsets, ["knee=8"])
        self.assertEqual(walk_yaw, 0.15)
        self.assertEqual(inplace_yaw, 0.35)
        scales, offsets, walk_yaw, inplace_yaw = expand_slow_gait_recipe(
            joint_offset_specs=["knee=6"],
            yaw_command_scale=0.18,
            yaw_inplace_scale=0.40,
        )
        self.assertEqual(offsets, ["knee=6"])
        self.assertEqual(walk_yaw, 0.18)
        self.assertEqual(inplace_yaw, 0.40)
        with self.assertRaises(ValueError):
            expand_slow_gait_recipe(joint_scale_specs=["sagittal=0.33"])
        with self.assertRaises(ValueError):
            expand_slow_gait_recipe(joint_scale_specs=["hip_pitch=0.85"])

    def test_physical_sagittal_maps_to_cpp_isaaclab_slots(self):
        physical = np.ones(23)
        for index in SAGITTAL_PHYSICAL_IDX:
            physical[index] = 0.28
        il = G1_23DOF.physical_to_isaaclab(physical, fill=1.0)
        for index in (0, 1, 9, 10, 13, 14):
            self.assertEqual(il[index], 0.28)
        self.assertEqual(il[2], 1.0)
        self.assertEqual(il[3], 1.0)
        self.assertEqual(il[4], 1.0)

    def test_gait_yaw_command_scale_follows_smallest_sagittal_joint(self):
        healthy = np.ones(23)
        self.assertEqual(gait_yaw_command_scale(healthy), 1.0)
        sagittal = healthy.copy()
        sagittal[[0, 3, 4, 6, 9, 10]] = 0.40
        self.assertAlmostEqual(gait_yaw_command_scale(sagittal), 0.40)
        hip_only = healthy.copy()
        hip_only[[0, 6]] = 0.85
        self.assertAlmostEqual(gait_yaw_command_scale(hip_only), 0.85)
        self.assertEqual(gait_yaw_command_scale(np.ones(11)), 1.0)

    def test_landing_hold_freezes_landed_leg_until_clip_stance(self):
        hold = LandingHoldAdapter(
            enabled=True,
            confirm_down_steps=2,
            min_swing_steps=2,
            confirm_release_steps=2,
        )
        measured = np.linspace(0.1, 0.4, 29)
        raw = np.zeros((1, 2, 29), np.float32)
        raw[..., 1] = 0.40
        raw[..., 3] = 0.05
        source = self._reference()
        source["dof_fut"] = raw.copy()
        source["dvel_fut"] = np.ones_like(raw)
        airborne = hold.apply(
            source,
            measured_q_il=measured,
            raw_dof_fut=raw,
            foot_z=(0.06, 0.01),
            foot_contact=(False, True),
        )
        airborne = hold.apply(
            airborne,
            measured_q_il=measured,
            raw_dof_fut=raw,
            foot_z=(0.06, 0.01),
            foot_contact=(False, True),
        )
        self.assertFalse(hold.last_holding[0])
        landed = hold.apply(
            source,
            measured_q_il=measured,
            raw_dof_fut=raw,
            foot_z=(0.01, 0.01),
            foot_contact=(True, True),
        )
        landed = hold.apply(
            landed if landed is not source else source,
            measured_q_il=measured,
            raw_dof_fut=raw,
            foot_z=(0.01, 0.01),
            foot_contact=(True, True),
        )
        self.assertTrue(hold.last_holding[0])
        np.testing.assert_allclose(
            landed["dof_fut"][..., list(LEFT_SAGITTAL_IL)],
            np.broadcast_to(measured[list(LEFT_SAGITTAL_IL)], (1, 2, 3)),
        )
        np.testing.assert_array_equal(
            landed["dvel_fut"][..., list(LEFT_SAGITTAL_IL)], 0.0
        )
        np.testing.assert_array_equal(
            landed["dof_fut"][..., 6], raw[..., 6]
        )
        np.testing.assert_array_equal(source["dof_fut"], raw)
        stance = raw.copy()
        stance[..., 1] = 0.05
        stance[..., 3] = 0.40
        source["dof_fut"] = stance
        released = hold.apply(
            source,
            measured_q_il=measured,
            raw_dof_fut=stance,
            foot_z=(0.01, 0.01),
            foot_contact=(True, True),
        )
        released = hold.apply(
            released if released is not source else source,
            measured_q_il=measured,
            raw_dof_fut=stance,
            foot_z=(0.01, 0.01),
            foot_contact=(True, True),
        )
        self.assertFalse(hold.last_holding[0])
        self.assertFalse(np.array_equal(
            released["dof_fut"][..., list(LEFT_SAGITTAL_IL)],
            measured[list(LEFT_SAGITTAL_IL)],
        ))

    def test_landing_hold_dvel_zeros_velocity_and_keeps_clip_q(self):
        hold = LandingHoldAdapter(
            enabled=True,
            mode="dvel",
            confirm_down_steps=2,
            min_swing_steps=2,
            confirm_release_steps=2,
        )
        measured = np.linspace(0.1, 0.4, 29)
        raw = np.zeros((1, 2, 29), np.float32)
        raw[..., 1] = 0.40
        raw[..., 3] = 0.05
        source = self._reference()
        source["dof_fut"] = raw.copy()
        source["dvel_fut"] = np.ones_like(raw)
        for _ in range(2):
            hold.apply(
                source,
                measured_q_il=measured,
                raw_dof_fut=raw,
                foot_z=(0.06, 0.06),
                foot_contact=(False, False),
            )
        for _ in range(2):
            source = hold.apply(
                source,
                measured_q_il=measured,
                raw_dof_fut=raw,
                foot_z=(0.01, 0.06),
                foot_contact=(True, False),
            )
        self.assertTrue(hold.last_holding[0])
        np.testing.assert_array_equal(
            source["dvel_fut"][..., list(LEFT_SAGITTAL_IL)], 0.0
        )
        np.testing.assert_array_equal(
            source["dof_fut"][..., list(LEFT_SAGITTAL_IL)],
            raw[..., list(LEFT_SAGITTAL_IL)],
        )

    def test_landing_hold_times_out_while_clip_still_swings(self):
        hold = LandingHoldAdapter(
            enabled=True,
            mode="dvel",
            confirm_down_steps=1,
            min_swing_steps=1,
            confirm_release_steps=3,
            max_hold_steps=4,
        )
        measured = np.linspace(0.1, 0.4, 29)
        raw = np.zeros((1, 2, 29), np.float32)
        raw[..., 1] = 0.40
        raw[..., 3] = 0.05
        source = self._reference()
        source["dof_fut"] = raw.copy()
        source["dvel_fut"] = np.ones_like(raw)
        hold.apply(
            source,
            measured_q_il=measured,
            raw_dof_fut=raw,
            foot_z=(0.06, 0.06),
            foot_contact=(False, False),
        )
        for _ in range(4):
            source = hold.apply(
                source,
                measured_q_il=measured,
                raw_dof_fut=raw,
                foot_z=(0.01, 0.06),
                foot_contact=(True, False),
            )
        self.assertTrue(hold.last_holding[0])
        source = self._reference()
        source["dof_fut"] = raw.copy()
        source["dvel_fut"] = np.ones_like(raw)
        released = hold.apply(
            source,
            measured_q_il=measured,
            raw_dof_fut=raw,
            foot_z=(0.01, 0.06),
            foot_contact=(True, False),
        )
        self.assertFalse(hold.last_holding[0])
        self.assertIs(released, source)
        np.testing.assert_array_equal(
            released["dvel_fut"][..., list(LEFT_SAGITTAL_IL)], 1.0
        )

    def test_disabled_landing_hold_is_bitwise_pass_through(self):
        source = self._reference()
        hold = LandingHoldAdapter(enabled=False)
        self.assertIs(
            hold.apply(source, measured_q_il=np.zeros(29)),
            source,
        )


if __name__ == "__main__":
    unittest.main()
