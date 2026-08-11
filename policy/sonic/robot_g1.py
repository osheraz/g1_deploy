"""Vendored from the SONIC research repo so this deploy is standalone.
Source: sonic/robot_g1.py
Pure numpy; do not edit here -- re-copy if upstream changes.
"""
"""G1 29-DOF joint-order maps for the SONIC ONNX (ONNX = IsaacLab order, our mjcf = MuJoCo; convert
with il_to_mj/mj_to_il). Gains/default/action_scale live in config/robot/g1/g1_29dof.yaml."""

import numpy as np

NUM_DOF = 29

# Gather maps (the names follow the official C++ constants):
#   mujoco_vector = isaaclab_vector[ISAACLAB_TO_MUJOCO_DOF]
#   isaaclab_vector = mujoco_vector[MUJOCO_TO_ISAACLAB_DOF]
ISAACLAB_TO_MUJOCO_DOF = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28])
MUJOCO_TO_ISAACLAB_DOF = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])

# 14 tracked bodies, IsaacLab order (vr/anchor obs)
BODY_NAMES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]
ANCHOR_BODY = "pelvis"

def mj_to_il(x):
    return np.asarray(x)[..., MUJOCO_TO_ISAACLAB_DOF]


def il_to_mj(x):
    return np.asarray(x)[..., ISAACLAB_TO_MUJOCO_DOF]
