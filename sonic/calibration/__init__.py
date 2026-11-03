"""Suspended-joint identification and bounded command calibration.

The calibration package is deliberately independent of SONIC inference.  It
identifies the plant at the last boundary before joint PD and emits an opt-in
profile; it never changes a released encoder, decoder, planner, action history,
or reference trajectory.
"""

from sonic.calibration.joint_mapping import JointCalibrationProfile
from sonic.calibration.protocol import CalibrationPlan, build_plan, load_plan, save_plan

__all__ = [
    "CalibrationPlan",
    "JointCalibrationProfile",
    "build_plan",
    "load_plan",
    "save_plan",
]
