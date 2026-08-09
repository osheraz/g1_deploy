#!/usr/bin/env python3
"""Hold the robot limp so you can move the joints by hand.

    python scripts/free_torque.py            # kd=8, joints resist a little
    python scripts/free_torque.py --zero     # kd=0, fully free

HANG THE ROBOT FIRST. It will drop.  Ctrl-C to stop.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from common.command_helper import create_damping_cmd, create_zero_cmd
from config import Config
from robot.robot_interface import RobotInterface


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="g1_23dof_sonic.yaml")
    ap.add_argument("--zero", action="store_true", help="kd=0 instead of kd=8")
    ap.add_argument("--viser", action="store_true", help="live robot model in the browser")
    args = ap.parse_args()

    config = Config(args.config)
    ChannelFactoryInitialize(0)      # same as deploy_eskin
    robot = RobotInterface(config)

    viser_display = None
    if args.viser:
        from common.viser_display import DeployViser

        viser_display = DeployViser(Path(config.urdf_path), with_depth=False,
                                    depth_min=0, depth_max=1, depth_scale=1)

    fill = create_zero_cmd if args.zero else create_damping_cmd
    print(f"{'zero torque' if args.zero else 'damping'} -- Ctrl-C to stop")
    counter = 0
    try:
        while True:
            fill(robot.low_cmd)
            robot.send_cmd()
            if viser_display is not None and counter % 5 == 0:   # 10 Hz
                robot.update_joint_state()
                viser_display.update(robot.low_state, robot.qj)
                print(f"\rwaist yaw {np.degrees(robot.qj[12]):+7.2f} deg   ",
                      end="", flush=True)
            counter += 1
            time.sleep(config.control_dt)
    except KeyboardInterrupt:
        print("\nstopped -- still limp, robot holds nothing")


if __name__ == "__main__":
    main()
