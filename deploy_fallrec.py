"""Fall-recovery deployment: diffusion planner -> z_g -> RL tracker.

Separate from `deploy_g1.py` (VIGOR) and from the SONIC runners; shares only the robot
interface, depth client and logging.

  python deploy_fallrec.py --config g1_23dof_sonic.yaml --fallrec fallrec.yaml
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import argparse
import time

import numpy as np
import yaml
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from core.common.depth_client import SimpleDepthClient
from core.common.command_helper import create_damping_cmd
from core.common.remote_controller import KeyMap
from core.common.real_logger import RealLogger
from core.config import Config
from core.robot.robot_interface import RobotInterface
from vigor2.runner import FallRecRunner


def load_fallrec_config(path: str) -> SimpleNamespace:
    full = Path(path)
    if not full.is_file():
        full = Path(__file__).resolve().parent / "vigor2" / path
    with open(full) as handle:
        return SimpleNamespace(**yaml.safe_load(handle))


def init_depth_client(config: Config) -> SimpleDepthClient:
    if not config.use_depth_obs:
        raise ValueError("Fall recovery needs use_depth_obs=true in the robot config.")
    client = SimpleDepthClient(
        server_address=config.depth_server_ip,
        port=config.depth_server_port,
        height=config.depth_raw_height,
        width=config.depth_raw_width,
        show_debug=False,
    )
    client.start()
    print("Depth client started.")
    return client


class FallRecController:
    def __init__(self, config: Config, robot: RobotInterface, runner: FallRecRunner,
                 depth_client: SimpleDepthClient, logger: Optional[RealLogger] = None):
        self.config = config
        self.robot = robot
        self.runner = runner
        self.depth_client = depth_client
        self.logger = logger
        self.counter = 0
        self.elapsed_time = 0.0
        self._t_next = time.perf_counter()
        self._last_seq = -1

    def step(self) -> None:
        t0 = time.perf_counter()
        self.counter += 1
        self.elapsed_time += self.config.control_dt

        self.robot.update_joint_state()
        imu_quat, _ = self.robot.get_imu()
        base_ang_vel, projected_gravity, dof_pos, dof_vel = (
            self.robot.get_base_features_scaled()
        )

        seq = self.depth_client.latest_seq
        if seq == self._last_seq:
            raise RuntimeError(
                f"Stale depth at tick {self.counter}: seq {seq} unchanged since the last step."
            )
        self._last_seq = seq

        action = self.runner.compute_action(
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            depth_raw=self.depth_client.latest_depth,
            depth_full=self.depth_client.latest_depth_full,
            base_quat_wxyz=imu_quat,
        )

        target_dof_pos = (
            action * self.config.action_scale + self.config.physical_default_angles
        )
        if self.runner.act_count < 50:
            target_dof_pos = 0.5 * target_dof_pos + 0.5 * self.robot.qj
        self.robot.apply_target_dof_pos(target_dof_pos)

        loop_dt_ms = (time.perf_counter() - t0) * 1000.0
        if self.counter % 25 == 0:
            plan_ms = (
                self.runner.async_planner.last_latency_ms
                if self.runner.async_planner is not None else 0.0
            )
            print(f"[fallrec] tick={self.counter} loop={loop_dt_ms:.1f} ms "
                  f"plan={plan_ms:.1f} ms plan_age={self.runner.plan_age}")
        if self.logger is not None:
            self.logger.push({
                "t": np.float32(self.elapsed_time),
                "loop_dt_ms": np.float32(loop_dt_ms),
                "qj": self.robot.qj.copy().astype(np.float32),
                "action": action.copy().astype(np.float32),
                "target_dof_pos": target_dof_pos.copy().astype(np.float32),
                "imu_quat_wxyz": np.asarray(imu_quat, dtype=np.float32),
                "plan_age": np.float32(self.runner.plan_age),
            })

        self._t_next += self.config.control_dt
        sleep_s = self._t_next - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            self._t_next = time.perf_counter()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="g1_23dof_sonic.yaml")
    parser.add_argument("--fallrec", type=str, default="fallrec.yaml")
    parser.add_argument("--log", type=str, default="FALLREC.npz")
    args = parser.parse_args()

    config = Config(args.config)
    fallrec_cfg = load_fallrec_config(args.fallrec)
    depth_client = init_depth_client(config)

    ChannelFactoryInitialize(0)
    robot = RobotInterface(config)
    runner = FallRecRunner(fallrec_cfg)
    runner.verify_robot_config(config)

    logger = RealLogger(args.log)
    logger.start()

    controller = FallRecController(config, robot, runner, depth_client, logger)
    print("Fall recovery ready. START to run, SELECT to damp and exit.")
    try:
        while True:
            if robot.remote_controller.button[KeyMap.select] == 1:
                break
            controller.step()
    finally:
        runner.close()
        create_damping_cmd(robot.low_cmd)
        robot.send_cmd()
        logger.stop()
        logger.save()
        print("Damped and stopped.")


if __name__ == "__main__":
    main()
