"""Fall-recovery deployment: diffusion planner -> z_g -> RL tracker.

Separate from `deploy_g1.py` (VIGOR) and from the SONIC runners; shares only the robot
interface, depth client and logging.

  python deploy_vigor2.py --config g1_23dof_sonic.yaml --cfg fallrec.yaml
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import argparse
import gc
import math
import time

import numpy as np
import yaml
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from core.common.depth_client import SimpleDepthClient
from core.common.remote_controller import KeyMap
from core.common.real_logger import RealLogger
from core.config import Config
from core.robot.robot_interface import RobotInterface
from vigor2.planner import _wrap_pi
from vigor2.robot import WatchdogRobot
from vigor2.runner import FallRecRunner


def load_fallrec_config(path: str) -> SimpleNamespace:
    full = Path(path)
    if not full.is_file():
        full = Path(__file__).resolve().parent / "vigor2" / path
    with open(full) as handle:
        return SimpleNamespace(**yaml.safe_load(handle))


def init_depth_client(config: Config) -> SimpleDepthClient:
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


class _NoPublish:
    """Stands in for the LowCmd publisher: counts what would have been sent."""
    def __init__(self) -> None:
        self.count = 0

    def Write(self, msg) -> None:
        self.count += 1


class FallRecController:
    def __init__(self, config: Config, robot: RobotInterface, runner: FallRecRunner,
                 depth_client: SimpleDepthClient, fallrec_cfg: SimpleNamespace,
                 logger: Optional[RealLogger] = None):
        self.config = config
        self.robot = robot
        self.runner = runner
        self.depth_client = depth_client
        self.logger = logger
        self.counter = 0
        self.elapsed_time = 0.0
        self._dpad = (0, 0)
        self.yaw_step_rad = math.radians(float(fallrec_cfg.yaw_step_deg))
        self._t_next = time.perf_counter()
        self._last_seq = -1
        self._last_depth_t = time.perf_counter()
        self.depth_max_age_s = float(fallrec_cfg.depth_max_age_s)
        self.handover_ticks = int(fallrec_cfg.handover_ticks)
        self.stand_blend_ticks = int(fallrec_cfg.stand_blend_ticks)
        self._ramp_origin = None
        self._ramp_k = 0
        self._ramp_n = 0
        self._was_held = False

    def _start_ramp(self, n: int) -> None:
        """Raised-cosine handover from the measured pose (SONIC): the first command is where the joints are."""
        self._ramp_origin = self.robot.qj.copy()
        self._ramp_k, self._ramp_n = 0, max(1, int(n))

    def _ramp(self, target: np.ndarray) -> np.ndarray:
        if self._ramp_origin is None or self._ramp_k >= self._ramp_n:
            return target
        self._ramp_k += 1
        a = 0.5 - 0.5 * np.cos(np.pi * self._ramp_k / self._ramp_n)
        return (1.0 - a) * self._ramp_origin + a * target

    def step(self) -> None:
        t0 = time.perf_counter()
        self.counter += 1
        self.elapsed_time += self.config.control_dt

        state, age = self.robot.low_state_snapshot_with_age()
        if age > self.config.max_low_state_age_s:
            raise RuntimeError(f"LowState stale for {age:.3f} s at tick {self.counter}.")
        self.robot.update_joint_state(state)
        imu_quat, _ = self.robot.get_imu()
        rc = self.robot.remote_controller
        dpad = (int(rc.button[KeyMap.left]), int(rc.button[KeyMap.right]))
        if dpad != self._dpad and any(dpad):  # D-pad left/right: turn the commanded heading, relative to the heading at start
            self.runner.yaw_command_rad = _wrap_pi(self.runner.yaw_command_rad + self.yaw_step_rad * (dpad[1] - dpad[0]))
            print(f"[cmd] yaw command {math.degrees(self.runner.yaw_command_rad):+.0f} deg")
        self._dpad = dpad
        base_ang_vel, projected_gravity, dof_pos, dof_vel = (
            self.robot.get_base_features_scaled()
        )

        seq = self.depth_client.latest_seq
        now = time.perf_counter()
        if seq != self._last_seq:
            self._last_seq, self._last_depth_t = seq, now
        elif now - self._last_depth_t > self.depth_max_age_s:
            raise RuntimeError(
                f"Stale depth at tick {self.counter}: no new frame for {now - self._last_depth_t:.3f} s."
            )

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
        if self.counter == 1:  # first command of any kind (history warm-up sends the default pose)
            self._start_ramp(self.handover_ticks)
        if self.runner.stand_held and not self._was_held:  # takeover: ramp onto the default pose
            self._start_ramp(self.stand_blend_ticks)
        self._was_held = self.runner.stand_held
        target_dof_pos = self._ramp(target_dof_pos)
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
                "yaw_cmd_deg": np.float32(math.degrees(self.runner.yaw_command_rad)),
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
    parser.add_argument("--cfg", type=str, default="fallrec.yaml", help="vigor2/ config: fallrec.yaml (new stack) or fallrec_old.yaml (old stack)")
    parser.add_argument("--log", type=str, default="FALLREC.npz")
    parser.add_argument("--no-publish", action="store_true", help="run everything but publish no LowCmd (motors never receive a command)")
    parser.add_argument("--keep-planning", action="store_true", help="no stand takeover: the planner keeps replanning (viewing only; requires --no-publish)")
    args = parser.parse_args()

    config = Config(args.config)
    fallrec_cfg = load_fallrec_config(args.cfg)
    depth_client = init_depth_client(config)

    ChannelFactoryInitialize(0)
    robot = WatchdogRobot(config)
    if args.no_publish:
        robot.lowcmd_publisher_ = _NoPublish()
        print("[NO-PUBLISH] LowCmd publisher replaced: the robot receives no command in this run.")
    runner = FallRecRunner(fallrec_cfg)
    runner.verify_robot_config(config)
    if args.keep_planning:
        if not args.no_publish:
            raise ValueError("--keep-planning is for watching the planner; it needs --no-publish")
        runner.stand_takeover = False
        print("[KEEP-PLANNING] stand takeover off: the planner replans for the whole run.")

    logger = RealLogger(args.log)
    logger.start()

    controller = FallRecController(config, robot, runner, depth_client, fallrec_cfg, logger)
    viser_thread = None
    if config.viser_display:
        from vigor2.viser_view import Vigor2Viser, ViserThread
        view = Vigor2Viser(str(config.urdf_path), config.min_d, config.max_d, config.depth_scale, runner.planner.bundle, runner)
        viser_thread = ViserThread(view, robot, depth_client, runner)
        viser_thread.start()
    while depth_client.latest_seq == 0:
        time.sleep(0.01)
    robot.update_joint_state()
    imu_quat, _ = robot.get_imu()
    secs = runner.warmup(*robot.get_base_features_scaled(), depth_client.latest_depth_full, imu_quat)
    gc.collect(); gc.freeze(); gc.disable()  # a gen-2 collection stalls the DDS thread 70-120 ms (measured); nothing in the loop needs the collector
    print(f"Warm-up {secs:.2f} s. Fall recovery ready. Press A to run, SELECT to damp and exit.")
    try:
        rc = robot.remote_controller
        while rc.button[KeyMap.A] != 1 and rc.button[KeyMap.select] != 1:  # nothing is published until A
            time.sleep(0.02)
        while rc.button[KeyMap.select] != 1:
            controller.step()
    finally:
        if viser_thread is not None:
            viser_thread.running = False
        runner.close()
        robot.send_damping_now()
        logger.stop()
        logger.save()
        robot.stop_rewriter()
        print("Damped and stopped." + (f" [NO-PUBLISH] {robot.lowcmd_publisher_.count} commands intercepted, none sent." if args.no_publish else ""))


if __name__ == "__main__":
    main()
