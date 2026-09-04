"""Fall-recovery deployment: diffusion planner -> z_g -> RL tracker.

Separate from `deploy_g1.py` (VIGOR) and from the SONIC runners; shares only the robot
interface, depth client and logging.

  python deploy_vigor2.py --config g1_29dof_sonic.yaml --cfg fallrec_lidar_29dof.yaml
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

from core.common.depth_process import DepthProcess
from core.common.livox_process import LivoxProcess
from core.common.remote_controller import KeyMap
from core.common.real_logger import RealLogger
from core.config import Config
from core.robot.robot_interface import RobotInterface
from vigor2.planner import _wrap_pi
from vigor2.robot import WatchdogRobot
from vigor2.runner import HV2_ROOT, FallRecRunner


def load_fallrec_config(path: str) -> SimpleNamespace:
    full = Path(path)
    if not full.is_file():
        full = Path(__file__).resolve().parent / "vigor2" / path
    with open(full) as handle:
        return SimpleNamespace(**yaml.safe_load(handle))


def init_depth_client(config: Config) -> DepthProcess:
    client = DepthProcess(
        server_address=config.depth_server_ip,
        port=config.depth_server_port,
        height=config.depth_raw_height,
        width=config.depth_raw_width,
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
                 depth_client: Optional[DepthProcess], lidar_client: Optional[LivoxProcess],
                 fallrec_cfg: SimpleNamespace,
                 logger: Optional[RealLogger] = None):
        self.config = config
        self.robot = robot
        self.runner = runner
        self.depth_client = depth_client
        self.lidar_client = lidar_client
        self.logger = logger
        self.counter = 0
        self.elapsed_time = 0.0
        self._dpad = (0, 0)
        self.yaw_step_rad = math.radians(float(fallrec_cfg.yaw_step_deg))
        self.viser = None
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

        state, age, torso_sample, torso_age = self.robot.sensor_snapshot_with_age()
        if age > self.config.max_low_state_age_s:
            raise RuntimeError(f"LowState stale for {age:.3f} s at tick {self.counter}.")
        self.robot.update_joint_state(state)
        imu_quat, imu_gyro = self.robot.get_imu(state)
        rc = self.robot.remote_controller
        dpad = (int(rc.button[KeyMap.left]), int(rc.button[KeyMap.right]))
        if dpad != self._dpad and any(dpad):  # D-pad left/right: turn the commanded heading, relative to the heading at start
            self.runner.yaw_command_rad = _wrap_pi(self.runner.yaw_command_rad + self.yaw_step_rad * (dpad[1] - dpad[0]))
            print(f"[cmd] yaw command {math.degrees(self.runner.yaw_command_rad):+.0f} deg")
        self._dpad = dpad
        if self.viser is not None:
            slider = self.viser.slider_command()
            if slider is not None:
                self.runner.yaw_command_rad = slider
        base_ang_vel, projected_gravity, dof_pos, dof_vel = (
            self.robot.get_base_features_scaled((imu_quat, imu_gyro))
        )

        now = time.perf_counter()
        sensor_kwargs = {}
        depth_raw = depth_full = None
        if self.runner.use_depth:
            seq = self.depth_client.latest_seq
            if seq != self._last_seq:
                self._last_seq, self._last_depth_t = seq, now
            elif now - self._last_depth_t > self.depth_max_age_s:
                raise RuntimeError(
                    f"Stale depth at tick {self.counter}: no new frame for "
                    f"{now - self._last_depth_t:.3f} s."
                )
            depth_raw = self.depth_client.latest_depth
            depth_full = self.depth_client.latest_depth_full
        else:
            if torso_age > self.config.max_torso_imu_age_s:
                raise RuntimeError(f"Torso IMU stale for {torso_age:.3f} s")
            lidar_scan, lidar_seq, lidar_age, scan_receipt, _ = (
                self.lidar_client.snapshot()
            )
            if lidar_age > self.config.livox_max_age_s:
                raise RuntimeError(f"Livox scan stale for {lidar_age:.3f} s")
            if lidar_seq >= 10 and self.lidar_client.rate_hz < self.config.livox_min_rate_hz:
                raise RuntimeError(
                    f"Livox rolling-window rate {self.lidar_client.rate_hz:.1f} Hz < "
                    f"required {self.config.livox_min_rate_hz:.1f} Hz"
                )
            torso_for_scan, torso_skew = self.robot.torso_imu_nearest(scan_receipt)
            state_for_scan, state_skew = self.robot.low_state_nearest(scan_receipt)
            max_skew = self.config.max_sensor_sync_skew_s
            if torso_skew > max_skew or state_skew > max_skew:
                raise RuntimeError(
                    "LiDAR/state synchronization skew is too large: "
                    f"torso={torso_skew * 1e3:.1f} ms, "
                    f"LowState={state_skew * 1e3:.1f} ms"
                )
            torso_quat, torso_gyro, _ = self.robot.get_torso_imu(torso_for_scan)
            lidar_q, _ = self.robot.active_joint_state(state_for_scan)
            sensor_kwargs = dict(
                lidar_scan=lidar_scan,
                lidar_seq=lidar_seq,
                lidar_age_s=lidar_age,
                torso_quat_wxyz=torso_quat,
                torso_gyro=torso_gyro,
                lidar_physical_dof_pos=lidar_q,
            )

        action = self.runner.compute_action(
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            depth_raw=depth_raw,
            depth_full=depth_full,
            base_quat_wxyz=imu_quat,
            physical_dof_pos=self.robot.qj,
            physical_dof_vel=self.robot.dqj,
            physical_base_ang_vel=imu_gyro,
            **sensor_kwargs,
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
        if self.viser is not None:
            self.viser.publish(state, self.robot.qj, self.runner.plan_age, self.runner.stand_held, self.runner.planner.start_yaw, self.runner.yaw_command_rad)

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
                "vision_age_ms": np.float32(
                    ((now - self._last_depth_t) if self.runner.use_depth
                     else self.lidar_client.age_s) * 1000.0
                ),
                "torso_imu_age_ms": np.float32(torso_age * 1000.0),
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
    parser.add_argument("--no-publish", action="store_true", help="publish no LowCmd (motors never receive a command); the stand latch is off so the planner keeps running, for viewing")
    args = parser.parse_args()

    config = Config(args.config)
    fallrec_cfg = load_fallrec_config(args.cfg)
    ChannelFactoryInitialize(0)
    robot = WatchdogRobot(config)
    if args.no_publish:
        robot.lowcmd_publisher_ = _NoPublish()
        print("[NO-PUBLISH] LowCmd publisher replaced: the robot receives no command in this run.")
    runner = FallRecRunner(fallrec_cfg)
    runner.verify_robot_config(config)
    depth_client = None
    lidar_client = None
    if runner.use_depth:
        depth_client = init_depth_client(config)
    else:
        if not config.require_torso_imu:
            raise ValueError("LiDAR deployment requires require_torso_imu: true")
        lidar_client = LivoxProcess(
            config.livox_topic,
            config.livox_imu_topic,
            config.livox_max_points,
            config.livox_scan_duration_s,
        )
        lidar_client.start()
        print(
            f"Livox receiver started on {config.livox_topic} with IMU "
            f"{config.livox_imu_topic}."
        )
    if args.no_publish:
        runner.stand_takeover = False

    logger = RealLogger(args.log)
    logger.start()

    controller = FallRecController(
        config, robot, runner, depth_client, lidar_client, fallrec_cfg, logger
    )
    viser = None
    if config.viser_display and runner.use_depth:  # current viser panel is depth-specific
        from vigor2.viser_process import ViserProcess
        ap = runner.async_planner
        if ap is None:
            raise ValueError("viser_display needs planner_async: true (the view reads the planner process buffers)")
        viser = ViserProcess(
            str(HV2_ROOT), dict(urdf_path=str(config.urdf_path), depth_min=config.min_d, depth_max=config.max_d,
                                depth_scale=config.depth_scale, feature_layout=list(runner.planner.bundle.feature_layout),
                                dof_names=list(runner.planner.bundle.dof_names), fk_urdf_path=str(runner.fk_urdf_path)),
            depth_client.frames, depth_client.seq,
            config.num_actions, ap.resp_states, ap.ctl, ap.stat, runner.yaw_command_rad)
        controller.viser = viser
    sensor_client = depth_client if runner.use_depth else lidar_client
    sensor_deadline = time.monotonic() + (
        float(config.livox_startup_timeout_s) if runner.use_lidar else 10.0
    )
    while sensor_client.latest_seq == 0:
        if not sensor_client.running:
            raise RuntimeError("Vision receiver exited:\n" + sensor_client.error)
        if time.monotonic() >= sensor_deadline:
            detail = (
                " No synchronized /livox/imu samples were available for deskew."
                if runner.use_lidar else ""
            )
            raise RuntimeError(
                f"No {'Livox scan' if runner.use_lidar else 'depth frame'} after "
                f"{(config.livox_startup_timeout_s if runner.use_lidar else 10.0):.1f} s."
                + detail
            )
        time.sleep(0.01)
    robot.update_joint_state()
    state, _, torso_sample, _ = robot.sensor_snapshot_with_age()
    imu_quat, imu_gyro = robot.get_imu(state)
    warmup_kwargs = {}
    if runner.use_depth:
        vision = depth_client.latest_depth_full
        lidar_seq = None
        lidar_age = None
    else:
        vision, lidar_seq, lidar_age, _, _ = lidar_client.snapshot()
    if runner.use_lidar:
        torso_quat, torso_gyro, _ = robot.get_torso_imu(torso_sample)
        warmup_kwargs = dict(
            lidar_scan=vision,
            lidar_seq=lidar_seq,
            lidar_age_s=lidar_age,
            torso_quat_wxyz=torso_quat,
            torso_gyro=torso_gyro,
            lidar_physical_dof_pos=robot.qj,
            physical_dof_pos=robot.qj,
            physical_dof_vel=robot.dqj,
            physical_base_ang_vel=imu_gyro,
        )
    else:
        warmup_kwargs = {
            "physical_dof_pos": robot.qj,
            "physical_dof_vel": robot.dqj,
            "physical_base_ang_vel": imu_gyro,
        }
    secs = runner.warmup(
        *robot.get_base_features_scaled((imu_quat, imu_gyro)),
        vision,
        imu_quat,
        **warmup_kwargs,
    )
    gc.collect(); gc.freeze(); gc.disable()  # a gen-2 collection stalls the DDS thread 70-120 ms (measured); nothing in the loop needs the collector
    print(f"Warm-up {secs:.2f} s. Fall recovery ready. Press A to run, SELECT to damp and exit.")
    try:
        rc = robot.remote_controller
        while rc.button[KeyMap.A] != 1 and rc.button[KeyMap.select] != 1:  # nothing is published until A
            time.sleep(0.02)
        while rc.button[KeyMap.select] != 1:
            controller.step()
    finally:
        if viser is not None:
            if viser.error:
                print("[viser process] " + viser.error.strip().splitlines()[-1])
            viser.stop()
        runner.close()
        if depth_client is not None:
            depth_client.running = False
        if lidar_client is not None:
            lidar_client.stop()
        robot.send_damping_now()
        logger.stop()
        logger.save()
        robot.stop_rewriter()
        print("Damped and stopped." + (f" [NO-PUBLISH] {robot.lowcmd_publisher_.count} commands intercepted, none sent." if args.no_publish else ""))


if __name__ == "__main__":
    main()
