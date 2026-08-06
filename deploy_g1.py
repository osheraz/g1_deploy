from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from pathlib import Path
import argparse
import time
import threading
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from common.depth_client import SimpleDepthClient, pack_depth_images
from common.command_helper import create_damping_cmd
from common.remote_controller import KeyMap
from common.timer import TicToc
from common.real_logger import RealLogger
from common.tactile_render import render_sensor_panels
from config import Config

if TYPE_CHECKING:
    from common.tactile_client import TactileClient

from robot.robot_interface import RobotInterface
from policy.policy_runner import PolicyRunner

TACTILE_RAW_LEN = 256
TACTILE_GYRO_LEN = 16


class ViserUpdateThread(threading.Thread):
    def __init__(
        self,
        viser_display,
        robot: RobotInterface,
        depth_client: Optional[SimpleDepthClient],
        tactile_client: Optional[TactileClient],
        hz: float = 30.0,
        ext_cam=None,
    ):
        super().__init__(daemon=True)
        self.viser_display = viser_display
        self.robot = robot
        self.depth_client = depth_client
        self.tactile_client = tactile_client
        self.ext_cam = ext_cam

        self.hz = hz
        self.running = True

    def run(self):
        dt = 1.0 / self.hz
        while self.running:
            depth_raw = None
            if self.depth_client is not None:
                depth_raw = self.depth_client.latest_depth

            rgb_bgr = None
            if self.ext_cam is not None:
                rgb_bgr = self.ext_cam.latest_bgr

            self.viser_display.update(
                low_state=self.robot.low_state,
                qj=self.robot.qj,
                depth_raw=depth_raw,
                rgb_bgr=rgb_bgr,
            )
            if self.tactile_client is not None:
                snapshot = self.tactile_client.snapshot()
                sensor_status = {}
                panels = {}
                overall_on = False
                for sensor_name, sample in snapshot.items():
                    raw = sample.get("raw_sensor_bytes", b"")
                    gyro = sample.get("raw_gyro_bytes", b"")
                    on = bool(raw or gyro)
                    overall_on = overall_on or on
                    status = "ON" if on else "OFF"
                    sensor_status[sensor_name] = (
                        f"{status}(raw={len(raw)},gyro={len(gyro)})"
                    )
                    if raw:
                        panels.update(render_sensor_panels(sensor_name, raw))
                self.viser_display.update_tactile(
                    "ON" if overall_on else "OFF",
                    sensor_status,
                )
                self.viser_display.update_tactile_images(panels)
            time.sleep(dt)


def init_depth_client(config: Config) -> Optional[SimpleDepthClient]:
    if not config.use_depth_obs:
        return None
    depth_client = SimpleDepthClient(
        server_address=config.depth_server_ip,
        port=config.depth_server_port,
        height=config.depth_raw_height,
        width=config.depth_raw_width,
        show_debug=False,
    )
    depth_client.start()
    print("Depth client started.")
    return depth_client


def init_loco_client():
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    loco_client = LocoClient()
    loco_client.SetTimeout(10.0)
    try:
        loco_client.Init()
        print("LocoClient connected successfully.")
    except Exception as e:
        print(f"LocoClient connection failed: {e}")
        return None
    return loco_client


def init_tactile_client(config: Config) -> Optional["TactileClient"]:
    if not config.use_tactile:
        return None

    from common.tactile_client import TactileClient  # lazy: pulls in rclpy/ROS

    tactile_client = TactileClient(sensor_names=config.tactile_sensor_order)
    tactile_client.start()
    print(f"Tactile ROS client started for sensors={config.tactile_sensor_order}.")
    return tactile_client


class HighLevelController:
    def __init__(
        self,
        config: Config,
        robot: RobotInterface,
        policy_runner: PolicyRunner,
        depth_client: Optional[SimpleDepthClient] = None,
        tactile_client: Optional[TactileClient] = None,
        logger: Optional[RealLogger] = None,
        ext_cam=None,
    ):
        self.config = config
        self.robot = robot
        self.policy_runner = policy_runner
        self.depth_client = depth_client
        self.tactile_client = tactile_client
        self.ext_cam = ext_cam
        self.logger = logger

        self.counter = 0
        self.elapsed_time = 0.0
        self._t_next = time.perf_counter()
        self._fps_last_time = self._t_next
        self.tt = TicToc()
        self._ext_cam_last_t = 0.0

    def step(self):
        t0 = time.perf_counter()

        self.counter += 1
        self.elapsed_time += self.config.control_dt

        self.robot.update_joint_state()
        imu_quat, _ = self.robot.get_imu()

        base_ang_vel, projected_gravity, dof_pos, dof_vel = (
            self.robot.get_base_features_scaled()
        )

        depth_raw = None
        if self.depth_client is not None:
            depth_raw = self.depth_client.latest_depth

        tactile_snapshot = {}
        if self.tactile_client is not None:
            tactile_snapshot = self.tactile_client.snapshot()

        action = self.policy_runner.compute_action(
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            depth_raw=depth_raw,
            base_quat_wxyz=imu_quat,
        )

        
        target_dof_pos = (
            action * self.config.action_scale + self.config.default_angles_23
        )

        if self.policy_runner.act_count < 50:
            e = 0.5
            target_dof_pos = e * target_dof_pos + (1-e)*self.robot.qj
        
        self.robot.apply_target_dof_pos(target_dof_pos) 

        # Log
        loop_dt_ms = (time.perf_counter() - t0) * 1000.0
        self.log_step(
            loop_dt_ms=loop_dt_ms,
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            action=action,
            target_dof_pos=target_dof_pos,
            depth_raw=depth_raw,
            tactile_snapshot=tactile_snapshot,
            imu_quat=imu_quat,
        )

        if self.counter % 25 == 0:
            print(
                f"[State] low_state: mode_machine={self.robot.low_state.mode_machine} | "
                f"tick={self.robot.low_state.tick}"
            )

        now = time.perf_counter()
        dt = now - self._fps_last_time
        self._fps_last_time = now
        fps = 1.0 / dt if dt > 0 else 0.0
        print(f"[Loop FPS] {fps:.1f} Hz | dt={dt*1000:.2f} ms")

        self._t_next += self.config.control_dt
        sleep_s = self._t_next - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            self._t_next = time.perf_counter()

    def log_step(
        self,
        loop_dt_ms: float,
        base_ang_vel: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        action: np.ndarray,
        target_dof_pos: np.ndarray,
        depth_raw,
        tactile_snapshot: dict,
        imu_quat,
    ):
        if self.logger is None:
            return

        sample = {
            "t": np.float32(self.elapsed_time),
            "loop_dt_ms": np.float32(loop_dt_ms),
            "qj": self.robot.qj.copy().astype(np.float32),
            "dqj": self.robot.dqj.copy().astype(np.float32),
            "base_ang_vel": base_ang_vel.copy().astype(np.float32),
            "projected_gravity": projected_gravity.copy().astype(np.float32),
            "dof_pos_scaled": dof_pos.copy().astype(np.float32),
            "dof_vel_scaled": dof_vel.copy().astype(np.float32),
            "action": action.copy().astype(np.float32),
            "target_dof_pos": target_dof_pos.copy().astype(np.float32),
            "imu_quat_wxyz": np.asarray(imu_quat, dtype=np.float32),
        }

        if depth_raw is not None:
            vis_u8, ov_u8 = pack_depth_images(
                depth_raw=depth_raw,
                out_h=self.config.depth_img_height,
                out_w=self.config.depth_img_width,
                min_d=self.config.min_d,
                max_d=self.config.max_d,
            )
            sample["depth_vis_u8"] = vis_u8
            sample["depth_obs_u8"] = ov_u8

        for sensor_name, tactile in tactile_snapshot.items():
            tag = sensor_name.lower()
            raw_u8 = np.zeros(TACTILE_RAW_LEN, dtype=np.uint8)
            gyro_u8 = np.zeros(TACTILE_GYRO_LEN, dtype=np.uint8)

            raw_src = np.frombuffer(tactile["raw_sensor_bytes"], dtype=np.uint8)
            gyro_src = np.frombuffer(tactile["raw_gyro_bytes"], dtype=np.uint8)
            raw_u8[: min(TACTILE_RAW_LEN, raw_src.size)] = raw_src[:TACTILE_RAW_LEN]
            gyro_u8[: min(TACTILE_GYRO_LEN, gyro_src.size)] = gyro_src[:TACTILE_GYRO_LEN]

            sample[f"tactile_{tag}_raw_u8"] = raw_u8
            sample[f"tactile_{tag}_gyro_u8"] = gyro_u8

        if self.ext_cam is not None:
            frame = self.ext_cam.latest_bgr
            if frame is not None:
                sample["ext_cam_bgr_u8"] = frame

        self.logger.push(sample)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default="enp132s0")
    parser.add_argument(
        "--config",
        type=str,
        default="g1_23dof.yaml",
    )
    parser.add_argument("--T", type=float, default=100000000000000000000.0)
    parser.add_argument("--log", type=str, default=f"TEST.npz")

    args = parser.parse_args()

    config = Config(args.config)

    depth_client = init_depth_client(config)
    tactile_client = init_tactile_client(config)

    viser_display = None
    if config.viser_display:
        from common.viser_display import DeployViser

        viser_display = DeployViser(
            urdf_path=Path(config.urdf_path),
            with_depth=config.use_depth_obs,
            depth_min=config.min_d,
            depth_max=config.max_d,
            depth_scale=config.depth_scale,
            tactile_sensors=config.tactile_sensor_order if config.use_tactile else [],
        )

    ChannelFactoryInitialize(0)
    loco_client = init_loco_client()
    if loco_client is None:
        return

    robot = RobotInterface(config)
    policy_runner = PolicyRunner(config)

    logger = RealLogger(args.log)
    logger.start()
    
    # logger = None

    from common.external_camera import ExtCamRGB

    ext_cam = ExtCamRGB(
        cam_id=0,
        width=800,
        height=600,
        fps=30,
        warmup=10,
    )
    ext_cam.start()

    # ext_cam = None

    hl = HighLevelController(
        config,
        robot,
        policy_runner,
        depth_client=depth_client,
        tactile_client=tactile_client,
        logger=logger,
        ext_cam=ext_cam,
    )

    viser_thread = None
    if viser_display is not None:
        viser_thread = ViserUpdateThread(
            viser_display=viser_display,
            robot=robot,
            depth_client=depth_client,
            tactile_client=tactile_client,
            ext_cam=ext_cam,
            hz=50.0,
        )
        viser_thread.start()

    # Zero torque state...
    # while True:
    #     robot.zero_torque_state()

    robot.move_to_default_pos() 
    robot.default_pos_state()

    exit_reason = None
    max_steps = int(args.T / config.control_dt) if args.T is not None else None

    warmup_steps = 1000
    for _ in range(warmup_steps):
        robot.update_joint_state()
        base_ang_vel, projected_gravity, dof_pos, dof_vel = (
            robot.get_base_features_scaled()
        )
        policy_runner._update_history_long(
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity,
            dof_pos=dof_pos,

            dof_vel=dof_vel,
        )

    try:
        while True:
            hl.step()

            if max_steps is not None and hl.counter >= max_steps:
                exit_reason = f"Reached control time limit {args.T}s"
                break

            if robot.remote_controller.button[KeyMap.select] == 1:
                exit_reason = "select"
                break

    except KeyboardInterrupt:
        exit_reason = "keyboard"

    create_damping_cmd(robot.low_cmd)
    robot.send_cmd()

    if depth_client is not None:
        depth_client.running = False

    if tactile_client is not None:
        tactile_client.stop()

    if viser_thread is not None:
        viser_thread.running = False
        viser_thread.join(timeout=1.0)

    logger.stop()
    logger.save(
        {
            "exit_reason": exit_reason,
            "steps": int(hl.counter),
            "T": float(hl.elapsed_time),
            # Schema per modality, so readers don't hardcode it.
            "tactile_sensors": (
                list(config.tactile_sensor_order) if config.use_tactile else []
            ),
            "tactile_raw_len": TACTILE_RAW_LEN,
            "tactile_gyro_len": TACTILE_GYRO_LEN,
            "depth_enabled": bool(config.use_depth_obs),
            "depth_shape": [int(config.depth_img_height), int(config.depth_img_width)],
            "depth_min_m": float(config.min_d),
            "depth_max_m": float(config.max_d),
        }
    )
    if ext_cam is not None:
        ext_cam.stop()

    print(f"Exit reason: {exit_reason}")


if __name__ == "__main__":
    main()
