from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from pathlib import Path
import argparse
import time
import threading
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from core.common.depth_client import SimpleDepthClient, pack_depth_images
from core.common.command_helper import create_damping_cmd
from core.common.remote_controller import KeyMap
from core.common.timer import TicToc
from core.common.real_logger import RealLogger
from core.config import Config

from core.robot.robot_interface import RobotInterface
from vigor.policy_runner import PolicyRunner

class ViserUpdateThread(threading.Thread):
    def __init__(
        self,
        viser_display,
        robot: RobotInterface,
        depth_client: Optional[SimpleDepthClient],
        hz: float = 30.0,
        ext_cam=None,
    ):
        super().__init__(daemon=True)
        self.viser_display = viser_display
        self.robot = robot
        self.depth_client = depth_client
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



class HighLevelController:
    def __init__(
        self,
        config: Config,
        robot: RobotInterface,
        policy_runner: PolicyRunner,
        depth_client: Optional[SimpleDepthClient] = None,
        logger: Optional[RealLogger] = None,
        ext_cam=None,
    ):
        self.config = config
        self.robot = robot
        self.policy_runner = policy_runner
        self.depth_client = depth_client
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
        depth_full = None
        color_full = None
        if self.depth_client is not None:
            depth_raw = self.depth_client.latest_depth
            depth_full = self.depth_client.latest_depth_full
            color_full = self.depth_client.latest_color

        action = self.policy_runner.compute_action(
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            depth_raw=depth_raw,
            base_quat_wxyz=imu_quat,
        )

        
        target_dof_pos = (
            action * self.config.action_scale + self.config.physical_default_angles
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
            depth_full=depth_full,
            color_full=color_full,
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
        depth_full,
        color_full,
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

        # Full-resolution streams for data collection, uncropped. Depth stays in
        # millimetres as uint16 so it is lossless; the u8 fields above are the
        # policy's cropped/normalised view and are not a substitute.
        if depth_full is not None:
            sample["depth_full_mm_u16"] = (
                np.clip(depth_full * 1000.0, 0, 65535).astype(np.uint16)
            )
        if color_full is not None:
            sample["color_full_bgr_u8"] = color_full.astype(np.uint8)



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

    viser_display = None
    if config.viser_display:
        from core.common.viser_display import DeployViser

        viser_display = DeployViser(
            urdf_path=Path(config.urdf_path),
            with_depth=config.use_depth_obs,
            depth_min=config.min_d,
            depth_max=config.max_d,
            depth_scale=config.depth_scale,
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

    from core.common.external_camera import ExtCamRGB

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
        logger=logger,
        ext_cam=ext_cam,
    )

    viser_thread = None
    if viser_display is not None:
        viser_thread = ViserUpdateThread(
            viser_display=viser_display,
            robot=robot,
            depth_client=depth_client,
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

    if viser_thread is not None:
        viser_thread.running = False
        viser_thread.join(timeout=1.0)

    logger.stop()
    logger.save(
        {
            "exit_reason": exit_reason,
            "steps": int(hl.counter),
            "T": float(hl.elapsed_time),
            "depth_enabled": bool(config.use_depth_obs),
            "depth_shape": [int(config.depth_img_height), int(config.depth_img_width)],
            "depth_full_shape": [
                int(config.depth_raw_height), int(config.depth_raw_width)
            ],
            "depth_full_units": "mm_uint16",
            "color_full_shape": [
                int(config.depth_raw_height), int(config.depth_raw_width), 3
            ],
            "color_full_format": "bgr8",
            "depth_min_m": float(config.min_d),
            "depth_max_m": float(config.max_d),
        }
    )
    if ext_cam is not None:
        ext_cam.stop()

    print(f"Exit reason: {exit_reason}")


if __name__ == "__main__":
    main()
