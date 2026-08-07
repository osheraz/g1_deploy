#!/usr/bin/env python3
"""e-skin deploy: PICO teleop + tactile, with gamepad-triggered recording.

Same runtime as deploy_g1.py -- same RobotInterface, PolicyRunner and configs --
but this is the data-collection variant: it adds the tactile stream and records
episodes on a button press instead of one log per run.

    python deploy_eskin.py --mode record --out ~/episodes

**Moves the robot on launch.**

Gamepad, while running:

    A       start an episode
    B       stop it and write to disk
    select  quit

Each episode is a directory:

    episodes/episode_001/
        state.npz     robot joints, IMU, tactile, PICO poses, actions
        frames.bin    colour+depth exactly as the camera server sent them
        meta.json     shapes, rates, schema

Camera frames are stored still-encoded (JPEG colour, PNG depth). The server
encodes before sending, so decoding and re-storing them would be ~5x larger for
identical pixels; depth PNG is lossless either way. Frames stream to disk as
they arrive, so RAM stays flat.

Read an episode back with ``load_episode()`` at the bottom of this file.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from common.command_helper import create_damping_cmd
from common.depth_client import SimpleDepthClient
from common.episode import Recorder, load_episode  # noqa: F401  (re-exported)
from common.remote_controller import KeyMap
from common.tactile_record import sample_fields as tactile_fields
from config import Config
from robot.robot_interface import RobotInterface


class EskinController:
    def __init__(self, config, robot, policy_runner, camera, tactile, recorder):
        self.config = config
        self.robot = robot
        self.policy_runner = policy_runner
        self.camera = camera
        self.tactile = tactile
        self.recorder = recorder

        # record mode has no policy: hold the default pose and only stream.
        self.is_sonic = hasattr(policy_runner, "reference")
        self.record_only = policy_runner is None
        self.counter = 0
        self.elapsed = 0.0
        self._last_camera_seq = -1
        self._t_next = time.perf_counter()

    def step(self) -> None:
        self.counter += 1
        self.elapsed += self.config.control_dt

        self.robot.update_joint_state()
        imu_quat, imu_gyro = self.robot.get_imu()
        base_ang_vel, projected_gravity, dof_pos, dof_vel = (
            self.robot.get_base_features_scaled()
        )

        if self.record_only:
            # Hold the default pose. Nothing is inferred and no new target is
            # computed, so this is the safe way to prove the recorder works.
            action = np.zeros(self.config.num_actions, np.float32)
            target_dof_pos = self.config.default_angles_23.copy()
            self.robot.apply_target_dof_pos(target_dof_pos)
        else:
            depth_raw = (self.camera.latest_depth
                         if (self.camera and self.config.use_depth_obs) else None)
            if self.is_sonic:
                ref = self.policy_runner.reference
                if hasattr(ref, "initialize_from_robot") and not ref._initialized:
                    ref.initialize_from_robot(imu_quat, self.robot.qj)
                # SONIC builds its own observation: RAW joint state in robot
                # order and world-frame angular velocity, NOT the scaled
                # actor features.
                action = self.policy_runner.compute_action(
                    dof_pos=self.robot.qj,
                    dof_vel=self.robot.dqj,
                    base_quat_wxyz=imu_quat,
                    base_ang_vel_world=imu_gyro,
                )
                self.policy_runner.reference.step()
            else:
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
                target_dof_pos = e * target_dof_pos + (1 - e) * self.robot.qj
            self.robot.apply_target_dof_pos(target_dof_pos)

        self.recorder.handle_buttons(self.robot.remote_controller.button)
        if self.recorder.recording:
            self._record(action, target_dof_pos, imu_quat,
                         base_ang_vel, projected_gravity, dof_pos, dof_vel)

        if self.counter % 50 == 0:
            print(f"\r[{self.elapsed:7.1f}s] {self.recorder.status()}          ",
                  end="", flush=True)

        self._t_next += self.config.control_dt
        sleep = self._t_next - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        else:
            self._t_next = time.perf_counter()

    def _record(self, action, target_dof_pos, imu_quat,
                base_ang_vel, projected_gravity, dof_pos, dof_vel) -> None:
        sample = {
            "t": np.float32(self.elapsed),
            "qj": self.robot.qj.copy().astype(np.float32),
            "dqj": self.robot.dqj.copy().astype(np.float32),
            "action": action.copy().astype(np.float32),
            "target_dof_pos": target_dof_pos.copy().astype(np.float32),
            "base_ang_vel": base_ang_vel.copy().astype(np.float32),
            "projected_gravity": projected_gravity.copy().astype(np.float32),
            "dof_pos_scaled": dof_pos.copy().astype(np.float32),
            "dof_vel_scaled": dof_vel.copy().astype(np.float32),
            "imu_quat_wxyz": np.asarray(imu_quat, np.float32),
        }

        # Teleop command being imitated: mode, base nav, hands, VR endpoints.
        ref = getattr(self.policy_runner, "reference", None)
        if ref is not None and hasattr(ref, "teleop_fields"):
            sample.update(ref.teleop_fields())

        if self.tactile is not None:
            sample.update(
                tactile_fields(self.tactile.snapshot(), self.config.tactile_sensor_order)
            )

        # The camera publishes at 30 Hz while this loop runs at 50, so only
        # write the image bytes when the sequence number actually advanced.
        seq = self.camera.latest_seq if self.camera else 0
        new_frame = seq != self._last_camera_seq
        self._last_camera_seq = seq
        sample["camera_seq"] = np.int64(seq)
        self.recorder.push(
            sample,
            self.camera.latest_color_jpg if self.camera else None,
            self.camera.latest_depth_png if self.camera else None,
            new_frame=new_frame,
        )


def build_policy_runner(config, mode: str):
    """SONIC encoder/decoder driven by `mode`, the actor runner, or nothing."""
    if mode == "record":
        return None            # hold position and only record
    if getattr(config, "policy_type", "actor") == "sonic":
        from policy.sonic_reference import build_reference
        from policy.sonic_runner import SonicRunner

        return SonicRunner(config, build_reference(config, mode))

    from policy.policy_runner import PolicyRunner

    return PolicyRunner(config)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="g1_23dof_sonic.yaml")
    ap.add_argument("--mode", choices=["record", "motion", "pico"], default="record",
                    help="record: hold position, stream+record only (no policy). "
                         "motion: replay an exported SONIC clip. pico: live VR.")
    ap.add_argument("--out", default="episodes", help="episode output directory")
    ap.add_argument("--net", default="enp132s0")
    ap.add_argument("--tactile-ip", default=None,
                    help="override config; use 127.0.0.1 when the gloves are "
                         "plugged into this workstation")
    ap.add_argument("--camera-ip", default=None, help="override config")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-tactile", action="store_true")
    args = ap.parse_args()

    config = Config(args.config)

    # The camera is recorded even when the policy does not consume depth
    # (the PICO/sonic policy does not), so it is not gated on use_depth_obs.
    camera_ip = args.camera_ip or config.depth_server_ip
    tactile_ip = args.tactile_ip or config.tactile_server_ip

    camera = None
    if not args.no_camera:
        camera = SimpleDepthClient(
            server_address=camera_ip,
            port=config.depth_server_port,
            height=config.depth_raw_height,
            width=config.depth_raw_width,
            show_debug=False,
        )
        camera.start()
        print(f"camera:  tcp://{camera_ip}:{config.depth_server_port}")

    tactile = None
    if config.use_tactile and not args.no_tactile:
        from common.tactile_client import TactileClient

        tactile = TactileClient(
            sensor_names=config.tactile_sensor_order,
            server_address=tactile_ip,
            port=config.tactile_server_port,
        )
        tactile.start()
        print(f"tactile: tcp://{tactile_ip}:{config.tactile_server_port} "
              f"{config.tactile_sensor_order}")

    ChannelFactoryInitialize(0)
    robot = RobotInterface(config)
    policy_runner = build_policy_runner(config, args.mode)

    # X, not A: A is consumed by default_pos_state() to release the loop,
    # so recording on A would start an episode the instant you start moving.
    recorder = Recorder(Path(args.out).expanduser(), {
        "policy_type": getattr(config, "policy_type", "actor"),
        "mode": args.mode,
        "config": str(config.config_path),
        "policy": str(config.policy_path),
        "control_dt": float(config.control_dt),
        "rate_hz": 1.0 / float(config.control_dt),
        "sensors": list(config.tactile_sensor_order) if tactile else [],
        "tactile_raw_len": 256,
        "tactile_imu_len": 16,
        "camera_shape": [int(config.depth_raw_height), int(config.depth_raw_width)],
    }, button=KeyMap.X)

    robot.move_to_default_pos()
    robot.default_pos_state()

    hl = EskinController(config, robot, policy_runner, camera, tactile, recorder)

    print("\n  X = toggle recording   select = quit\n")

    exit_reason = None
    try:
        while True:
            hl.step()
            if robot.remote_controller.button[KeyMap.select] == 1:
                exit_reason = "select"
                break
    except KeyboardInterrupt:
        exit_reason = "keyboard"

    recorder.stop()  # flush an episode still open
    create_damping_cmd(robot.low_cmd)
    robot.send_cmd()
    if camera is not None:
        camera.running = False
    if tactile is not None:
        tactile.stop()

    print(f"\nexit: {exit_reason}")



if __name__ == "__main__":
    main()
