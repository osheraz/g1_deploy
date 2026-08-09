#!/usr/bin/env python3
"""e-skin deploy: SONIC tracker + tactile, with gamepad-triggered recording.

    python deploy_eskin.py --mode record --out ~/episodes
    python deploy_eskin.py --mode motion --clip policy/sonic/motion_stand.npz

MOVES THE ROBOT ON LAUNCH.  A = start, X = toggle recording, select = quit.

Episodes are directories: state.npz + frames.bin + meta.json.
Read one back with common.episode.load_episode().
"""
from __future__ import annotations

import argparse
import copy
import threading
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


class EskinRobot(RobotInterface):
    """RobotInterface + the official deploy's 500 Hz command re-writer:
    the last command is re-sent every 2 ms as packet-loss insurance."""

    def __init__(self, config):
        super().__init__(config)
        self._snap_lock = threading.Lock()
        self._snap = None
        threading.Thread(target=self._rewriter, daemon=True).start()

    def send_cmd(self):
        with self._snap_lock:
            super().send_cmd()
            self._snap = copy.deepcopy(self.low_cmd)

    def _rewriter(self):
        while True:
            with self._snap_lock:
                if self._snap is not None:
                    self.lowcmd_publisher_.Write(self._snap)
            time.sleep(0.002)


class EskinController:
    def __init__(self, config, robot, policy_runner, camera, tactile, recorder):
        self.config = config
        self.robot = robot
        self.policy_runner = policy_runner
        self.camera = camera
        self.tactile = tactile
        self.recorder = recorder

        self.teleop = None
        self.blend = False
        self.record_only = policy_runner is None
        self.is_sonic = not self.record_only and policy_runner.reference is not None
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

        if self.record_only:                      # hold default, no inference
            action = np.zeros(self.config.num_actions, np.float32)
            target_dof_pos = self.config.default_angles_23.copy()
            self.robot.apply_target_dof_pos(target_dof_pos)
        else:
            depth_raw = (self.camera.latest_depth
                         if (self.camera and self.config.use_depth_obs) else None)
            if self.is_sonic:
                ref = self.policy_runner.reference
                if not ref.initialized:
                    ref.initialize_from_robot(imu_quat, self.robot.qj)
                # SONIC builds its own obs from RAW state, not the scaled features
                action = self.policy_runner.compute_action(
                    dof_pos=self.robot.qj,
                    dof_vel=self.robot.dqj,
                    base_quat_wxyz=imu_quat,
                    base_ang_vel_body=imu_gyro,
                )
                ref.step()
                if self.teleop is not None:
                    rc = self.robot.remote_controller
                    self.teleop.set_axes(rc.lx, rc.ly, rc.rx)
                    self.teleop.tick(self.config.control_dt)
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
            # sim has no startup blend; --blend eases the handover anyway
            if self.blend and self.policy_runner.act_count < 50:
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

    def ramp_to(self, target, seconds: float = 3.0, label: str = "") -> None:
        """Linearly interpolate from the current pose to `target`."""
        self.robot.update_joint_state()
        start = self.robot.qj.copy()
        target = np.asarray(target, np.float32)
        n = max(1, int(seconds / self.config.control_dt))
        gap = np.degrees(np.abs(target - start).max())
        print(f"\n{label or 'ramping'}: {gap:.0f} deg over {seconds:.1f}s ...")
        for i in range(n):
            a = (i + 1) / n
            self.robot.apply_target_dof_pos((1 - a) * start + a * target)
            time.sleep(self.config.control_dt)

    def ramp_to_default(self, seconds: float = 2.0) -> None:
        self.ramp_to(self.config.default_angles_23, seconds, "returning to default")

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

        if self.is_sonic:
            sample.update(self.policy_runner.reference.teleop_fields())

        if self.tactile is not None:
            sample.update(
                tactile_fields(self.tactile.snapshot(), self.config.tactile_sensor_order)
            )

        # camera is 30 Hz, loop is 50: only write bytes when the frame changed
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


def build_policy_runner(config, mode: str, teleop: str = None, gated: bool = True):
    """SONIC encoder/decoder driven by `mode`, the actor runner, or nothing."""
    if mode == "record":
        return None            # hold position and only record
    if config.policy_type == "sonic":
        from policy.sonic_runner import SonicRunner

        if mode == "pico" and teleop:
            from policy.pico_reference import PicoReferenceWindow

            if teleop == "joystick":
                from common.joystick_teleop import JoystickTeleop

                client = JoystickTeleop()
            else:
                from common.scripted_teleop import ScriptedTeleop

                client = ScriptedTeleop(teleop)
            return SonicRunner(config, PicoReferenceWindow(config, client, gated=gated))

        from policy.sonic_reference import build_reference

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
    ap.add_argument("--clip", default=None,
                    help="--mode motion: override sonic_motion_clip from the config")
    ap.add_argument("--loop", action="store_true",
                    help="--mode motion: repeat the clip instead of stopping at the end")
    ap.add_argument("--teleop", choices=["nav", "vr3", "both", "idle", "carry", "joystick"],
                    default=None,
                    help="--mode pico: scripted command instead of a headset")
    ap.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    ap.add_argument("--blend", action="store_true",
                    help="ease the first second toward policy targets (sim has none)")
    ap.add_argument("--replan-ungated", action="store_true",
                    help="--mode pico: replan on every command instead of official_gated")
    ap.add_argument("--net", default="enp132s0")
    ap.add_argument("--tactile-ip", default=None,
                    help="override config; use 127.0.0.1 when the gloves are "
                         "plugged into this workstation")
    ap.add_argument("--camera-ip", default=None, help="override config")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-tactile", action="store_true")
    args = ap.parse_args()

    config = Config(args.config)
    if args.clip:
        config.sonic_motion_clip = args.clip
    config.sonic_motion_loop = args.loop

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
    robot = EskinRobot(config)
    policy_runner = build_policy_runner(config, args.mode, args.teleop,
                                        gated=not args.replan_ungated)

    # X, not A: A is consumed by default_pos_state() to release the loop,
    # so recording on A would start an episode the instant you start moving.
    recorder = Recorder(Path(args.out).expanduser(), {
        "policy_type": config.policy_type,
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
    hl.blend = args.blend

    if args.mode == "pico":
        ref = policy_runner.reference
        robot.update_joint_state()          # qj is zeros until this runs
        ref.initialize_from_robot(robot.get_imu()[0], robot.qj)
        if args.teleop:
            hl.teleop = ref.client

    print("\n  X = toggle recording   A = kill (straight to damping)   "
          "select = quit\n")

    exit_reason = None
    a_released = False        # A is still held from the start press
    try:
        while True:
            hl.step()
            btn = robot.remote_controller.button
            if btn[KeyMap.A] == 0:
                a_released = True
            elif a_released:
                exit_reason = "A"
                break
            if btn[KeyMap.select] == 1:
                exit_reason = "select"
                break
            if hl.is_sonic and policy_runner.reference.done:
                exit_reason = "clip finished"
                break
            if args.duration and hl.elapsed >= args.duration:
                exit_reason = "duration"
                break
    except KeyboardInterrupt:
        exit_reason = "keyboard"

    recorder.stop()  # flush an episode still open

    # Ramp back to the default pose before going limp, so the robot does not
    # drop from wherever the clip ended. A and Ctrl-C skip it: damping NOW.
    if exit_reason not in ("keyboard", "A"):
        hl.ramp_to_default()

    create_damping_cmd(robot.low_cmd)
    robot.send_cmd()
    if camera is not None:
        camera.running = False
    if tactile is not None:
        tactile.stop()

    print(f"\nexit: {exit_reason}")



if __name__ == "__main__":
    main()
