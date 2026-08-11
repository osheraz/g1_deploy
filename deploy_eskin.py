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
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd
from common.depth_client import SimpleDepthClient
from common.episode import Recorder, load_episode  # noqa: F401  (re-exported)
from common.remote_controller import KeyMap
from common.tactile_record import sample_fields as tactile_fields
from config import Config
from robot.robot_interface import RobotInterface


def _rpy_wxyz(q) -> np.ndarray:
    """Pelvis roll/pitch/yaw from a normalized WXYZ LowState IMU quaternion."""
    w, x, y, z = np.asarray(q, np.float64).reshape(4)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], np.float32)


class EskinRobot(RobotInterface):
    """RobotInterface + the official deploy's 500 Hz command re-writer:
    the last command is re-sent every 2 ms as packet-loss insurance."""

    def __init__(self, config):
        super().__init__(config)
        self._snap_lock = threading.Lock()
        self._snap = None
        self._snap_monotonic = 0.0
        self.watchdog_tripped = False
        self.command_publish_count = 0
        self.rewriter_publish_count = 0
        self._rewriter_running = True
        self._rewriter_thread = threading.Thread(
            target=self._rewriter, name="lowcmd-rewriter", daemon=True
        )
        self._rewriter_thread.start()

    def send_cmd(self):
        with self._snap_lock:
            if self.watchdog_tripped:
                raise RuntimeError("motor-command watchdog is latched")
            super().send_cmd()
            self._snap = copy.deepcopy(self.low_cmd)
            self._snap_monotonic = time.monotonic()
            self.command_publish_count += 1

    def _rewriter(self):
        while self._rewriter_running:
            with self._snap_lock:
                if self._snap is not None:
                    age = time.monotonic() - self._snap_monotonic
                    command = self._snap
                    if age > self.config.max_command_age_s:
                        self.watchdog_tripped = True
                        command = copy.deepcopy(self._snap)
                        create_damping_cmd(command)
                        command.crc = CRC().Crc(command)
                    self.lowcmd_publisher_.Write(command)
                    self.rewriter_publish_count += 1
            time.sleep(0.002)

    def send_damping_now(self) -> None:
        """Latch and publish damping even after the command watchdog fires.

        Normal ``send_cmd`` calls are rejected after the watchdog has latched,
        which prevents a stale control loop from resuming.  Emergency damping
        is the one intentional exception and must remain publishable from the
        common cleanup path.
        """
        with self._snap_lock:
            create_damping_cmd(self.low_cmd)
            self.low_cmd.crc = CRC().Crc(self.low_cmd)
            self._snap = copy.deepcopy(self.low_cmd)
            self._snap_monotonic = time.monotonic()
            self.watchdog_tripped = True
            self.lowcmd_publisher_.Write(self._snap)
            self.command_publish_count += 1

    def stop_rewriter(self) -> None:
        self._rewriter_running = False
        self._rewriter_thread.join(timeout=1.0)


ANKLE_IDX = np.array([4, 5, 10, 11])


class EskinController:
    def __init__(self, config, robot, policy_runner, camera, tactile, recorder,
                 feedback=None):
        self.config = config
        self.robot = robot
        self.policy_runner = policy_runner
        self.camera = camera
        self.tactile = tactile
        self.recorder = recorder
        self.feedback = feedback

        self.teleop = None
        self.blend = False
        self.record_only = policy_runner is None
        self.is_sonic = not self.record_only and policy_runner.reference is not None
        self.counter = 0
        self.elapsed = 0.0
        self._last_camera_seq = -1
        self._t_next = time.perf_counter()
        self._last_step_start = None
        self._blend_origin = None
        self._handover_alpha = 1.0
        self._step_low_state = None
        self._step_low_state_age_s = np.inf

    def step(self) -> None:
        step_start = time.perf_counter()
        loop_period_s = (
            self.config.control_dt if self._last_step_start is None
            else step_start - self._last_step_start
        )
        self._last_step_start = step_start
        self.counter += 1
        self.elapsed += self.config.control_dt

        state, state_age_s = self.robot.low_state_snapshot_with_age()
        self._step_low_state = state
        self._step_low_state_age_s = state_age_s
        self.robot.update_joint_state(state)
        if self.robot.watchdog_tripped:
            raise RuntimeError("500Hz motor-command watchdog published damping")
        if state_age_s > self.config.max_low_state_age_s:
            raise RuntimeError(
                f"LowState is stale for {state_age_s:.3f}s "
                f"(limit {self.config.max_low_state_age_s:.3f}s)"
            )
        max_dq = float(np.max(np.abs(self.robot.dqj)))
        if max_dq > self.config.max_joint_velocity:
            raise RuntimeError(
                f"measured joint velocity {max_dq:.2f}rad/s exceeds "
                f"{self.config.max_joint_velocity:.2f}rad/s"
            )
        imu_quat, imu_gyro = self.robot.get_imu(state)
        motor = self.robot.motor_state_arrays(state=state)
        present = self.robot.robot_profile.physical_motor_indices
        temperatures = motor["temperature"][present]
        if not np.isfinite(temperatures).any(axis=1).all():
            raise RuntimeError("a physical motor has no valid temperature telemetry")
        hottest = float(np.nanmax(temperatures))
        if hottest >= self.config.max_motor_temperature_c:
            raise RuntimeError(
                f"motor temperature {hottest:.1f}C exceeds "
                f"{self.config.max_motor_temperature_c:.1f}C"
            )
        faults = motor["motor_error"][present]
        if np.any(faults != 0):
            i = int(np.flatnonzero(faults != 0)[0])
            raise RuntimeError(
                f"motor {int(present[i])} reports fault code {int(faults[i])}"
            )
        quat_norm = float(np.linalg.norm(imu_quat))
        if not np.isfinite(quat_norm) or quat_norm < 1e-6:
            raise RuntimeError("pelvis IMU quaternion is invalid")
        pelvis_rpy = _rpy_wxyz(np.asarray(imu_quat) / quat_norm)
        if float(np.linalg.norm(pelvis_rpy[:2])) > self.config.max_pelvis_tilt_rad:
            raise RuntimeError(
                f"pelvis tilt {float(np.linalg.norm(pelvis_rpy[:2])):.3f}rad exceeds "
                f"{self.config.max_pelvis_tilt_rad:.3f}rad"
            )
        tracking = np.abs(self.robot.last_target_dof_pos - self.robot.qj)
        if np.any(tracking > self.config.max_tracking_error_rad):
            i = int(np.argmax(tracking))
            raise RuntimeError(
                f"joint {i} tracking error {tracking[i]:.3f}rad exceeds "
                f"{self.config.max_tracking_error_rad:.3f}rad"
            )
        base_ang_vel, projected_gravity, dof_pos, dof_vel = (
            self.robot.get_base_features_scaled(imu=(imu_quat, imu_gyro))
        )

        if self.record_only:                      # hold default, no inference
            action = np.zeros(self.config.num_actions, np.float32)
            target_dof_pos = self.config.physical_default_angles.copy()
            target_dof_pos = self.robot.apply_target_dof_pos(target_dof_pos)
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
                action * self.config.action_scale + self.config.physical_default_angles
            )
            # Raised-cosine policy handover. The official controller switches
            # directly from its default-pose hold to the first policy target,
            # but that target is not generally identical to the measured pose.
            # Start at the exact measured pose and reach the policy at tick 50;
            # --no-blend exists for numerical parity/off-robot diagnostics.
            self._handover_alpha = 1.0
            handover_steps = 50
            if self.blend and self.policy_runner.act_count <= handover_steps:
                if self._blend_origin is None:
                    self._blend_origin = self.robot.qj.copy()
                phase = np.clip(
                    (self.policy_runner.act_count - 1) / (handover_steps - 1),
                    0.0,
                    1.0,
                )
                alpha = 0.5 - 0.5 * np.cos(np.pi * phase)
                self._handover_alpha = float(alpha)
                target_dof_pos = (
                    alpha * target_dof_pos + (1.0 - alpha) * self._blend_origin
                )
            target_dof_pos = self.robot.apply_target_dof_pos(target_dof_pos)

        if self.feedback is not None:
            self.feedback.publish(
                q_physical=self.robot.qj,
                dq_physical=self.robot.dqj,
                command_q_physical=target_dof_pos,
                base_quat_wxyz=imu_quat,
                base_ang_vel=imu_gyro,
            )

        self.recorder.handle_buttons(self.robot.remote_controller.button)
        if self.recorder.recording:
            self._record(
                action, target_dof_pos, imu_quat, base_ang_vel,
                projected_gravity, dof_pos, dof_vel,
                loop_period_s=loop_period_s,
                compute_s=time.perf_counter() - step_start,
            )

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
        """Interpolate to ``target`` while retaining live-state safety gates."""
        state, age = self.robot.low_state_snapshot_with_age()
        if age > self.config.max_low_state_age_s:
            raise RuntimeError(
                f"cannot start pose ramp: LowState is stale for {age:.3f}s"
            )
        self.robot.update_joint_state(state)
        start = self.robot.qj.copy()
        target = np.asarray(target, np.float32)
        n = max(1, int(seconds / self.config.control_dt))
        gap = np.degrees(np.abs(target - start).max())
        print(f"\n{label or 'ramping'}: {gap:.0f} deg over {seconds:.1f}s ...")
        for i in range(n):
            state, age = self.robot.low_state_snapshot_with_age()
            if age > self.config.max_low_state_age_s:
                raise RuntimeError(
                    f"LowState became stale during pose ramp ({age:.3f}s)"
                )
            self.robot.update_joint_state(state)
            max_dq = float(np.max(np.abs(self.robot.dqj)))
            if max_dq > self.config.max_joint_velocity:
                raise RuntimeError(
                    f"measured joint velocity {max_dq:.2f}rad/s during pose "
                    f"ramp exceeds {self.config.max_joint_velocity:.2f}rad/s"
                )
            a = (i + 1) / n
            self.robot.apply_target_dof_pos((1 - a) * start + a * target)
            time.sleep(self.config.control_dt)

    def ramp_to_default(self, seconds: float = 2.0) -> None:
        self.ramp_to(self.config.physical_default_angles, seconds, "returning to default")

    def hold_and_verify(
        self,
        target,
        seconds: float,
        *,
        max_error_deg: float,
        max_velocity: float,
        label: str,
    ) -> None:
        """Hold a pose and refuse policy activation unless hardware has settled."""
        target = np.asarray(target, np.float32).reshape(self.config.num_actions)
        n = max(1, int(seconds / self.config.control_dt))
        tail_start = max(0, n - max(5, n // 4))
        tail_error_deg = []
        tail_per_joint = []
        tail_velocity = []
        print(f"{label}: holding for {seconds:.1f}s ...")
        for i in range(n):
            state, age = self.robot.low_state_snapshot_with_age()
            if age > self.config.max_low_state_age_s:
                raise RuntimeError(
                    f"LowState became stale during settle hold ({age:.3f}s)"
                )
            self.robot.update_joint_state(state)
            speed = float(np.max(np.abs(self.robot.dqj)))
            if speed > self.config.max_joint_velocity:
                raise RuntimeError(
                    f"measured joint velocity {speed:.2f}rad/s during settle hold "
                    f"exceeds {self.config.max_joint_velocity:.2f}rad/s"
                )
            self.robot.apply_target_dof_pos(target)
            if i >= tail_start:
                err = np.abs(target - self.robot.qj)
                tail_error_deg.append(float(np.degrees(err.max())))
                tail_per_joint.append(np.degrees(np.abs(target - self.robot.qj)))
                tail_velocity.append(speed)
            time.sleep(self.config.control_dt)

        observed_error = max(tail_error_deg, default=float("inf"))
        observed_velocity = max(tail_velocity, default=float("inf"))
        print(
            f"{label}: tail max error={observed_error:.2f}deg, "
            f"tail max |dq|={observed_velocity:.3f}rad/s"
        )
        if tail_per_joint:
            per = np.max(tail_per_joint, axis=0)
            print(f"{label}: per-joint deviation (deg, raw):")
            for j in np.argsort(-per):
                mark = " (ankle)" if j in ANKLE_IDX else ""
                name = self.robot.robot_profile.physical_joint_names[j]
                print(f"    {name:28s} {per[j]:6.2f}{mark}")
        if observed_error > max_error_deg or observed_velocity > max_velocity:
            raise RuntimeError(
                f"motion frame 0 did not settle: error={observed_error:.2f}deg "
                f"(limit {max_error_deg:.2f}), |dq|={observed_velocity:.3f}rad/s "
                f"(limit {max_velocity:.3f})"
            )

    def _record(self, action, target_dof_pos, imu_quat,
                base_ang_vel, projected_gravity, dof_pos, dof_vel,
                *, loop_period_s, compute_s) -> None:
        motor = self.robot.motor_state_arrays(state=self._step_low_state)
        pelvis_rpy = _rpy_wxyz(imu_quat)
        sample = {
            "t": np.float32(self.elapsed),
            "t_monotonic": np.float64(time.monotonic()),
            "measured_q_physical": self.robot.qj.copy().astype(np.float32),
            "measured_dq_physical": self.robot.dqj.copy().astype(np.float32),
            # Backward-compatible aliases used by replay_episode.py.
            "qj": self.robot.qj.copy().astype(np.float32),
            "dqj": self.robot.dqj.copy().astype(np.float32),
            "action": action.copy().astype(np.float32),
            "target_dof_pos": target_dof_pos.copy().astype(np.float32),
            "command_q_physical": self.robot.last_target_dof_pos.copy(),
            "command_q_physical_unclipped": self.robot.last_target_unclipped.copy(),
            "command_saturated_physical": self.robot.last_target_saturated.copy(),
            "tracking_error_q_physical": (
                self.robot.last_target_dof_pos - self.robot.qj
            ).astype(np.float32),
            "command_q_motor_29": self.robot.commanded_q_29(),
            "measured_q_motor_29": motor["q"],
            "measured_dq_motor_29": motor["dq"],
            "motor_temperature_29x2": motor["temperature"],
            "motor_error_29": motor["motor_error"],
            "motor_tau_est_29": motor["tau_est"],
            "base_ang_vel": base_ang_vel.copy().astype(np.float32),
            "projected_gravity": projected_gravity.copy().astype(np.float32),
            "dof_pos_scaled": dof_pos.copy().astype(np.float32),
            "dof_vel_scaled": dof_vel.copy().astype(np.float32),
            "imu_quat_wxyz": np.asarray(imu_quat, np.float32),
            # Base/pelvis roll and pitch are real proprioception on a 23-DoF
            # G1. SONIC receives the same information through gravity history;
            # these explicit fields make hardware diagnosis straightforward.
            "pelvis_roll_rad": pelvis_rpy[0],
            "pelvis_pitch_rad": pelvis_rpy[1],
            "pelvis_yaw_rad": pelvis_rpy[2],
            "low_state_age_s": np.float32(self._step_low_state_age_s),
            "low_state_tick": np.int64(
                getattr(self._step_low_state, "tick", -1)
            ),
            "loop_period_s": np.float32(loop_period_s),
            "control_compute_s": np.float32(compute_s),
            "policy_handover_alpha": np.float32(self._handover_alpha),
            "command_publish_count": np.int64(self.robot.command_publish_count),
            "rewriter_publish_count": np.int64(self.robot.rewriter_publish_count),
            "low_state_crc_rejected_count": np.int64(self.robot.rejected_crc_count),
        }

        # Preserve the existing episode schema only for the embodiment it
        # actually describes. Generic readers should use the *_physical keys.
        if self.robot.robot_profile.name == "g1_23dof":
            sample.update({
                "measured_q_23": sample["measured_q_physical"],
                "measured_dq_23": sample["measured_dq_physical"],
                "command_q_23": sample["command_q_physical"],
                "command_q_23_unclipped": sample["command_q_physical_unclipped"],
                "command_saturated_23": sample["command_saturated_physical"],
                "tracking_error_q_23": sample["tracking_error_q_physical"],
            })

        if self.is_sonic:
            sample.update(self.policy_runner.diagnostic_fields())
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


def build_policy_runner(config, mode: str, teleop: str = None, gated: bool = True,
                        idle_hold: bool = False, slow: bool = False):
    """SONIC encoder/decoder driven by `mode`, the actor runner, or nothing."""
    if mode == "record":
        return None            # hold position and only record
    if config.policy_type == "sonic":
        from policy.sonic_runner import SonicRunner

        if mode in ("pico", "planner", "smpl") and teleop:
            from policy.pico_reference import PicoReferenceWindow

            if teleop == "joystick":
                from common.joystick_teleop import JoystickTeleop
                from common.pico_client import LocomotionMode

                client = JoystickTeleop(
                    locomotion_mode=LocomotionMode.SLOW_WALK if slow
                    else LocomotionMode.WALK)
            else:
                from common.scripted_teleop import ScriptedTeleop

                client = ScriptedTeleop(teleop)
            return SonicRunner(
                config,
                PicoReferenceWindow(
                    config,
                    client,
                    gated=gated,
                    force_g1=(mode == "planner"),
                    force_smpl=(mode == "smpl"),
                    idle_hold=idle_hold,
                ),
            )

        from policy.sonic_reference import build_reference

        return SonicRunner(config, build_reference(config, mode, gated=gated,
                                                   idle_hold=idle_hold))

    from policy.policy_runner import PolicyRunner

    return PolicyRunner(config)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="g1_23dof_sonic.yaml")
    ap.add_argument(
        "--mode",
        choices=["record", "motion", "pico", "planner", "smpl"],
        default="record",
                    help="record: hold position, stream+record only (no policy). "
                         "motion: replay an exported SONIC clip. pico: live, "
                         "message-selected G1/VR3. planner: planner-only G1 "
                         "encoder with planner-controlled upper body.")
    ap.add_argument("--out", default="episodes", help="episode output directory")
    ap.add_argument("--clip", default=None,
                    help="--mode motion: override sonic_motion_clip from the config")
    ap.add_argument("--loop", action=argparse.BooleanOptionalAction, default=None,
                    help="--mode motion: override the config's finite/loop setting")
    ap.add_argument("--teleop", choices=["nav", "vr3", "both", "idle", "carry", "joystick"],
                    default=None,
                    help="--mode pico/planner: scripted or Unitree-remote command "
                         "instead of a headset")
    ap.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    ap.add_argument("--blend", action=argparse.BooleanOptionalAction, default=True,
                    help="raised-cosine handover to policy targets over one second "
                         "(safe default; --no-blend is parity/debug only)")
    ap.add_argument("--replan-ungated", action="store_true",
                    help="--mode pico/planner: replan on every command instead of "
                         "official gating")
    ap.add_argument("--idle-hold", action="store_true",
                    help="--mode pico/planner: IDLE holds the default pose instead "
                         "of the planner's idle plan (which contains stepping)")
    ap.add_argument("--slow", action="store_true",
                    help="--teleop joystick: SLOW_WALK (0.1-0.6 m/s stick-scaled) "
                         "instead of WALK (~0.9 m/s planner default)")
    ap.add_argument("--net", default="enp132s0")
    ap.add_argument("--auto-record", action="store_true",
                    help="start experiment logging as soon as control begins")
    ap.add_argument("--feedback-port", type=int, default=None,
                    help="g1_debug PUB port for official PICO VR3 wrist calibration")
    ap.add_argument("--no-feedback", action="store_true",
                    help="disable g1_debug publication (VR3 calibration will lack measured q)")
    ap.add_argument("--motion-init-seconds", type=float, default=3.0,
                    help="ramp to a motion clip's frame-0 joint pose before inference")
    ap.add_argument("--max-motion-init-gap-deg", type=float, default=20.0,
                    help="abort motion mode if frame 0 is farther from measured q than this")
    ap.add_argument("--motion-settle-seconds", type=float, default=1.0,
                    help="hold frame 0 this long and verify measured settling")
    ap.add_argument("--max-motion-settle-error-deg", type=float, default=5.0,
                    help="maximum tail pose error before motion policy activation")
    ap.add_argument("--max-motion-settle-velocity", type=float, default=0.5,
                    help="maximum tail |dq| rad/s before motion policy activation")
    ap.add_argument("--tactile-ip", default=None,
                    help="override config; use 127.0.0.1 when the gloves are "
                         "plugged into this workstation")
    ap.add_argument("--camera-ip", default=None, help="override config")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-tactile", action="store_true")
    args = ap.parse_args()

    config = Config(args.config)
    if args.clip:
        config.sonic_motion_clip = str(Path(args.clip).expanduser().resolve())
    if args.loop is not None:
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

    policy_runner = build_policy_runner(config, args.mode, args.teleop,
                                        gated=not args.replan_ungated,
                                        idle_hold=args.idle_hold, slow=args.slow)

    feedback = None
    if args.mode in ("pico", "smpl") and not args.no_feedback:
        from common.robot_feedback import RobotFeedbackPublisher

        feedback = RobotFeedbackPublisher(
            port=args.feedback_port or config.pico_feedback_port,
            robot_profile=config.sonic_robot_profile,
        )

    missing_action_indices = []
    missing_action_names = []
    if config.policy_type == "sonic":
        from policy.sonic.robot_profile import get_robot_profile

        profile = get_robot_profile(config.sonic_robot_profile)
        missing_action_indices = [int(i) for i in profile.missing_isaaclab]
        missing_action_names = list(profile.missing_names)
    reference_for_meta = getattr(policy_runner, "reference", None)
    planner_for_meta = getattr(reference_for_meta, "planner", None)

    # X, not A: A is consumed by default_pos_state() to release the loop,
    # so recording on A would start an episode the instant you start moving.
    recorder = Recorder(Path(args.out).expanduser(), {
        "policy_type": config.policy_type,
        "mode": args.mode,
        "sonic_model_variant": config.sonic_model_variant,
        "sonic_robot_profile": config.sonic_robot_profile,
        "force_g1_encoder": bool(args.mode == "planner"),
        "config": str(config.config_path),
        "policy": str(config.policy_path),
        "control_dt": float(config.control_dt),
        "rate_hz": 1.0 / float(config.control_dt),
        "network_interface": args.net,
        "encoder_model": str(config.sonic_encoder),
        "decoder_model": str(config.sonic_decoder),
        "planner_model": str(config.sonic_planner_onnx),
        "encoder_providers": list(getattr(policy_runner, "encoder_providers", ())),
        "decoder_providers": list(getattr(policy_runner, "decoder_providers", ())),
        "planner_providers": list(getattr(planner_for_meta, "providers", ())),
        "motion_clip": str(config.sonic_motion_clip),
        "motion_loop": bool(config.sonic_motion_loop),
        "sonic_mask_missing_reference": bool(config.sonic_mask_missing_reference),
        "sonic_missing_action_il_indices": missing_action_indices,
        "sonic_missing_action_order": missing_action_names,
        "physical_joint_names": list(profile.physical_joint_names) if config.policy_type == "sonic" else [],
        "policy_handover_enabled": bool(args.blend),
        "motion_settle_seconds": float(args.motion_settle_seconds),
        "max_motion_settle_error_deg": float(args.max_motion_settle_error_deg),
        "max_motion_settle_velocity": float(args.max_motion_settle_velocity),
        "sensors": list(config.tactile_sensor_order) if tactile else [],
        "tactile_raw_len": 256,
        "tactile_imu_len": 16,
        "camera_shape": [int(config.depth_raw_height), int(config.depth_raw_width)],
    }, button=KeyMap.X)

    hl = None
    robot = None
    exit_reason = None
    pending_error = None
    try:
        # Load models, open feedback, and construct the recorder before DDS.
        # Once RobotInterface exists, every later failure is inside this common
        # damping/cleanup path.
        ChannelFactoryInitialize(0, args.net)
        robot = EskinRobot(config)
        reference = getattr(policy_runner, "reference", None)
        if reference is not None and hasattr(reference, "robot"):
            reference.robot = robot

        robot.move_to_default_pos()
        robot.default_pos_state()

        hl = EskinController(
            config, robot, policy_runner, camera, tactile, recorder, feedback=feedback
        )
        hl.blend = args.blend

        if args.mode == "motion":
            ref = policy_runner.reference
            robot.update_joint_state()
            frame0 = ref.initial_joint_pose
            gap_deg = float(np.degrees(np.max(np.abs(frame0 - robot.qj))))
            if gap_deg > args.max_motion_init_gap_deg:
                raise RuntimeError(
                    f"motion initialization refused: frame 0 is {gap_deg:.1f} deg "
                    f"from the measured pose (limit "
                    f"{args.max_motion_init_gap_deg:.1f} deg). Use a "
                    "neutral-start prepared clip; do not raise the limit on an "
                    "unsupported robot."
                )
            hl.ramp_to(frame0, args.motion_init_seconds, "aligning to motion frame 0")
            hl.hold_and_verify(
                frame0,
                args.motion_settle_seconds,
                max_error_deg=args.max_motion_settle_error_deg,
                max_velocity=args.max_motion_settle_velocity,
                label="verifying motion frame 0",
            )
            robot.update_joint_state()
            ref.initialize_from_robot(robot.get_imu()[0], robot.qj)

        if args.mode in ("pico", "planner", "smpl"):
            ref = policy_runner.reference
            robot.update_joint_state()
            ref.initialize_from_robot(robot.get_imu()[0], robot.qj)
            if args.teleop:
                hl.teleop = ref.client

        if feedback is not None:
            imu_quat, imu_gyro = robot.get_imu()
            feedback.publish(
                q_physical=robot.qj,
                dq_physical=robot.dqj,
                command_q_physical=robot.last_target_dof_pos,
                base_quat_wxyz=imu_quat,
                base_ang_vel=imu_gyro,
            )
        if args.auto_record:
            recorder.start()

        print("\n  X = toggle recording   A = kill (straight to damping)   "
              "select = quit\n")

        a_released = False        # A is still held from the start press
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
    except BaseException as exc:
        exit_reason = f"error: {type(exc).__name__}"
        pending_error = exc
    finally:
        # Any failure takes the shortest path to damping. Normal completion may
        # first return to the default pose, but A/Ctrl-C/errors never do.
        try:
            recorder.stop()
        except BaseException as exc:
            print(f"\nwarning: recorder shutdown failed: {exc}")
            if pending_error is None:
                pending_error = exc
                exit_reason = f"error: {type(exc).__name__}"

        if hl is not None and pending_error is None and exit_reason not in ("keyboard", "A"):
            try:
                hl.ramp_to_default()
            except BaseException as exc:
                # A failed return ramp is a control failure: proceed directly
                # to damping and preserve the exception for the caller.
                print(f"\nwarning: return ramp failed; damping now: {exc}")
                pending_error = exc
                exit_reason = f"error: {type(exc).__name__}"

        # Damping is deliberately attempted before stopping auxiliary threads.
        # No recorder/planner/socket cleanup exception may prevent this packet.
        if robot is not None:
            try:
                robot.send_damping_now()
                # Allow the 500 Hz writer to repeat the damping packet before stopping.
                time.sleep(0.05)
            except BaseException as exc:
                print(f"\nwarning: failed to publish damping command: {exc}")
                if pending_error is None:
                    pending_error = exc
                    exit_reason = f"error: {type(exc).__name__}"
            finally:
                try:
                    robot.stop_rewriter()
                except BaseException as exc:
                    print(f"\nwarning: command rewriter shutdown failed: {exc}")
                    if pending_error is None:
                        pending_error = exc
                        exit_reason = f"error: {type(exc).__name__}"

        reference = getattr(policy_runner, "reference", None)
        stop_reference = getattr(reference, "stop", None)
        if callable(stop_reference):
            try:
                stop_reference()
            except BaseException as exc:
                print(f"\nwarning: planner shutdown failed: {exc}")
                if pending_error is None:
                    pending_error = exc
                    exit_reason = f"error: {type(exc).__name__}"

        if feedback is not None:
            try:
                feedback.close()
            except BaseException as exc:
                print(f"\nwarning: feedback shutdown failed: {exc}")
                if pending_error is None:
                    pending_error = exc
                    exit_reason = f"error: {type(exc).__name__}"
        if camera is not None:
            camera.running = False
        if tactile is not None:
            try:
                tactile.stop()
            except BaseException as exc:
                print(f"\nwarning: tactile shutdown failed: {exc}")
                if pending_error is None:
                    pending_error = exc
                    exit_reason = f"error: {type(exc).__name__}"

    print(f"\nexit: {exit_reason}")
    if pending_error is not None:
        raise pending_error



if __name__ == "__main__":
    main()
