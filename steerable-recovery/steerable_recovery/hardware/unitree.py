"""Thin G1 hg sensor and watchdog transport, independent of the g1_deploy submodule.

Official references: unitree_sdk2_python's G1 low-level
example and generated hg MotorCmd/LowState IDL. PR mode is0; velocity command
field is ``dq`` (not the ``qd`` spelling in some historical deploy helpers).
Importing this module opens no channels. A read-only instance has no publisher.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import math
import threading
import time

import numpy as np

from humanoidverse2.observations.streams import (
    motor_indices,
    observation_from_lowstate,
    CONTROL_OBSERVATION_WARN_S,
    StaleRobotObservation,
)


from humanoidverse2.robots.g1.control import NativePDContract


def default_pose_path(initial, default, elapsed_s):
    """The VIGOR startup path: measured joints -> default, linearly over 2 s."""
    initial = np.asarray(initial, dtype=np.float32)
    default = np.asarray(default, dtype=np.float32)
    if (
        initial.shape != (29,)
        or default.shape != (29,)
        or not np.isfinite(initial).all()
        or not np.isfinite(default).all()
        or not math.isfinite(elapsed_s)
        or elapsed_s < 0
    ):
        raise ValueError("Finite 29-joint endpoints and nonnegative elapsed time required")
    duration = 2.0
    return initial + (default - initial) * min(float(elapsed_s) / duration, 1.0), duration


def wait_for_policy_start(
    robot,
    pd,
    *,
    pose_only=False,
    preview=None,
    activate=None,
    start_pose=None,
    reset_preview=None,
    pose_event=None,
):
    """Wait for controls; A starts from here, START requests positioning.

    Called only before the first policy start. Button edges must be fresh after
    readiness; holding a button during startup never triggers a movement.
    """
    if not robot.enabled and activate is None:
        raise ValueError("Operator motor controls require explicit actuation")
    if start_pose is not None:
        from motion_diffusion.inference.start_pose import validate_start_pose

        validate_start_pose(start_pose, pd)
        if pose_only:
            raise ValueError("A saved start pose is for recovery, not default-pose-only mode")
    presses = robot.a_presses
    pose_requests = robot.default_pose_requests
    robot.arm_a_stop(after_press=presses if pose_only else presses + 1)
    print(
        (
            "Ready — zero torque active. "
            if getattr(robot, "zero_torque_active", False)
            else "Ready — no motor commands sent. "
        )
        + (
            "START: move to default. SELECT or A: stop. No policy in this mode."
            if pose_only
            else (
                f'A: start policy from current pose. START: optional saved {start_pose["posture"]} joint pose. SELECT: stop.'
                if start_pose is not None
                else "A: start policy from current pose. START: optional default pose. SELECT: stop."
            )
        ),
        flush=True,
    )
    while True:
        robot.observation()
        ready = True if preview is None else preview()
        if not pose_only and robot.a_presses > presses:
            robot.observation()  # Two rapid A presses must stop, not start.
            if not ready:
                print(
                    (
                        "Policy not ready yet. Zero torque remains active. "
                        if getattr(robot, "zero_torque_active", False)
                        else "Policy not ready yet. No takeover or motor commands. "
                    )
                    + "Press A again after ready.",
                    flush=True,
                )
                presses = robot.a_presses
                robot.arm_a_stop(after_press=presses + 1)
                continue
            if activate is not None:
                activate()
            if start_pose is not None and pose_event is not None:
                obs = robot.observation()
                pose_event(
                    dict(
                        event="start_pose_skipped",
                        monotonic_s=time.monotonic(),
                        pose_name=start_pose["name"],
                        q=obs.q.copy(),
                        dq=obs.dq.copy(),
                        reason="A started policy from current measured pose; START was not requested",
                    )
                )
            break
        if robot.default_pose_requests > pose_requests:
            robot.arm_a_stop(after_press=presses)  # A during the transition cancels it.
            robot.observation()
            if activate is not None:
                activate()
            prepare_default_pose(
                robot,
                pd,
                pose_only=pose_only,
                preview=preview,
                start_pose=start_pose,
                reset_preview=reset_preview,
                pose_event=pose_event,
            )
            break
        time.sleep(0.001 if preview is not None else 0.01)
    print(
        "Policy starting. A again or SELECT: stop in damping and exit. START is now ignored.",
        flush=True,
    )


def prepare_default_pose(
    robot,
    pd,
    *,
    pose_only=False,
    preview=None,
    start_pose=None,
    reset_preview=None,
    pose_event=None,
):
    """Explicitly requested default pose -> hold -> A starts policy.

    During the move, or in pose-only mode, A stops rather than starting a policy.
    Caller must close the robot on every exit. Live previews never issue policy commands.
    """
    if not robot.enabled:
        raise ValueError("Default-pose movement requires explicit actuation")
    robot.arm_a_stop()
    observation = robot.observation()
    initial = observation.q.copy()
    goal = pd.default
    path = default_pose_path
    label = "default"
    if start_pose is not None:
        from motion_diffusion.inference.start_pose import validate_start_pose, start_pose_path

        goal = validate_start_pose(start_pose, pd)
        path = start_pose_path
        label = start_pose["name"]
    if ((goal < pd.lower) | (goal > pd.upper)).any():
        raise RuntimeError("Decoder default pose exceeds robot joint limits")
    _, duration = path(initial, goal, 0.0)
    print(
        f"Moving to {label} over {duration:.1f} seconds. SELECT or A: cancel and damp.", flush=True
    )

    def report(event, obs, target):
        if pose_event is not None:
            pose_event(
                dict(
                    event=event,
                    monotonic_s=time.monotonic(),
                    pose_name=label,
                    q=obs.q.copy(),
                    dq=obs.dq.copy(),
                    target=target.copy(),
                    duration_s=duration,
                )
            )

    def send(target):
        obs = robot.observation()
        checked = pd.command((target - pd.default) / pd.scale, obs)
        robot.publish(checked)
        report("start_pose_control", obs, checked)
        return obs

    # First command holds the actual measured pose, not the final default.
    report("start_pose_begin", observation, goal)
    send(initial)
    start = time.monotonic()
    while True:
        time.sleep(0.02)
        elapsed = time.monotonic() - start
        target, _ = path(initial, goal, elapsed)
        observation = send(target)
        if elapsed >= duration:
            break
    report("start_pose_ramp_complete", observation, goal)
    if reset_preview is not None:
        reset_preview()
    # SONIC holds the default target after its ramp without a tracking-error
    # or settling gate. Spring setpoints need not equal measured joint angles.
    presses = robot.a_presses
    if not pose_only:
        # Permit exactly one new A press to start. The next press is already
        # armed in the receive callback, even if inference blocks the main loop.
        robot.arm_a_stop(after_press=presses + 1)
    print(
        ("Default-pose" if start_pose is None else "Saved-pose")
        + " ramp complete; holding target. "
        + (
            "Pose-only: SELECT or A stops in damping."
            if pose_only
            else "A: start policy. SELECT: stop holding and damp."
        ),
        flush=True,
    )
    next_hold = time.monotonic() + 0.02
    while True:
        # Preview has its own 50 Hz clock. Sleeping 20 ms plus publish/inference
        # work made its clock drift, repeatedly clearing startup readiness.
        time.sleep(0.001 if preview is not None else 0.02)
        now = time.monotonic()
        if now >= next_hold:
            obs = send(goal)
            next_hold = now + 0.02
        else:
            robot.observation()  # Keep SELECT responsive between hold updates.
        ready = True if preview is None else preview()
        if not pose_only and robot.a_presses > presses:
            robot.observation()  # A second rapid press must stop, not start.
            if not ready:
                reason = getattr(preview, "readiness_reason", "waiting for live inference")
                print(
                    f"Policy not ready yet: {reason}; holding {label}. Press A after live inference is ready.",
                    flush=True,
                )
                presses = robot.a_presses
                robot.arm_a_stop(after_press=presses + 1)
                continue
            if start_pose is not None:
                obs = robot.observation()
                error = np.abs(obs.q - goal)
                j = int(error.argmax())
                print(
                    f"Saved-pose joint error at A: max {error[j]:.3f} rad ({pd.dof_names[j]}). "
                    "Policy history starts from measured joints.",
                    flush=True,
                )
                report("start_pose_at_policy_start", obs, goal)
            return


def fill_lowcmd(message, *, indices, target, kp, kd, mode_machine, damping=False):
    """Populate all commanded motors, leaving reserved motors disabled."""
    message.mode_pr = 0
    message.mode_machine = int(mode_machine)
    for motor in message.motor_cmd:
        try:
            motor.dq
        except AttributeError as exc:
            raise ValueError("hg SDK command schema missing dq") from exc
        motor.mode = 0
        motor.q = 0.0
        motor.dq = 0.0
        motor.tau = 0.0
        motor.kp = 0.0
        motor.kd = 0.0
    for j, index in enumerate(indices):
        motor = message.motor_cmd[index]
        motor.mode = 1
        motor.q = 0.0 if damping else float(target[j])
        motor.dq = 0.0
        motor.tau = 0.0
        motor.kp = 0.0 if damping else float(kp[j])
        motor.kd = float(kd[j])


class UnitreeHG:
    """CRC-checked coherent sensor snapshots and optional500Hz command rewrite.

    ``capture_clock=None`` estimates tick alignment from startup packets.
    A supplied mapping is used directly. The startup estimate cannot measure
    constant acquisition/transport latency; its measured spread is logged.
    ``enable_actuation=False`` constructs no motor publisher or rewrite thread.
    """

    def __init__(
        self,
        *,
        interface,
        dof_names,
        capture_clock,
        tick_period_s,
        enable_actuation=False,
        pd_contract=None,
        max_command_age_s=0.06,
        domain_id=0,
    ):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        from steerable_recovery.hardware.fast_crc import CRC

        if not math.isfinite(tick_period_s) or tick_period_s <= 0:
            raise ValueError("Qualified LowState tick period required")
        if enable_actuation and pd_contract is None:
            raise ValueError("Explicit verified PD contract required")
        self.dof_names = dof_names
        self.clock = capture_clock
        self.tick_period = float(tick_period_s)
        from steerable_recovery.hardware.clocks import RobotTickClock

        self.tick_alignment = RobotTickClock() if capture_clock is None else None
        self.buttons = 0
        self.a_presses = 0
        self.default_pose_requests = 0
        self.standing_hold_requests = 0
        self._a_stop_after = None
        self.operator_stop = False
        self.operator_stop_reason = "Remote A stop"
        self.pd = pd_contract
        self.enabled = False
        self.max_command_age = float(max_command_age_s)
        if not 0 < self.max_command_age <= 0.1:
            raise ValueError("Command watchdog budget must be in(0,.1] seconds")
        self.crc = CRC()
        self._lock = threading.Lock()
        self._observations = deque(maxlen=512)
        self._latest = None
        self._command = None
        self._command_time = 0.0
        self._stop = threading.Event()
        self.fault = None
        self.rejected_crc = 0
        self.rejected_stale = 0
        self.publisher = None
        self._writer = None
        self.last_transport_error = None
        self.timing_hold = False
        self.timing_hold_count = 0
        self._timing_hold_target = None
        self.zero_torque_active = False
        self._actuation_attempted = False
        ChannelFactoryInitialize(int(domain_id), str(interface))
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        # Handle the small CRC/state packet on the receive thread. The SDK's
        # optional FIFO drops new packets when full, retaining stale ones.
        self.subscriber.Init(self._receive)
        if enable_actuation:
            self.enable_motor_output()

    def enable_motor_output(self):
        """Open the single writer after control ownership has been released."""
        if self.enabled:
            return
        if self.pd is None:
            raise ValueError("Explicit verified PD contract required")
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

        if self.operator_stop:
            raise KeyboardInterrupt(self.operator_stop_reason)
        if self.fault:
            raise RuntimeError(self.fault)
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

        owner = MotionSwitcherClient()
        owner.SetTimeout(3.0)
        owner.Init()
        status, result = owner.CheckMode()
        if status != 0 or not isinstance(result, dict) or "name" not in result:
            raise RuntimeError(
                f"Cannot verify low-level control ownership: status={status}, result={result!r}"
            )
        if result["name"]:
            raise RuntimeError(
                f'Built-in controller {result["name"]!r} is active. '
                "Release the built-in controller before low-level publication. "
                "No motor publisher was created."
            )
        if self.operator_stop:
            raise KeyboardInterrupt(self.operator_stop_reason)
        self._template = unitree_hg_msg_dds__LowCmd_()
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        self.enabled = True
        self._writer = threading.Thread(target=self._rewrite, daemon=True, name="native29-watchdog")
        self._writer.start()

    def enter_zero_torque(self):
        """Continuously rewrite zero gains/torque until the first PD publish."""
        self.observation()  # Require a coherent native-29 state before publication.
        self.enable_motor_output()
        command = deepcopy(self._template)
        fill_lowcmd(
            command,
            indices=self.pd.indices,
            target=np.zeros(29),
            kp=np.zeros(29),
            kd=np.zeros(29),
            mode_machine=self.pd.mode_machine,
        )
        command.crc = self.crc.Crc(command)
        with self._lock:
            if self.operator_stop:
                raise KeyboardInterrupt(self.operator_stop_reason)
            if self.fault:
                raise RuntimeError(self.fault)
            self.zero_torque_active = True
            self._actuation_attempted = True
            self._write_checked(command)
            self._command = command
            self._command_time = time.monotonic()
            self.timing_hold = False
            self._timing_hold_target = None

    def _receive(self, message):
        arrival = time.monotonic()
        # SDK CRC packing is read-only. Copying the full 35-motor message on
        # every 1 kHz report needlessly competes with the policy thread.
        if int(message.crc) != int(self.crc.Crc(message)):
            self.rejected_crc += 1
            return
        try:
            remote = bytes(message.wireless_remote)
            if len(remote) < 4:
                raise ValueError("Missing wireless remote state")
            buttons = int.from_bytes(remote[2:4], "little")
            with self._lock:
                if buttons & (1 << 8) and not self.buttons & (1 << 8):
                    self.a_presses += 1
                    if self._a_stop_after is not None and self.a_presses > self._a_stop_after:
                        self.operator_stop = True
                if buttons & (1 << 2) and not self.buttons & (1 << 2):
                    self.default_pose_requests += 1
                if buttons & (1 << 11) and not self.buttons & (1 << 11):
                    self.standing_hold_requests += 1
                if (
                    buttons & (1 << 3)
                    and not self.buttons & (1 << 3)
                    and self._a_stop_after is not None
                ):
                    self.operator_stop = True
                    self.operator_stop_reason = "Remote SELECT stop"
                self.buttons = buttons
            with self._lock:
                # DDS can deliver multiple reports carrying the same MCU tick.
                # Ignore duplicates without making an old observation fresh.
                if self._latest is not None and int(message.tick) == self._latest.sequence:
                    return
            if self.tick_alignment is not None:
                capture = self.tick_alignment.observe(int(message.tick) * self.tick_period, arrival)
                if capture is None:
                    return
                self.clock = self.tick_alignment.mapping
            else:
                capture = self.clock.convert(int(message.tick) * self.tick_period)
            obs = observation_from_lowstate(
                message,
                dof_names=self.dof_names,
                capture_s=capture,
                arrival_s=arrival,
                clock_uncertainty_s=self.clock.uncertainty_s,
            )
            obs.validate(arrival, max_age_s=CONTROL_OBSERVATION_WARN_S)
            if int(message.mode_machine) not in (2, 5):
                raise RuntimeError(
                    "LowState is not a full G1-29 profile; locked-waist and23-DoF plants rejected"
                )
            # Run on receipt, independently of inference. A violation latches
            # fault below, so the 500 Hz writer damps even if inference blocks.
            if self.enabled:
                self.pd.check_measured(obs)
            with self._lock:
                if self._latest is not None and obs.sequence <= self._latest.sequence:
                    raise RuntimeError(
                        "LowState tick reset/wrap/out-of-order; reinitialize clock explicitly"
                    )
                self._latest = obs
                self._observations.append(obs)
                if self.pd is not None and int(message.mode_machine) != self.pd.mode_machine:
                    raise RuntimeError(
                        "Actual robot mode_machine differs from the qualified29-DoF profile"
                    )
        except StaleRobotObservation:
            # One excessively delayed DDS packet is not a persistent outage.
            # Keep the last accepted sample and wait for newer feedback. Its
            # original timestamp still expires in observation() if none arrives.
            self.rejected_stale += 1
        except Exception as exc:
            self.fault = str(exc)

    def arm_a_stop(self, *, after_press=None):
        """Latch the next A edge, or the first edge after a permitted start press."""
        with self._lock:
            self._a_stop_after = self.a_presses if after_press is None else after_press
            if self.a_presses > self._a_stop_after:
                self.operator_stop = True

    def observation(self):
        if self.operator_stop:
            raise KeyboardInterrupt(self.operator_stop_reason)
        if self.fault:
            raise RuntimeError(self.fault)
        with self._lock:
            obs = self._latest
        if obs is None:
            if (
                self.tick_alignment is not None
                and self.tick_alignment.report.get("status") == "waiting_for_stable_delivery"
            ):
                jitter = self.tick_alignment.report["startup_jitter_p99_s"] * 1000
                raise RuntimeError(
                    f"Waiting for stable robot timestamps ({jitter:.1f} ms startup jitter)"
                )
            raise RuntimeError("No valid LowState received")
        obs.validate(time.monotonic())
        return obs

    def at_capture(self, timestamp, *, max_error_s=0.005):
        with self._lock:
            values = tuple(self._observations)
        if not values:
            raise RuntimeError("No proprioception history for camera timestamp")
        # Nearest sample must already have arrived; this never reads a future packet.
        value = min(values, key=lambda v: abs(v.capture_s - timestamp))
        if abs(value.capture_s - timestamp) > max_error_s:
            raise RuntimeError("Camera/proprioception timestamp mismatch")
        return value

    def publish(self, target, *, expected_hold_count=None):
        if not self.enabled:
            raise RuntimeError("Motor publication disabled in read-only mode")
        self.pd.check_measured(self.observation())  # fresh, CRC/profile-verified LowState
        if self._writer is None or not self._writer.is_alive():
            self.fault = "Native motor watchdog thread is not running"
            raise RuntimeError(self.fault)
        command = deepcopy(self._template)
        fill_lowcmd(
            command,
            indices=self.pd.indices,
            target=target,
            kp=self.pd.kp,
            kd=self.pd.kd,
            mode_machine=self.pd.mode_machine,
        )
        command.crc = self.crc.Crc(command)
        with self._lock:
            if self.operator_stop:
                raise KeyboardInterrupt(self.operator_stop_reason)
            if self.fault:
                raise RuntimeError(self.fault)
            if expected_hold_count is not None:
                self._latest.validate(time.monotonic())
                if (
                    self._command is not None
                    and time.monotonic() - self._command_time > self.max_command_age
                ):
                    self._enter_timing_hold_locked()
                if self.timing_hold_count != expected_hold_count:
                    raise StaleRobotObservation(
                        "Position hold changed during inference; discard the pending proposal"
                    )
            # The caller may acknowledge previous-issued action only after the
            # actual SDK write succeeds, not merely after queuing a new value.
            self._actuation_attempted = True
            self._write_checked(command)
            self._command = command
            self._command_time = time.monotonic()
            self.zero_torque_active = False
            self.timing_hold = False
            self._timing_hold_target = None

    def _enter_timing_hold_locked(self):
        """Capture one measured pose, not a continuously moving target."""
        if self.timing_hold:
            return
        if self._latest is None:
            raise RuntimeError("No validated pose available for position hold")
        self._timing_hold_target = self._latest.q.copy()
        command = deepcopy(self._template)
        fill_lowcmd(
            command,
            indices=self.pd.indices,
            target=self._timing_hold_target,
            kp=self.pd.kp,
            kd=self.pd.kd,
            mode_machine=self.pd.mode_machine,
        )
        command.crc = self.crc.Crc(command)
        self._command = command
        self.timing_hold = True
        self.timing_hold_count += 1

    def hold_for_timing(self):
        """Maintain position-PD support without fresh network feedback.

        Joint servos still use their onboard feedback. This is not a balance
        controller and does not guarantee that a moving robot will remain up.
        """
        if not self.enabled:
            raise RuntimeError("Motor publication disabled in read-only mode")
        with self._lock:
            if self.operator_stop:
                raise KeyboardInterrupt(self.operator_stop_reason)
            if self.fault:
                raise RuntimeError(self.fault)
            self._enter_timing_hold_locked()
            self._actuation_attempted = True
            self._write_checked(self._command)
            return self._timing_hold_target.copy()

    def _write_checked(self, command):
        try:
            # Current official ChannelPublisher.Write returns a Boolean;
            # success means accepted by DDS, not a physical actuator ACK.
            if self.publisher.Write(command) is not True:
                raise RuntimeError("SDK LowCmd Write returned failure")
        except Exception as exc:
            self.last_transport_error = str(exc)
            self.fault = self.fault or f"Native LowCmd transport failed: {exc}"
            raise RuntimeError(self.fault) from exc

    def _rewrite(self):
        while not self._stop.wait(0.002):
            try:
                with self._lock:
                    if self._command is None:
                        continue
                    if self.zero_torque_active:
                        # Waiting is intentional: age/stop must not restore PD gains.
                        pass
                    elif self.fault or self.operator_stop:
                        self.fault = self.fault or (
                            self.operator_stop_reason
                            if self.operator_stop
                            else "Native command watchdog expired"
                        )
                        fill_lowcmd(
                            self._command,
                            indices=self.pd.indices,
                            target=np.zeros(29),
                            kp=self.pd.kp,
                            kd=self.pd.kd,
                            mode_machine=self.pd.mode_machine,
                            damping=True,
                        )
                        self._command.crc = self.crc.Crc(self._command)
                    elif time.monotonic() - self._command_time > self.max_command_age:
                        self._enter_timing_hold_locked()
                    self._write_checked(self._command)
            except Exception as exc:
                # Keep the fault latched; retry zero torque while waiting or
                # damping after PD starts, rather than accepting new policy ACKs.
                self.last_transport_error = str(exc)
                self.fault = self.fault or f"Native watchdog rewrite failed: {exc}"

    def close(self):
        # Preserve zero torque when cancelled before START/A; active PD ends in damping.
        if (
            self.enabled
            and not self.zero_torque_active
            and (self._command is not None or self._actuation_attempted)
        ):
            self.damp("Native recovery session closed")
        self._stop.set()
        if self._writer:
            self._writer.join(timeout=1.0)

    def damp(self, reason):
        self.fault = self.fault or str(reason)
        if not self.enabled:
            return
        with self._lock:
            command = deepcopy(self._template)
            fill_lowcmd(
                command,
                indices=self.pd.indices,
                target=np.zeros(29),
                kp=self.pd.kp,
                kd=self.pd.kd,
                mode_machine=self.pd.mode_machine,
                damping=True,
            )
            command.crc = self.crc.Crc(command)
            self.zero_torque_active = False
            self._command = command
            try:
                self._write_checked(command)
            except RuntimeError:
                return False  # best effort, retain the original fault
        return True
