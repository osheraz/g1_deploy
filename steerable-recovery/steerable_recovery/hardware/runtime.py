"""Run native behavioral recovery against real sensors; motor output is opt-in.

python -m steerable_recovery.hardware.runtime --config deployment.json
The default uses real sensors but constructs no motor publisher. No robot is
contacted until the command is explicitly run, checkpoint contracts are loaded,
and calibration/clock inputs are provided.
"""

from __future__ import annotations

from motion_diffusion.inference.policy import load_controller

import argparse
import json
from pathlib import Path
import socket
import select
import time
import threading

import numpy as np
from humanoidverse2.observations.streams import CONTROL_OBSERVATION_WARN_S, StaleRobotObservation


class PelvisOdometryReceiver:
    """Local bridge from an external qualified estimator; no inertial invention.

    UDP JSON packets must carry actual mapped capture timestamps and pelvis
    position in the same gravity-aligned inertial axes as the pelvis quaternion.
    The estimator may run separately; LiDAR is not a planner/tracker requirement.
    """

    def __init__(self, port, *, background=False):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", int(port)))
        self.socket.setblocking(False)
        self.latest = None
        self.packet = None
        self.received_s = None
        self.error = None
        self.stream_id = None
        self.last_stamp = None
        self.stream_changed = False
        self._snapshot = (None, None, None, None)
        self._stop = threading.Event()
        self._thread = None
        if background:
            self._thread = threading.Thread(target=self._run, daemon=True, name="pelvis-odometry")
            self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            # Wake on packets instead of polling 500 times/s and adding up to
            # 2 ms of avoidable delivery delay. Timeout only bounds shutdown.
            ready, _, _ = select.select([self.socket], [], [], 0.05)
            if ready:
                self._receive()

    def _receive(self):
        # Bound work on the control thread even if a producer floods the socket.
        for _ in range(32):
            try:
                data, _ = self.socket.recvfrom(8192)
            except BlockingIOError:
                break
            self.received_s = time.monotonic()
            try:
                packet = json.loads(data)
                if (
                    packet.get("schema") != "hv2.pelvis_odometry.v1"
                    or packet.get("frame") != "pelvis_imu_gravity_frame"
                    or not packet.get("estimator")
                ):
                    raise ValueError(
                        "Odometry requires a declared estimator and calibrated pelvis/IMU common frame"
                    )
                stream = packet.get("stream_id")
                if self.stream_id is not None and stream != self.stream_id:
                    self.stream_changed = True
                self.stream_id = stream
                if self.stream_changed:
                    raise ValueError(
                        "Odometry estimator restarted; episode origin is no longer valid"
                    )
                stamp = packet.get("capture_monotonic_s")
                if stamp is not None and np.isfinite(float(stamp)):
                    if (
                        self.last_stamp is not None
                        and float(stamp) <= self.last_stamp
                        and (
                            packet.get("tracking_valid") is True
                            or packet.get("geometry_valid") is True
                        )
                    ):
                        raise ValueError("Odometry clock reset/out-of-order")
                    self.last_stamp = max(
                        float(stamp),
                        self.last_stamp if self.last_stamp is not None else float(stamp),
                    )
                self.packet = packet
                self.error = None
            except (ValueError, TypeError, AttributeError) as exc:
                self.packet = None
                self.error = str(exc)
            self._snapshot = (self.packet, self.received_s, self.error, self.stream_id)

    def sample(self, now_s=None, *, required=False, max_age_s=0.05, check_age=True):
        """Return diagnostics without using unqualified measurements as feedback.

        Required mode retains the existing clock/validity failure behavior. A
        monitoring-only deployment can inspect finite estimates and their status.
        """
        if self._thread is None:
            self._receive()
        raw, received, error, stream_id = self._snapshot
        # Live callers read the clock AFTER selecting the background snapshot.
        # A packet can arrive after a caller's earlier timestamp; comparing it
        # to that timestamp incorrectly labels its local receipt as future.
        # Explicit times remain available for deterministic/offline checks.
        if now_s is None:
            now_s = time.monotonic()
        packet = raw or {}
        self.latest = None
        try:
            position = np.asarray(packet.get("position_m"), dtype=float)
        except (ValueError, TypeError):
            position = np.asarray([])
        stamp = packet.get("capture_monotonic_s")
        stamp = float(stamp) if isinstance(stamp, (int, float)) and np.isfinite(stamp) else None
        finite = position.shape == (3,) and np.isfinite(position).all() and stamp is not None
        uncertainty = packet.get("clock_uncertainty_s")
        uncertainty = (
            float(uncertainty)
            if isinstance(uncertainty, (int, float)) and np.isfinite(uncertainty)
            else None
        )
        qualified = (
            packet.get("tracking_valid") is True
            and uncertainty is not None
            and 0 <= uncertainty <= 0.005
        )
        age = None if stamp is None else float(now_s - stamp)
        delay = packet.get("position_filter_delay_s", 0.0)
        valid_delay = isinstance(delay, (int, float)) and np.isfinite(delay) and delay >= 0
        if not valid_delay:
            error = error or "invalid_position_filter_delay"
        effective_stamp = None if stamp is None or not valid_delay else stamp - float(delay)
        effective_age = None if effective_stamp is None else float(now_s - effective_stamp)
        geometry = packet.get("geometry_valid", packet.get("tracking_valid", False)) is True
        fresh = (
            finite
            and valid_delay
            and (not check_age or (0 <= age and 0 <= effective_age <= max_age_s))
        )
        reason = error or packet.get("invalid_reason") or ""
        if not raw:
            reason = reason or "waiting_for_odometry"
        elif not finite:
            reason = "invalid_pose"
        elif not fresh:
            reason = "stale_or_future_odometry"
        elif not geometry:
            reason = reason or "geometry_invalid"
        elif not qualified:
            reason = reason or "clock_offset_unqualified"
        result = dict(
            position_m=position.tolist() if finite else None,
            capture_s=effective_stamp,
            age_s=effective_age,
            measurement_capture_s=stamp,
            measurement_age_s=age,
            position_filter_delay_s=delay if valid_delay else None,
            received_monotonic_s=received,
            arrival_age_s=None if received is None else float(now_s - received),
            clock_uncertainty_s=uncertainty,
            age_is_estimated=uncertainty is None,
            geometry_valid=geometry,
            tracking_valid=bool(qualified and fresh and geometry and not reason),
            available=bool(fresh and geometry and not error),
            invalid_reason=reason or None,
            estimator=packet.get("estimator"),
            stream_id=stream_id,
            used_for_steering=False,
        )
        for key in (
            "tilt_disagreement_rad",
            "yaw_disagreement_rad",
            "state_pair_error_s",
            "processing_s",
            "arrival_monotonic_s",
            "lidar_stamp_s",
            "clock_source",
            "orientation_disagreement_rad",
            "position_raw_m",
            "position_filter",
            "position_filter_tau_s",
        ):
            result[key] = packet.get(key)
        if required:
            if error:
                raise ValueError(error)
            if not raw:
                raise RuntimeError("Waiting for qualified pelvis odometry")
            if raw and not qualified:
                raise RuntimeError("Pelvis estimator tracking/clock qualification lost")
            if raw and not finite:
                raise ValueError("Invalid pelvis odometry")
            if raw and (not fresh or not geometry):
                raise RuntimeError("Pelvis odometry is stale or geometrically invalid")
            if reason:
                raise RuntimeError("Pelvis odometry invalid: " + str(reason))
        if result["tracking_valid"]:
            self.latest = (position.astype(np.float32), effective_stamp)
            result["used_for_steering"] = bool(required)
        return result

    def poll(self):
        self.sample(required=True, check_age=False)
        return self.latest

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.socket.close()


from motion_diffusion.inference.standing import load_standing_reference


class LiveStartupPreview:
    """Repeated real first-plan checks, never fictitious executed-action history.

    Each completed proposal starts a fresh preview. Advancing an unexecuted
    recovery for seconds against a stationary robot is not a closed-loop test.
    """

    def __init__(self, runtime, robot, pd, buffer, camera, telemetry, odom=None, depth_health=None):
        if pd is None:
            raise ValueError("Live preflight requires the configured PD and measured-state limits")
        self.runtime, self.robot, self.pd = runtime, robot, pd
        self.odom = odom
        runtime.preview = True
        self.buffer, self.camera, self.telemetry = buffer, camera, telemetry
        from steerable_recovery.hardware.camera import DepthHealth

        self.depth_health = depth_health or (
            DepthHealth(camera, buffer, telemetry) if camera else None
        )
        self.restart = True
        self.good = 0
        self.last_good = -float("inf")
        self.announced = False
        self.checks = 0
        self.failures = 0
        self.max_work_s = 0.0
        self.rejected_checks = 0
        self.blocked_reason = None
        self.waiting_for_depth = False
        self.last_depth_wait_message = -float("inf")
        self.last_steering_notice = None
        # Readiness certifies recent successful previews; the preview plan is
        # never executed at A. Keep that certificate through one allowed planner
        # reply interval plus a control tick, rather than expiring mid-inference.
        self.ready_valid_s = max(0.1, (runtime.queue.max_plan_age_ticks + 1) * 0.02)

    def reset_after_pose(self):
        """A pre-positioning preview cannot qualify the new joint pose."""
        self.restart = True
        self.good = 0
        self.last_good = -float("inf")
        self.announced = False
        self.blocked_reason = None

    @property
    def readiness_reason(self):
        if self.waiting_for_depth:
            return "waiting for delivered depth history"
        if self.blocked_reason:
            return self.blocked_reason
        if self.good < 5:
            return f"checking live plans ({self.good}/5 valid)"
        if time.monotonic() - self.last_good >= self.ready_valid_s:
            return "waiting for a fresh live plan"
        return "ready"

    def _ready(self, now):
        return self.good >= 5 and now - self.last_good < self.ready_valid_s

    def __call__(self):
        now = time.monotonic()
        if self.depth_health and not self.depth_health.check():
            self.waiting_for_depth = True
            self.restart = True
            self.good = 0
            return False
        obs = self.robot.observation()
        if self.buffer is not None:
            configs = [
                c
                for c in (self.runtime.controller.depth_config, self.runtime.info["depth_config"])
                if c.get("enabled")
            ]
            if not self.buffer.views_ready(now, configs):
                if now - self.last_depth_wait_message >= 2.0:
                    print(
                        "Waiting for fresh paired depth and the initial camera history span...",
                        flush=True,
                    )
                    self.last_depth_wait_message = now
                self.waiting_for_depth = True
                self.restart = True
                self.good = 0
                return False
            self.waiting_for_depth = False
        if self.restart:
            # A scheduler reset does not cancel an in-flight process request.
            # Drain it before creating fresh history, so a late preview cannot
            # turn into a recovery rollout against an unmoving robot.
            if self.runtime.worker.inflight:
                if self.runtime.worker.poll() is None:
                    return self._ready(now)
            self.runtime.reset()
            self.runtime.start_s = now
            self.restart = False
        scheduled = self.runtime.start_s + self.runtime.tick * 0.02
        if now < scheduled:
            return self._ready(now)
        if now - scheduled > 0.01:
            self.restart = True
            self.good = 0
            self.failures += 1
            return False
        began = time.monotonic()
        packet = None
        position_reason = None
        if (
            self.odom
            and self.runtime.info["sampling_contract"]["guidance"]["name"] == "motion_command"
        ):
            packet = self.odom.sample(required=False)
            from motion_diffusion.inference.steering import position_feedback

            position, position_reason = position_feedback(
                packet, getattr(self.runtime, "steering_feedback", "qualified")
            )
        else:
            position = self.odom.poll() if self.odom else None
        now = time.monotonic()  # Sensor callbacks may have delivered a newer capture.
        result = self.runtime.step(
            obs,
            np.zeros(29, np.float32),
            np.zeros(self.runtime.controller.latent_dim, np.float32),
            now_s=now,
            depth_buffer=self.buffer,
            pelvis_position=None if position is None else position[0],
            position_capture_s=None if position is None else position[1],
            position_stream_id=None if packet is None else packet["stream_id"],
            position_invalid_reason=position_reason,
        )
        if (
            self.runtime.steering_status is not None
            and self.runtime.steering_status != self.last_steering_notice
        ):
            status = self.runtime.steering_status
            print(
                (
                    f"Steering feedback ready ({getattr(self.runtime,'steering_feedback','qualified')}); will recheck at A."
                    if status["active"]
                    else f"Steering feedback unavailable: {status['reason']}. A would start unguided recovery."
                ),
                flush=True,
            )
            self.last_steering_notice = dict(status)
        if result is not None:
            action = result["requested_action"][0].detach().cpu().numpy()
            raw = getattr(self.runtime.controller, "last_unclipped_action", None)
            raw_max = None if raw is None else float(raw.detach().abs().max().cpu())
            self.telemetry.emit(
                dict(
                    event="startup_proposal",
                    check=self.checks + 1,
                    control_tick=self.runtime.tick - 1,
                    q=obs.q.copy(),
                    dq=obs.dq.copy(),
                    gyro_body=obs.gyro_body.copy(),
                    gravity_body=obs.gravity_body.copy(),
                    quat_xyzw=obs.quat_xyzw.copy(),
                    history=self.runtime.tracker_history.flatten().cpu().numpy().copy(),
                    native_code=result["native_code"].detach().cpu().numpy().copy(),
                    execution=result["execution"],
                    bootstrap_codes=(
                        result["prediction"]["prediction"]["action"].detach().cpu().numpy().copy()
                        if self.checks == 0 or (np.abs(action) >= self.pd.action_clip).any()
                        else None
                    ),
                    action=action.copy(),
                    raw_action_abs_max=raw_max,
                    issued_request=None,
                ),
                high_rate=True,
            )
            self.checks += 1
            try:
                self.pd.command(action, obs)
            except RuntimeError as exc:
                # A rejected no-output preview is not a motor-session fault.
                # Stay responsive to START/SELECT and evaluate fresh startup
                # measurements; never mark this proposal ready for A.
                reason = str(exc)
                self.rejected_checks += 1
                self.good = 0
                self.last_good = -float("inf")
                self.restart = True
                self.announced = False
                if reason != self.blocked_reason:
                    print(
                        f"Policy preview rejected: {reason}. Waiting for valid startup output.",
                        flush=True,
                    )
                    if raw_max is not None and (np.abs(action) >= self.pd.action_clip).any():
                        print(
                            f"Decoder raw action maximum: {raw_max:.1f}; "
                            f"{int((np.abs(action)>=self.pd.action_clip).sum())}/29 joints saturated. "
                            "START remains available for optional joint positioning.",
                            flush=True,
                        )
                self.blocked_reason = reason
                self.telemetry.emit(
                    dict(
                        event="startup_rejected",
                        check=self.checks,
                        reason=reason,
                        issued_request=None,
                    ),
                    high_rate=True,
                )
                return False
            self.blocked_reason = None
            work = time.monotonic() - began
            self.max_work_s = max(self.max_work_s, work)
            timely = time.monotonic() - scheduled <= 0.02
            self.good = self.good + 1 if timely else 0
            self.failures += int(not timely)
            if timely:
                self.last_good = time.monotonic()
            self.telemetry.emit(
                dict(
                    event="startup_preview",
                    check=self.checks,
                    timely=timely,
                    work_ms=work * 1000,
                    action_abs_max=float(np.abs(action).max()),
                    proposed_request=action.copy(),
                    issued_request=None,
                ),
                high_rate=True,
            )
            self.restart = True
        elif self.runtime.tick > self.runtime.queue.max_plan_age_ticks:
            # Only test the first plan from reset history. Waiting until the
            # normal replan interval fills history with unexecuted recovery.
            self.restart = True
            self.good = 0
            self.failures += 1
        ready = self._ready(time.monotonic())
        if ready and not self.announced:
            print(
                "Live inference ready. A: start from here. START: optional joint positioning. SELECT: stop.",
                flush=True,
            )
            self.announced = True
        return ready


def run(
    config,
    *,
    enable_actuation=False,
    duration_s=30.0,
    telemetry_jsonl=None,
    release_controller=None,
    preflight_only=False,
    on_policy_start=None,
    before_policy_ready=None,
):
    if preflight_only and enable_actuation:
        raise ValueError("Preflight never enables motor output")
    from humanoidverse2.observations.streams import ClockMapping, CausalDepthBuffer
    from humanoidverse2.robots.g1.control import NativePDContract
    from steerable_recovery.hardware.unitree import UnitreeHG, wait_for_policy_start
    from motion_diffusion.inference.execution import PlannerProcess, AsyncNativeRecovery
    from steerable_recovery.hardware.camera import (
        RealSenseDepth,
        DepthHealth,
        deployment_capture_config,
    )
    from steerable_recovery.hardware.telemetry import DeploymentTelemetry, latency_summary

    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Positive finite deployment duration required")
    telemetry = DeploymentTelemetry(telemetry_jsonl or config.get("telemetry_jsonl"))
    telemetry.record_preview = not enable_actuation
    worker = robot = camera = odom = depth_recorder = model_trace = None
    count = 0
    issued = 0
    proposed_ticks = 0
    hold_commands = 0
    controller_released = False
    summary = None
    last_recorded_plan = None
    command_limiter = None
    shaped_commands = 0
    try:
        telemetry.emit(
            dict(
                event="loading",
                controller_checkpoint=config["controller_checkpoint"],
                planner_checkpoint=config["planner_checkpoint"],
                motor_output=bool(enable_actuation),
            )
        )
        controller = load_controller(config)
        standing_reference = load_standing_reference(config, controller)
        request = np.zeros(29, np.float32)
        code = np.zeros(controller.latent_dim, np.float32)
        trace_options = {}
        if config.get("model_trace_directory"):
            from motion_diffusion.inference.model_trace import ModelTraceWriter, engine_identity

            trace_root = Path(config["model_trace_directory"])
            model_trace = ModelTraceWriter(trace_root / "control")
            (model_trace.directory / "runtime.json").write_text(
                json.dumps(
                    dict(
                        engines=engine_identity(controller),
                        precision=controller.precision,
                        observation_clip=controller.observation_clip,
                        action_clip=controller.action_clip,
                    ),
                    indent=2,
                )
                + "\n"
            )
            trace_options["trace_directory"] = str(trace_root / "planner")
        worker = PlannerProcess(
            config["planner_checkpoint"],
            device=config.get("planner_device", "cuda"),
            options=config.get("planner_options", {}),
            engine_config=config.get("planner_engine"),
            **trace_options,
        )
        runtime = AsyncNativeRecovery(
            controller,
            worker,
            max_plan_age_ticks=config.get("max_plan_age_ticks", 2),
            standing_reference=standing_reference,
        )
        runtime.trace = model_trace
        runtime.steering_feedback = config.get("steering_feedback", "qualified")
        motion_command = worker.info["sampling_contract"]["guidance"]["name"] == "motion_command"
        last_steering_status = None
        odom_required = False
        episode_pose = None
        if config.get("pelvis_odometry_port"):
            from steerable_recovery.hardware.measured_pose import EpisodePose

            odom = PelvisOdometryReceiver(config["pelvis_odometry_port"], background=True)
            episode_pose = EpisodePose()
            odom_mode = config.get("pelvis_odometry_mode", "required")
            if odom_mode not in ("monitor", "required"):
                raise ValueError("Unknown pelvis_odometry_mode")
            odom_required = odom_mode == "required" or worker.info["sampling_contract"]["guidance"][
                "name"
            ] in ("height_curve", "pelvis_path")
            if motion_command:
                odom_required = False  # degrade guidance, never kill recovery on pose loss
        telemetry.emit(
            dict(event="warming", scope="disposable inference only; no sensor or motor connection")
        )
        timings = runtime.warmup()
        telemetry.emit(dict(event="warmup_details", timings=timings), high_rate=True)
        telemetry.emit(dict(event="warmup_complete"))
        configs = [
            v for v in (controller.depth_config, worker.info["depth_config"]) if v.get("enabled")
        ]
        capture = None
        if configs:
            if any(v["capture"] != configs[0]["capture"] for v in configs):
                raise ValueError(
                    "Tracker/planner physical camera contracts must match for a shared sensor"
                )
            capture = deployment_capture_config(
                configs[0]["capture"], config["camera"], enable_actuation=enable_actuation
            )
            telemetry.emit(
                dict(
                    event="camera_intrinsics",
                    trained=configs[0]["capture"]["base_intrinsics"],
                    physical=capture["base_intrinsics"],
                    changed=capture["base_intrinsics"] != configs[0]["capture"]["base_intrinsics"],
                    motor_output=bool(enable_actuation),
                )
            )
        # The normal launcher supplies a complete envelope, checked in dry runs too.
        pd = (
            NativePDContract(controller.contract, config["pd"])
            if config.get("pd", {}).get("lower_joint_limits") is not None
            else None
        )
        shaping = config.get("command_limits", {"mode": "training"})
        if shaping["mode"] not in ("training", "bounded"):
            raise ValueError("Unknown command limit mode")
        if shaping["mode"] == "bounded":
            from humanoidverse2.robots.g1.control import PDTargetLimiter

            command_limiter = PDTargetLimiter(**shaping["parameters"])
            for actual, expected in (
                (command_limiter.default, controller.default_joint_pos.cpu().numpy()),
                (command_limiter.scale, pd.scale),
                (shaping["parameters"]["kp"], pd.kp),
            ):
                if not np.array_equal(
                    np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
                ):
                    raise ValueError("Command shaping differs from the frozen PD contract")
        remote_start = config["robot"].get("require_remote_start", False)
        if config.get("start_pose") is not None:
            from motion_diffusion.inference.start_pose import validate_start_pose

            if not enable_actuation or not remote_start or pd is None:
                raise ValueError(
                    "Saved start pose requires operator-controlled actuation with a complete PD contract"
                )
            validate_start_pose(config["start_pose"], pd)
        if enable_actuation and pd is None:
            raise ValueError("Complete command limits required before actuation")
        if (
            enable_actuation
            and not remote_start
            and config.get("hardware_commissioning_receipt", {}).get("qualified") is not True
        ):
            raise ValueError(
                "Motor mode requires an explicit hardware commissioning receipt; existing simulation results are insufficient"
            )
        robot_config = config["robot"]
        robot = UnitreeHG(
            interface=robot_config["interface"],
            dof_names=controller.contract["dof_names"],
            capture_clock=(
                ClockMapping(**robot_config["tick_clock"])
                if robot_config.get("tick_clock")
                else None
            ),
            tick_period_s=robot_config["tick_period_s"],
            enable_actuation=False,
            pd_contract=pd,
            max_command_age_s=robot_config.get("max_command_age_s", 0.06),
            domain_id=robot_config.get("domain_id", 0),
        )
        deadline = time.monotonic() + 5.0
        while True:
            try:
                robot.observation()
                break
            except RuntimeError:
                if robot.fault or time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        if pd is not None:
            pd.check_startup_pose(robot.observation(), block=enable_actuation)
        if robot_config.get("tick_clock") is None:
            telemetry.emit(dict(event="robot_clock_aligned", **robot.tick_alignment.report))
        if enable_actuation and config.get("startup_zero_torque", False):
            robot.arm_a_stop(after_press=float("inf"))  # SELECT cancels; discard early A/START.
            if release_controller is not None:
                release_controller()
                controller_released = True
            robot.enter_zero_torque()
            telemetry.emit(
                dict(
                    event="startup_zero_torque",
                    tau=0.0,
                    kp=0.0,
                    kd=0.0,
                    until="operator START/A",
                    rewrite_hz=500,
                )
            )
        buffer = None
        depth_health = None
        if configs:
            # The model-facing history retains its saved contract. Only the
            # physical camera validates/samples with the explicitly selected lens.
            buffer = CausalDepthBuffer(configs[0]["capture"], controller.default_joint_pos.cpu())
            buffer.record_references = bool(
                model_trace is not None and config.get("depth_recording_directory")
            )
            camera_config = config["camera"]
            receipt = (
                json.loads(Path(camera_config["calibration_receipt"]).read_text())
                if camera_config.get("calibration_receipt")
                else None
            )
            source = camera_config.get("source", "realsense")
            camera_class = RealSenseDepth
            extra = {}
            if config.get("depth_recording_directory"):
                from steerable_recovery.hardware.depth_recording import DepthFrameRecorder

                depth_recorder = DepthFrameRecorder(
                    config["depth_recording_directory"],
                    capture,
                    storage="native_png",
                    start_paused=not preflight_only,
                )
                extra["frame_recorder"] = depth_recorder
            if source == "network":
                from steerable_recovery.hardware.network_depth import NetworkDepth

                camera_class = NetworkDepth
                extra["endpoint"] = camera_config["endpoint"]
                if camera_config.get("processing") is not None:
                    extra["processing"] = camera_config["processing"]
                if camera_config.get("clock_endpoint"):
                    extra.update(
                        clock_endpoint=camera_config["clock_endpoint"],
                        mount_confirmed=camera_config["mount_confirmed"],
                    )
            elif source != "realsense":
                raise ValueError("Camera source must be realsense or network")
            camera = camera_class(
                serial=camera_config["serial"],
                buffer=buffer,
                robot=robot,
                calibration_receipt=receipt,
                capture_config=capture,
                hardware_clock=(
                    ClockMapping(**camera_config["hardware_clock"])
                    if camera_config.get("hardware_clock")
                    else None
                ),
                **extra,
            )
            depth_health = DepthHealth(camera, buffer, telemetry)
            if camera_config.get("clock_endpoint"):
                deadline = time.monotonic() + 5.0
                while camera.paired_frames < 3:
                    depth_health.check()
                    robot.observation()
                    if time.monotonic() > deadline:
                        raise RuntimeError("Camera frames are not pairing with robot state")
                    time.sleep(0.01)
                telemetry.emit(
                    dict(
                        event="camera_clock_aligned",
                        **camera.clock_report,
                        paired_frames=camera.paired_frames,
                        pair_error_s=camera.last_capture_timing["pair_error_s"],
                    )
                )
        if before_policy_ready is not None:
            if enable_actuation and remote_start:
                # SELECT remains responsive. A/START during warmup are discarded;
                # wait_for_policy_start captures fresh button baselines afterward.
                robot.arm_a_stop(after_press=float("inf"))

            def check_startup_sensors():
                robot.observation()
                if depth_health is not None:
                    depth_health.check()

            before_policy_ready(check_startup_sensors)
        if enable_actuation and not remote_start:
            robot.enable_motor_output()
        telemetry.emit(
            dict(
                event="session_contract",
                mode="actuation" if enable_actuation else "sensor_dry_run",
                tracker_sha256=controller.checkpoint_sha256,
                planner_contract=worker.info["contract"],
                sampling=worker.info["sampling_contract"],
                deployment_config=config,
                timing_policy=dict(
                    warn_age_s=CONTROL_OBSERVATION_WARN_S,
                    command_gap_hold_s=robot_config.get("max_command_age_s", 0.06),
                    timing_shutdown=False,
                    delay_behavior="measured-pose PD hold; discard obsolete plans; resume with fresh feedback and plans",
                ),
                depth_loss_policy="warn; mask expired or untrusted captures; continue policy; resume valid paired depth",
                telemetry_contract=dict(
                    schema="hv2.native_recovery.telemetry.v1",
                    measured_state="raw native-order measured q/dq/pelvisIMU; no simulator truth",
                    proposed_vs_issued="dry-run proposals are never reported as executed",
                    plans="unblended S/Z plus original measured anchor, origin tick and generation",
                    depth=(
                        "original-resolution lossless depth archived asynchronously; model/video views reconstructed"
                        if depth_recorder
                        else "metadata/timing only"
                    ),
                ),
                depth_recording_directory=config.get("depth_recording_directory"),
            ),
            high_rate=True,
        )
        telemetry.emit(
            dict(
                event="ready",
                mode="actuation" if enable_actuation else "sensor_dry_run",
                control_hz=50,
                replan_ticks=worker.info.get("replan_every_ticks"),
                camera_paired_frames=(
                    (None if camera is None else camera.paired_frames)
                    if config.get("camera", {}).get("clock_endpoint")
                    else None
                ),
            )
        )
        if not telemetry.barrier():
            raise RuntimeError("Deployment startup status could not be recorded")
        preview = (
            LiveStartupPreview(
                runtime,
                robot,
                pd,
                buffer,
                camera,
                telemetry,
                odom if odom_required or motion_command else None,
                depth_health=depth_health,
            )
            if ((enable_actuation and remote_start) or preflight_only)
            else None
        )
        if preflight_only:
            until = time.monotonic() + duration_s
            ready = False
            while time.monotonic() < until:
                ready = preview()
                time.sleep(0.001)
            if not ready:
                raise RuntimeError("Live preflight did not establish startup readiness")
            return dict(
                ticks=preview.checks,
                commands_issued=0,
                mode="preflight",
                startup_checks=preview.checks,
                startup_timing_failures=preview.failures,
                startup_rejected_checks=preview.rejected_checks,
                max_startup_work_ms=preview.max_work_s * 1000,
            )
        hold_target = None

        def activate():
            nonlocal hold_target, hold_commands
            pd.check_startup_pose(robot.observation())  # before service release or motor output
            if release_controller is not None and not controller_released:
                release_controller()
            takeover_observation = robot.observation()  # release can take several seconds
            pd.check_startup_pose(takeover_observation)
            robot.enable_motor_output()
            obs = robot.observation()
            hold_target = obs.q.copy()
            robot.publish(pd.command((hold_target - pd.default) / pd.scale, obs))
            hold_commands += 1
            print(
                "Holding the current pose during planner handover; not moving to default.",
                flush=True,
            )

        if enable_actuation and remote_start:
            pose_options = {}
            if config.get("start_pose") is not None:
                pose_options = dict(
                    start_pose=config["start_pose"],
                    reset_preview=preview.reset_after_pose,
                    pose_event=lambda row: telemetry.emit(row, high_rate=True),
                )
            wait_for_policy_start(robot, pd, preview=preview, activate=activate, **pose_options)
            if on_policy_start is not None:
                on_policy_start()
            runtime.preview = False
            runtime.reset()  # preview proposals were never executed
            obs = robot.observation()
            hold_target = obs.q.copy()
            request = (hold_target - pd.default) / pd.scale
            robot.publish(pd.command(request, obs))
            hold_commands += 1
            if depth_health:
                depth_health.check(policy_active=True)
        elif on_policy_start is not None:
            on_policy_start()  # sensor-only trial: start capture after sensors are ready
        if depth_recorder is not None:
            depth_recorder.start()
        # Logs/camera startup may block arbitrarily. Start the control clock
        # only after them, and use exactly one origin for scheduler and policy.
        start = time.monotonic()
        runtime.start_s = start
        # Consume pre-A presses. Holding Y through startup must not trigger a
        # delayed mode change when the first policy command becomes available.
        standing_presses = getattr(robot, "standing_hold_requests", 0)
        if standing_reference is not None:
            print(
                "Y: encoded standing hold after the robot settles. A / SELECT: stop. Y stays latched until this run ends.",
                flush=True,
            )
        last_standing_status = None
        # A does not reset physics or the camera. Keep actual pre-start captures.
        violations = {}
        checked_proposals = 0
        control_times = []
        publication_late_ticks = 0
        timing_warning_ticks = 0
        last_timing_warning = -float("inf")
        timing_hold_ticks = 0
        hold_active = False
        schedule_shift = 0.0
        command_gap_s = robot_config.get("max_command_age_s", 0.06)

        def maintain_support(reason, *, invalidate=True, force=False):
            nonlocal hold_target, request, code, hold_commands, hold_active, timing_hold_ticks, last_timing_warning
            if invalidate and (not hold_active or force):
                runtime.invalidate_for_timing()
            hold_active = True
            timing_hold_ticks += 1
            # A timing hold issued a measured-pose target. Start the next
            # bounded transition from that pose, not an obsolete policy target.
            if command_limiter is not None:
                command_limiter.reset()
            if enable_actuation:
                hold_target = robot.hold_for_timing()
                request = (hold_target - pd.default) / pd.scale
                hold_commands += 1
            code = np.zeros_like(code)
            stamp = time.monotonic()
            console = stamp - last_timing_warning >= 1.0
            telemetry.emit(
                dict(
                    event="timing_hold",
                    tick=count,
                    reason=reason,
                    commands_issued=issued,
                    motor_output=enable_actuation,
                    held_joint_target=None if not enable_actuation else hold_target.copy(),
                    issued_request=None if not enable_actuation else request.copy(),
                ),
                high_rate=not console,
            )
            if console:
                last_timing_warning = stamp

        sensor_to_proposal = []
        sensor_to_publish = []
        arrival_to_proposal = []
        while time.monotonic() - start < float(duration_s):
            target = start + schedule_shift + count * 0.02
            remaining = target - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            now = time.monotonic()
            if now >= start + float(duration_s):
                break
            # Resume at the current time rather than issuing a burst of old
            # scheduled ticks. Episode elapsed time and steering origin persist.
            if now - target > 0.02:
                schedule_shift += now - target
                target = now
            if enable_actuation and getattr(robot, "timing_hold", False) and not hold_active:
                maintain_support("Motor writer entered position hold during a command gap")
            if depth_health:
                depth_health.check(policy_active=True)
            odom_start = time.monotonic()
            standing_active = (
                standing_reference is not None and runtime.standing_hold_status is not None
            )
            odometry = odom.sample(required=odom_required and not standing_active) if odom else None
            odom_poll_s = time.monotonic() - odom_start
            position = odom.latest if odom and odom_required else None
            position_reason = None
            if motion_command:
                from motion_diffusion.inference.steering import position_feedback

                position, position_reason = position_feedback(odometry, runtime.steering_feedback)
            try:
                observation = robot.observation()
                now = time.monotonic()  # Receipt can occur during other sensor polling.
                observation.validate(now)
            except StaleRobotObservation as exc:
                maintain_support(str(exc))
                count += 1
                continue
            previous_request = request
            previous_code = code
            previous_code_available = issued > 0 and runtime.started
            hold_generation = getattr(robot, "timing_hold_count", 0)
            step_start = time.monotonic()
            runtime.trace_control_tick = count
            presses = getattr(robot, "standing_hold_requests", 0)
            standing_requested = enable_actuation and presses > standing_presses
            standing_presses = presses
            result = runtime.step(
                observation,
                request,
                code,
                now_s=now,
                depth_buffer=buffer,
                pelvis_position=None if position is None else position[0],
                position_capture_s=None if position is None else position[1],
                position_stream_id=None if odometry is None else odometry["stream_id"],
                position_invalid_reason=position_reason,
                standing_hold_requested=standing_requested,
            )
            if standing_requested:
                accepted = (
                    standing_reference is not None and runtime.standing_hold_status is not None
                )
                telemetry.emit(dict(event="standing_hold_requested", tick=count, accepted=accepted))
                if not accepted:
                    print(
                        "Y ignored: "
                        + (
                            "launch with --standing-hold to enable encoded standing intent."
                            if standing_reference is None
                            else "policy is not issuing normally yet; press again after recovery."
                        ),
                        flush=True,
                    )
            if (
                standing_reference is not None
                and runtime.standing_hold_status != last_standing_status
            ):
                last_standing_status = dict(runtime.standing_hold_status)
                telemetry.emit(dict(event="standing_hold_activated", **last_standing_status))
                print(
                    f"Y accepted: switching to encoded standing intent over {standing_reference.blend_s:g} s. Planner bypassed; live tracker continues.",
                    flush=True,
                )
            if motion_command:
                status = runtime.steering_status
                if status != last_steering_status:
                    telemetry.emit(dict(event="steering_status", **status))
                    print(
                        (
                            f"Steering active: {runtime.steering_feedback} measured feedback."
                            if status["active"]
                            else (
                                "Steering bypassed by manual encoded standing hold."
                                if status["reason"] == "standing_hold"
                                else (
                                    "Upright pose confirmed; switching to unguided recovery for the rest of this run."
                                    if status["reason"] == "upright_reached"
                                    else (
                                        "Steering time window ended; continuing unguided recovery."
                                        if status["reason"] == "command_window_elapsed"
                                        else (
                                            "Requested side roll confirmed; continuing unguided recovery."
                                            if status["reason"] == "turn_completed"
                                            else (
                                                f"Steering paused: {status['reason']}; recovery continues, waiting for fresh feedback in the same frame."
                                                if status.get("paused")
                                                else f"Steering disabled for this episode: {status['reason']}. Continuing unguided recovery."
                                            )
                                        )
                                    )
                                )
                            )
                        ),
                        flush=True,
                    )
                    last_steering_status = dict(status)
                if odometry is not None:
                    odometry["used_for_steering"] = status["active"]
                    odometry["steering_feedback_mode"] = runtime.steering_feedback
            proposal_ready_s = command_publish_s = None
            command_shaping = None
            if result is not None:
                proposed = result["requested_action"][0].cpu().numpy()
                proposal_ready_s = time.monotonic()
                sensor_to_proposal.append(proposal_ready_s - observation.capture_s)
                arrival_to_proposal.append(proposal_ready_s - observation.arrival_s)
                proposed_ticks += 1
                if time.monotonic() - target > 0.02:
                    publication_late_ticks += 1
                if not enable_actuation and pd is not None:
                    try:
                        pd.command(proposed, observation)
                        checked_proposals += 1
                    except RuntimeError as exc:
                        reason = str(exc).partition(":")[0]
                        violations[reason] = violations.get(reason, 0) + 1
                if enable_actuation:
                    if proposal_ready_s - step_start > command_gap_s or (
                        getattr(robot, "timing_hold", False) and not hold_active
                    ):
                        maintain_support(
                            "Inference completed after a position-hold transition; discarding delayed proposal",
                            force=True,
                        )
                        count += 1
                        continue
                    try:
                        candidate = proposed
                        if command_limiter is not None:
                            candidate, _, command_shaping = command_limiter.apply(
                                proposed, observation.q, elapsed_s=time.monotonic() - start
                            )
                        target_q = pd.command(candidate, observation)
                        observation.validate(time.monotonic(), max_age_s=None)
                        robot.publish(target_q, expected_hold_count=hold_generation)
                    except StaleRobotObservation as exc:
                        maintain_support(str(exc), force=True)
                        count += 1
                        continue
                    except RuntimeError as exc:
                        telemetry.emit(
                            dict(
                                event="command_rejected",
                                tick=count,
                                reason=str(exc),
                                elapsed_ms=(time.monotonic() - target) * 1000,
                                control_work_ms=(time.monotonic() - step_start) * 1000,
                                q=observation.q.copy(),
                                dq=observation.dq.copy(),
                                proposed_request=proposed.copy(),
                                target_joint_pos=pd.default + pd.scale * proposed,
                                proposed_native_code=result["native_code"][0]
                                .detach()
                                .cpu()
                                .numpy()
                                .copy(),
                                issued_request=None,
                            ),
                            high_rate=True,
                        )
                        raise
                    command_publish_s = time.monotonic()
                    sensor_to_publish.append(command_publish_s - observation.capture_s)
                    request = candidate.copy()
                    code = result["native_code"][0].cpu().numpy().copy()
                    if (
                        command_shaping is not None
                        and (
                            command_shaping["position_limited"] | command_shaping["rate_limited"]
                        ).any()
                    ):
                        shaped_commands += 1
                        if shaped_commands == 1:
                            changed = np.flatnonzero(
                                command_shaping["position_limited"]
                                | command_shaping["rate_limited"]
                            )
                            telemetry.emit(
                                dict(
                                    event="command_shaping_started",
                                    tick=count,
                                    joints=[pd.dof_names[i] for i in changed],
                                    requested_target=command_shaping["requested_target"],
                                    issued_target=target_q,
                                    reason="Bounded PD target transition; policy continues",
                                ),
                                high_rate=True,
                            )
                    runtime.acknowledge()
                    issued += 1
                    if model_trace is not None:
                        runtime.flush_trace(
                            outcome="issued",
                            issued_request=request,
                            issued_target=target_q,
                            command_shaping=command_shaping,
                            command_publish_s=command_publish_s,
                        )
                    if hold_active:
                        telemetry.emit(
                            dict(event="timing_resumed", tick=count, commands_issued=issued)
                        )
                    hold_active = False
                elif model_trace is not None:
                    runtime.flush_trace(outcome="sensor_only", issued_request=None)
                age = proposal_ready_s - observation.capture_s
                lag = proposal_ready_s - target
                if age > CONTROL_OBSERVATION_WARN_S or lag > CONTROL_OBSERVATION_WARN_S:
                    timing_warning_ticks += 1
                    # Journal every delayed tick, throttle console output. The
                    # writer thread must never wait for formatting or stdout.
                    stamp = time.monotonic()
                    console = stamp - last_timing_warning >= 1.0
                    telemetry.emit(
                        dict(
                            event="timing_warning",
                            tick=count,
                            capture_age_ms=age * 1000,
                            receipt_age_ms=(proposal_ready_s - observation.arrival_s) * 1000,
                            scheduled_lag_ms=lag * 1000,
                            commands_issued=issued,
                            motor_command_published=command_publish_s is not None,
                        ),
                        high_rate=not console,
                    )
                    if console:
                        last_timing_warning = stamp
                # In dry-run the proposed command was NOT executed: keep prior
                # requests/codes absent rather than inventing causal feedback.
                plan = result.get("prediction")
                if plan is not None:
                    identity = (plan["generation"], plan["origin_tick"])
                    if identity != last_recorded_plan:
                        telemetry.emit(
                            dict(
                                event="plan",
                                control_tick=count,
                                arrival_monotonic_s=time.monotonic(),
                                **plan,
                            ),
                            high_rate=True,
                        )
                        last_recorded_plan = identity
            elif enable_actuation and hold_target is not None:
                maintain_support("Waiting for a fresh executable plan", invalidate=False)
            if telemetry.path is not None:
                if odometry is not None:
                    odometry = episode_pose.update(
                        odometry, observation.quat_xyzw, float(runtime.heading[0])
                    )
                    odometry["poll_s"] = odom_poll_s
                # Copy small current arrays now. Prediction tensors came from
                # immutable CPU worker snapshots, never deferred CUDA reads.
                control_record = dict(
                    event="control",
                    tick=count,
                    monotonic_s=now,
                    observation_capture_s=observation.capture_s,
                    observation_arrival_s=observation.arrival_s,
                    proposal_ready_s=proposal_ready_s,
                    command_publish_s=command_publish_s,
                    observation_sequence=observation.sequence,
                    clock_uncertainty_s=observation.clock_uncertainty_s,
                    q=observation.q.copy(),
                    dq=observation.dq.copy(),
                    pelvis_quat_xyzw=observation.quat_xyzw.copy(),
                    gyro_body=observation.gyro_body.copy(),
                    gravity_body=observation.gravity_body.copy(),
                    causal_heading_rad=runtime.heading.detach().cpu().numpy().copy(),
                    previous_issued_request=previous_request.copy(),
                    previous_executed_code=previous_code.copy(),
                    previous_code_available=previous_code_available,
                    proposed_request=(None if result is None else proposed.copy()),
                    proposed_native_code=(
                        None
                        if result is None
                        else result["native_code"][0].detach().cpu().numpy().copy()
                    ),
                    issued_request=(
                        request.copy() if enable_actuation and result is not None else None
                    ),
                    executed_native_code=(
                        code.copy() if enable_actuation and result is not None else None
                    ),
                    execution=(None if result is None else result.get("execution")),
                    command_shaping=command_shaping,
                    control_mode=(
                        "standing_hold" if last_standing_status is not None else "recovery"
                    ),
                    steering_progress=(
                        runtime.progress_steering.diagnostics()
                        if getattr(runtime, "progress_steering", None) is not None
                        else None
                    ),
                    steering_status=runtime.steering_status if motion_command else None,
                    pelvis_odometry=odometry,
                    depth_timing=(None if camera is None else dict(camera.last_capture_timing)),
                    control_wall_s=time.monotonic() - step_start,
                    scheduled_lag_s=now - target,
                    rejected_replies=runtime.rejected_replies,
                )
                telemetry.emit(control_record, high_rate=True)
            if count % 50 == 0:
                telemetry.emit(
                    dict(
                        event="health",
                        tick=count,
                        commands_issued=issued,
                        proposed_control_ticks=proposed_ticks,
                        planner_bootstrapped=runtime.queue.valid[0].item(),
                        rejected_replies=runtime.rejected_replies,
                        last_sample_wall_s=(
                            None
                            if runtime.last_prediction is None
                            else runtime.last_prediction["sample_wall_s"]
                        ),
                        depth_timing=(None if camera is None else dict(camera.last_capture_timing)),
                        command_limit_violations=violations.copy(),
                        publication_late_ticks=publication_late_ticks,
                        timing_warning_ticks=timing_warning_ticks,
                        timing_hold_ticks=timing_hold_ticks,
                        shaped_commands=shaped_commands,
                        rejected_stale_robot_packets=getattr(robot, "rejected_stale", 0),
                        pelvis_odometry=odometry,
                        telemetry=telemetry.status(),
                    )
                )
            control_times.append(time.monotonic() - step_start)
            count += 1
        if not proposed_ticks:
            raise RuntimeError(
                "Session ended without any timely executable planner/tracker proposal; "
                f"{runtime.rejected_replies} planner replies rejected, zero commands issued"
            )
        next_cursor = runtime.tick - int(runtime.queue.origin_tick[0])
        summary = dict(
            ticks=count,
            commands_issued=issued,
            proposed_control_ticks=proposed_ticks,
            shaped_commands=shaped_commands,
            planner_bootstrapped=True,
            rejected_replies=runtime.rejected_replies,
            final_plan_valid_for_next_tick=bool(runtime.queue.valid[0])
            and 0 <= next_cursor < runtime.queue.horizon,
            inference_completed=True,
            checked_proposals=checked_proposals,
            command_limit_violations=violations,
            publication_late_ticks=publication_late_ticks,
            timing_warning_ticks=timing_warning_ticks,
            timing_hold_ticks=timing_hold_ticks,
            control_wall_p50_ms=float(np.median(control_times) * 1000) if control_times else None,
            control_wall_p95_ms=(
                float(np.percentile(control_times, 95) * 1000) if control_times else None
            ),
            control_wall_p99_ms=(
                float(np.percentile(control_times, 99) * 1000) if control_times else None
            ),
            latency=dict(
                sensor_capture_to_proposal=latency_summary(sensor_to_proposal),
                sensor_arrival_to_proposal=latency_summary(arrival_to_proposal),
                sensor_capture_to_dds_publish=latency_summary(sensor_to_publish),
                scope="Aligned robot-state capture to host proposal or DDS publish return; excludes robot receipt and motor response",
            ),
            completion_scope="Requested sensor/inference session completed; no recovery, steering, or robot-safety success claim",
            mode="actuation" if enable_actuation else "sensor_dry_run",
        )
    except KeyboardInterrupt as exc:
        telemetry.emit(
            dict(
                event="operator_stop",
                tick=count,
                reason=str(exc) or "keyboard interrupt",
                commands_issued=issued,
                proposed_control_ticks=proposed_ticks,
            )
        )
        raise
    except BaseException as exc:
        telemetry.emit(
            dict(
                event="fault",
                tick=count,
                error_type=type(exc).__name__,
                error=str(exc),
                commands_issued=issued,
                proposed_control_ticks=proposed_ticks,
                planner_bootstrapped=proposed_ticks > 0,
                rejected_replies=(
                    None
                    if worker is None or "runtime" not in locals()
                    else runtime.rejected_replies
                ),
            )
        )
        raise
    finally:
        # Stop movement first, before any camera or model cleanup can block.
        if robot:
            robot.close()
        if model_trace is not None and "runtime" in locals():
            runtime.flush_trace(outcome="stopped_before_issuance")
        if camera:
            camera.close()
        depth_status = depth_recorder.close() if depth_recorder else None
        if depth_status:
            telemetry.emit(dict(event="depth_recording", **depth_status), high_rate=True)
        if odom:
            odom.close()
        if worker:
            worker.close()
        if model_trace is not None:
            status = model_trace.close()
            try:
                planner_status = json.loads((trace_root / "planner/summary.json").read_text())
            except (OSError, ValueError) as exc:
                planner_status = dict(complete=False, error=str(exc))
            trace_status = dict(
                control=status,
                planner=planner_status,
                complete=status["complete"] and planner_status["complete"],
            )
            if depth_status is not None:
                trace_status["depth"] = depth_status
                trace_status["complete"] &= depth_status["complete"]
            try:
                (trace_root / "summary.json").write_text(json.dumps(trace_status, indent=2) + "\n")
            except OSError as exc:
                trace_status.update(complete=False, error=str(exc))
            print(
                (
                    "Model trace saved without drops: "
                    if trace_status["complete"]
                    else "Model trace INCOMPLETE; see summaries: "
                )
                + str(trace_root),
                flush=True,
            )
            if summary is not None:
                summary["model_trace"] = trace_status
        telemetry.emit(
            dict(
                event="stopped",
                ticks=count,
                commands_issued=issued,
                startup_hold_commands=hold_commands,
                proposed_control_ticks=proposed_ticks,
                planner_bootstrapped=proposed_ticks > 0,
            )
        )
        telemetry_status = telemetry.close()
        if summary is not None:
            summary["telemetry"] = telemetry_status
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--enable-actuation",
        action="store_true",
        help="Explicitly enable verified native29 PD motor output; default sends nothing",
    )
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument(
        "--telemetry-jsonl",
        help="Optional exclusive JSONL or JSONL.gz journal; bounded asynchronous writes, no raw depth images",
    )
    args = parser.parse_args()
    if not np.isfinite(args.duration_s) or args.duration_s <= 0:
        parser.error("Positive finite duration required")
    print(
        json.dumps(
            run(
                json.loads(Path(args.config).read_text()),
                enable_actuation=args.enable_actuation,
                duration_s=args.duration_s,
                telemetry_jsonl=args.telemetry_jsonl,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
