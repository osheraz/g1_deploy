"""Check live first-plan proposals before accepting the operator's A press."""

import time
import numpy as np


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
        from rise.sensing.camera import DepthHealth

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
