"""Bounded optional deployment journal; storage/stdout never block control ticks."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import queue
import threading


def latency_summary(seconds):
    """Host-clock samples; callers must name the measured endpoints."""
    import numpy as np

    values = np.asarray(seconds, dtype=float) * 1000
    if not len(values):
        return dict(count=0, p50_ms=None, p95_ms=None, p99_ms=None, max_ms=None)
    return dict(
        count=len(values),
        p50_ms=float(np.percentile(values, 50)),
        p95_ms=float(np.percentile(values, 95)),
        p99_ms=float(np.percentile(values, 99)),
        max_ms=float(values.max()),
    )


def _json_default(value):
    if hasattr(value, "tolist"):
        if hasattr(value, "device") and str(value.device) != "cpu":
            raise ValueError("Telemetry requires CPU snapshots; no deferred CUDA reads")
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported deployment telemetry value {type(value).__name__}")


def console_status(record):
    """Short operator messages; the complete records stay in the JSONL journal."""
    event = record["event"]
    if event == "loading":
        return "Loading planner and action decoder..."
    if event == "warming":
        return "Warming up inference..."
    if event == "warmup_complete":
        return "Inference warmup complete. Connecting robot and camera..."
    if event == "robot_clock_aligned":
        return "Robot state connected; timestamps aligned."
    if event == "startup_zero_torque":
        return "Zero torque active: tau=0, kp=0, kd=0. Waiting for START/A after sensor readiness."
    if event == "camera_clock_aligned":
        return "Depth camera connected; frames paired with robot state."
    if event == "camera_intrinsics":
        return (
            "Camera: using the configured lens calibration (different from this older model)."
            if record["changed"]
            else "Camera calibration matches the model."
        )
    if event == "ready":
        return (
            "Sensors ready. Waiting for operator controls."
            if record["mode"] == "actuation"
            else "Sensors ready. Running sensor-only inference; no motor commands."
        )
    if event == "health":
        sample = record["last_sample_wall_s"]
        planner = "waiting for first plan" if sample is None else f"planner {sample*1000:.0f} ms"
        limits = record["command_limit_violations"]
        result = (
            f'Running {record["tick"]/50:.0f}s | {planner} | '
            f'commands sent {record["commands_issued"]} | '
            f'plans rejected {record["rejected_replies"]} | '
            f'late control ticks {record["publication_late_ticks"]} | '
            f"command-limit failures {sum(limits.values())}"
        )
        if limits:
            result += " — " + "; ".join(limits)
        pose = record.get("pelvis_odometry")
        if pose is not None:
            xyz = pose.get("position_reset_m")
            result += (
                " | XYZ unavailable"
                if xyz is None
                else " | XYZ " + ",".join(f"{v:+.2f}" for v in xyz) + " m"
            )
            if not pose.get("origin_available", True):
                result += " (no pose at episode start)"
            if pose.get("age_s") is not None:
                result += f' | pose age {pose["age_s"]*1000:.0f} ms'
            if not pose.get("tracking_valid"):
                result += " [" + str(pose.get("invalid_reason") or "unqualified") + "]"
        journal = record["telemetry"]
        if journal["dropped"] or journal["error"]:
            result += f' | log warning: {journal["dropped"]} dropped, error={journal["error"]}'
        return result
    if event == "operator_stop":
        return "Stop requested: " + record["reason"]
    if event == "timing_warning":
        return (
            f'Timing warning: observation {record["capture_age_ms"]:.1f} ms old, '
            f'control {record["scheduled_lag_ms"]:.1f} ms late; policy continuing.'
        )
    if event == "timing_hold":
        return (
            "Timing delay: position hold, session remains active. "
            if record["motor_output"]
            else "Timing delay: waiting for fresh feedback; sensor session remains active. "
        ) + record["reason"]
    if event == "timing_resumed":
        return "Fresh feedback and plan available; policy control resumed."
    if event == "fault":
        return "Stopped on fault: " + record["error"]
    if event == "stopped":
        return f'Session closed. Policy commands sent: {record["commands_issued"]}.'
    return event.replace("_", " ").capitalize()


class DeploymentTelemetry:
    """Nonblocking bounded queue with explicit drop/error counters.

    The caller passes immutable scalar/CPU-array snapshots. High-rate packets
    go only to the requested JSONL file. Lifecycle/health records also go to
    stdout. No raw depth images are archived here. A .gz suffix compresses the
    stream; existing files are never overwritten.
    """

    def __init__(self, path=None, *, capacity=128, stdout=True):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError("Positive integer telemetry queue capacity required")
        self.path = None if path is None else str(Path(path).expanduser().resolve())
        self.stdout = bool(stdout)
        self.queue = queue.Queue(maxsize=capacity)
        self.record_preview = True
        self.dropped = 0
        self.written = 0
        self.error = None
        self.closing = threading.Event()
        self.stream = None
        if self.path is not None:
            target = Path(self.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            self.stream = (
                gzip.open(target, "xt", encoding="utf-8")
                if target.suffix == ".gz"
                else target.open("x", encoding="utf-8")
            )
        self.thread = threading.Thread(
            target=self._write, daemon=True, name="native-recovery-telemetry"
        )
        self.thread.start()

    def emit(self, record, *, high_rate=False):
        if not self.record_preview and record["event"] in (
            "startup_proposal",
            "startup_preview",
            "startup_rejected",
        ):
            return True
        if high_rate and self.stream is None:
            return True
        if self.closing.is_set() or self.error:
            self.dropped += 1
            return False
        try:
            self.queue.put_nowait((record, high_rate, None))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def barrier(self, timeout_s=5.0):
        """Startup-only flush; never call this from the50Hz control loop."""
        event = threading.Event()
        try:
            self.queue.put((None, False, event), timeout=timeout_s)
        except queue.Full:
            return False
        return event.wait(timeout_s) and self.error is None

    def _write(self):
        try:
            while not self.closing.is_set() or not self.queue.empty():
                try:
                    record, high_rate, barrier = self.queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    if barrier is not None:
                        if self.stream:
                            self.stream.flush()
                        barrier.set()
                        continue
                    line = json.dumps(
                        record, default=_json_default, allow_nan=False, separators=(",", ":")
                    )
                    if self.stream:
                        self.stream.write(line + "\n")
                    if self.stdout and not high_rate:
                        print(console_status(record), flush=True)
                    self.written += 1
                finally:
                    self.queue.task_done()
        except Exception as exc:
            self.error = str(exc)
        finally:
            if self.stream:
                self.stream.close()

    def status(self):
        return dict(
            path=self.path,
            written=self.written,
            dropped=self.dropped,
            queued=self.queue.qsize(),
            error=self.error,
        )

    def close(self, timeout_s=2.0):
        self.closing.set()
        self.thread.join(timeout=timeout_s)
        return dict(self.status(), complete=not self.thread.is_alive() and self.error is None)
