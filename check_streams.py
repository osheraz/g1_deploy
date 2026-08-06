from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG

from common.depth_client import SimpleDepthClient
from common.remote_controller import KeyMap, RemoteController
from common.tactile_render import render_sensor_panels
from common.viser_display import DeployViser
from config import CONFIGS_DIR, DEFAULT_URDF_PATH


@dataclass
class StreamConfig:
    msg_type: str
    lowstate_topic: str
    control_dt: float
    viser_display: bool = False
    use_depth_obs: bool = False
    depth_server_ip: str = "127.0.0.1"
    depth_server_port: int = 5555
    depth_raw_height: int = 480
    depth_raw_width: int = 640
    use_tactile: bool = False
    tactile_sensor_order: list[str] = field(default_factory=list)
    action_joint2motor_idx: list[int] = field(default_factory=list)
    min_d: float = 0.1
    max_d: float = 2.0
    depth_scale: float = 0.001
    urdf_path: str = DEFAULT_URDF_PATH


def load_stream_config(file_path: str | Path) -> StreamConfig:
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = (CONFIGS_DIR / path).resolve()

    with path.open("r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    return StreamConfig(
        msg_type=data["msg_type"],
        lowstate_topic=data["lowstate_topic"],
        control_dt=data["control_dt"],
        viser_display=data.get("viser_display", False),
        use_depth_obs=data.get("use_depth_obs", False),
        depth_server_ip=data.get("depth_server_ip", "127.0.0.1"),
        depth_server_port=data.get("depth_server_port", 5555),
        depth_raw_height=data.get("depth_raw_height", 480),
        depth_raw_width=data.get("depth_raw_width", 640),
        use_tactile=data.get("use_tactile", False),
        tactile_sensor_order=data.get(
            "tactile_sensor_order", ["LF", "RF", "WB", "LH", "RH"]
        ),
        action_joint2motor_idx=data.get("action_joint2motor_idx", []),
        min_d=data.get("min_d", 0.1),
        max_d=data.get("max_d", 2.0),
        depth_scale=data.get("depth_scale", 0.001),
        urdf_path=data.get("urdf_path", DEFAULT_URDF_PATH),
    )


class LowStateMonitor:
    def __init__(self, config: StreamConfig):
        self.config = config
        self.remote = RemoteController()
        self.latest_msg = None
        self.last_rx_time = 0.0
        self.message_count = 0

        if config.msg_type == "hg":
            msg_type = LowStateHG
        elif config.msg_type == "go":
            msg_type = LowStateGo
        else:
            raise ValueError(f"Unsupported msg_type: {config.msg_type}")

        self.subscriber = ChannelSubscriber(config.lowstate_topic, msg_type)
        self.subscriber.Init(self._handler, 10)

    def _handler(self, msg):
        self.latest_msg = msg
        self.last_rx_time = time.time()
        self.message_count += 1
        self.remote.set(msg.wireless_remote)

    @property
    def connected(self) -> bool:
        return self.latest_msg is not None

    def age_seconds(self) -> Optional[float]:
        if not self.last_rx_time:
            return None
        return time.time() - self.last_rx_time

    def summary(self) -> str:
        if self.latest_msg is None:
            return "robot: waiting for low_state..."

        imu = self.latest_msg.imu_state
        tick = getattr(self.latest_msg, "tick", -1)
        mode_machine = getattr(self.latest_msg, "mode_machine", -1)
        gyro = ",".join(f"{v:+.3f}" for v in imu.gyroscope)
        quat = ",".join(f"{v:+.3f}" for v in imu.quaternion)
        pressed = [
            name
            for name in ("A", "B", "X", "Y", "start", "select")
            if self.remote.button[getattr(KeyMap, name)] == 1
        ]
        buttons = ",".join(pressed) if pressed else "-"
        age_s = self.age_seconds()
        age_text = f"{age_s:.2f}s" if age_s is not None else "n/a"

        return (
            f"robot: ok msgs={self.message_count} tick={tick} mode={mode_machine} "
            f"age={age_text} gyro=[{gyro}] quat=[{quat}] buttons={buttons}"
        )

    def get_qj(self) -> Optional[list[float]]:
        if self.latest_msg is None:
            return None
        if not self.config.action_joint2motor_idx:
            return None
        return [
            float(self.latest_msg.motor_state[motor_idx].q)
            for motor_idx in self.config.action_joint2motor_idx
        ]


class ViserUpdateThread(threading.Thread):
    def __init__(
        self,
        viser_display: DeployViser,
        low_state_monitor: LowStateMonitor,
        depth_client: Optional[SimpleDepthClient],
        tactile_client: Optional["TactileBridgeMonitor"],
        hz: float = 30.0,
    ):
        super().__init__(daemon=True)
        self.viser_display = viser_display
        self.low_state_monitor = low_state_monitor
        self.depth_client = depth_client
        self.tactile_client = tactile_client
        self.hz = hz
        self.running = True

    def run(self):
        dt = 1.0 / self.hz
        while self.running:
            msg = self.low_state_monitor.latest_msg
            qj = self.low_state_monitor.get_qj()
            if msg is not None and qj is not None:
                self.viser_display.update(
                    low_state=msg,
                    qj=qj,
                    depth_raw=self.depth_client.latest_depth if self.depth_client else None,
                    rgb_bgr=self.depth_client.latest_color if self.depth_client else None,
                )
            if self.tactile_client is not None:
                overall, sensor_status = self.tactile_client.sensor_status_text()
                self.viser_display.update_tactile(overall, sensor_status)
                self.viser_display.update_tactile_images(
                    self.tactile_client.render_panels()
                )
            time.sleep(dt)


class TactileBridgeMonitor:
    def __init__(self, sensor_names: list[str]):
        self.sensor_names = sensor_names
        self.proc: Optional[subprocess.Popen] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.running = False
        self.frame_count = 0
        self.last_snapshot: dict = {
            "frame_count": 0,
            "samples": {
                sensor: {"raw_len": 0, "gyro_len": 0, "last_rx_time": 0.0}
                for sensor in sensor_names
            },
            "timestamp": 0.0,
        }
        self.calibration_window = 30
        self.calibration_counts = {sensor: 0 for sensor in sensor_names}
        self.calibration_sums = {
            sensor: [0.0] * 256 for sensor in sensor_names
        }
        self.baselines: dict[str, list[int] | None] = {
            sensor: None for sensor in sensor_names
        }
        self.deadzone = 12

    def start(self) -> None:
        env = os.environ.copy()
        # Inherit the RMW instead of forcing one: the robot publishes with the
        # default rmw_fastrtps_cpp, and FastDDS/CycloneDDS do not interoperate,
        # so hardcoding cyclonedds here made the bridge silently see no topics.
        # Override with RMW_IMPLEMENTATION in the shell if both ends change.
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "common" / "tactile_bridge.py"),
            "--period",
            "0.05",
        ]
        for sensor in self.sensor_names:
            cmd.extend(["--sensor", sensor])
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self.running = True
        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.reader_thread.start()

    def _read_loop(self) -> None:
        assert self.proc is not None
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                snapshot = json.loads(line)
            except json.JSONDecodeError:
                print(f"[tactile_bridge] {line}")
                continue
            self.last_snapshot = snapshot
            self.frame_count = snapshot.get("frame_count", 0)
            self._update_calibration(snapshot)
        self.running = False

    def _update_calibration(self, snapshot: dict) -> None:
        for sensor in self.sensor_names:
            if self.baselines[sensor] is not None:
                continue
            raw_values = snapshot.get("samples", {}).get(sensor, {}).get("raw_values", [])
            if len(raw_values) < 256:
                continue
            for idx, value in enumerate(raw_values[:256]):
                self.calibration_sums[sensor][idx] += float(value)
            self.calibration_counts[sensor] += 1
            if self.calibration_counts[sensor] >= self.calibration_window:
                count = float(self.calibration_counts[sensor])
                self.baselines[sensor] = [
                    int(round(total / count)) for total in self.calibration_sums[sensor]
                ]

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=1.0)

    def sensor_status_text(self) -> tuple[str, dict[str, str]]:
        samples = self.last_snapshot.get("samples", {})
        parts = {}
        overall_on = False
        for sensor in self.sensor_names:
            sample = samples.get(sensor, {})
            raw_len = int(sample.get("raw_len", 0))
            gyro_len = int(sample.get("gyro_len", 0))
            on = raw_len > 0 or gyro_len > 0
            overall_on = overall_on or on
            status = "ON" if on else "OFF"
            parts[sensor] = f"{status}(raw={raw_len},gyro={gyro_len})"
        return ("ON" if overall_on else "OFF"), parts

    def render_panels(self) -> dict[str, object]:
        samples = self.last_snapshot.get("samples", {})
        panels = {}
        for sensor in self.sensor_names:
            raw_values = samples.get(sensor, {}).get("raw_values", [])
            if raw_values:
                panels.update(
                    render_sensor_panels(
                        sensor,
                        raw_values,
                        baseline_values=self.baselines.get(sensor),
                        deadzone=self.deadzone,
                    )
                )
        return panels


def summarize_depth(
    client: Optional[SimpleDepthClient], last_count: int, dt: float
) -> tuple[str, int]:
    if client is None:
        return "depth: disabled", last_count

    depth = client.latest_depth
    new_frames = client.frame_count - last_count
    # measured over the real elapsed time, so --period does not skew it
    fps = new_frames / dt if dt > 0 else 0.0
    if depth is None:
        return (
            f"depth: OFF frames={client.frame_count} fps={fps:.1f}"
        ), client.frame_count

    return (
        "depth: ON "
        f"frames={client.frame_count} (+{new_frames}) fps={fps:.1f} "
        f"shape={depth.shape} min={float(depth.min()):.3f} max={float(depth.max()):.3f}"
    ), client.frame_count


def summarize_tactile(
    client: Optional[TactileBridgeMonitor], last_count: int
) -> tuple[str, int]:
    if client is None:
        return "tactile: disabled", last_count

    new_frames = client.frame_count - last_count
    overall, sensor_parts = client.sensor_status_text()
    parts = [f"{sensor}:{sensor_parts[sensor]}" for sensor in client.sensor_names]

    return (
        f"tactile: {overall} frames={client.frame_count} (+{new_frames}) "
        + " ".join(parts)
    ), client.frame_count


def main():
    parser = argparse.ArgumentParser(
        description="Safe stream monitor for robot low-state, depth, and tactile feeds."
    )
    parser.add_argument("--config", type=str, default="g1_23dof.yaml")
    parser.add_argument("--period", type=float, default=1.0)
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="Override URDF path; defaults to urdf_path from the config yaml.",
    )
    args = parser.parse_args()

    config = load_stream_config(args.config)
    urdf_path = Path(args.urdf or config.urdf_path).expanduser()

    depth_client = None
    if config.use_depth_obs:
        depth_client = SimpleDepthClient(
            server_address=config.depth_server_ip,
            port=config.depth_server_port,
            height=config.depth_raw_height,
            width=config.depth_raw_width,
            show_debug=False,
        )
        depth_client.start()

    tactile_client = None
    if config.use_tactile:
        tactile_client = TactileBridgeMonitor(sensor_names=config.tactile_sensor_order)
        tactile_client.start()

    ChannelFactoryInitialize(0)
    low_state_monitor = LowStateMonitor(config)

    viser_display = None
    if config.viser_display:
        viser_display = DeployViser(
            urdf_path=urdf_path,
            with_depth=config.use_depth_obs,
            depth_min=config.min_d,
            depth_max=config.max_d,
            depth_scale=config.depth_scale,
            tactile_sensors=config.tactile_sensor_order if config.use_tactile else [],
        )

    print(f"Loaded config: {args.config}")
    print(f"Low-state topic: {config.lowstate_topic} ({config.msg_type})")
    if config.use_depth_obs:
        print(
            "Depth stream: "
            f"tcp://{config.depth_server_ip}:{config.depth_server_port} "
            f"({config.depth_raw_height}x{config.depth_raw_width})"
        )
    if config.use_tactile:
        print(f"Tactile sensors: {config.tactile_sensor_order}")
    if viser_display is not None:
        print(f"Viser: ON urdf={urdf_path}")
    print("This script never publishes robot commands. Press Ctrl+C to stop.")

    last_depth_count = 0
    last_tactile_count = 0
    viser_thread = None
    if viser_display is not None:
        viser_thread = ViserUpdateThread(
            viser_display=viser_display,
            low_state_monitor=low_state_monitor,
            depth_client=depth_client,
            tactile_client=tactile_client,
            hz=30.0,
        )
        viser_thread.start()

    last_time = time.perf_counter()

    try:
        while True:
            time.sleep(args.period)
            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            print(low_state_monitor.summary())

            depth_summary, last_depth_count = summarize_depth(
                depth_client, last_depth_count, dt
            )
            print(depth_summary)

            tactile_summary, last_tactile_count = summarize_tactile(
                tactile_client, last_tactile_count
            )
            print(tactile_summary)
            print("-" * 80)
    except KeyboardInterrupt:
        print("Stopping stream monitor...")
    finally:
        if viser_thread is not None:
            viser_thread.running = False
            viser_thread.join(timeout=1.0)
        if depth_client is not None:
            depth_client.running = False
            depth_client.join(timeout=2.0)
        if tactile_client is not None:
            tactile_client.stop()


if __name__ == "__main__":
    main()
