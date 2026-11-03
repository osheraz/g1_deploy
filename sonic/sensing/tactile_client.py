from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional

import zmq

DEFAULT_PORT = 5557

# Sensor ids the server may publish; see jq_tactile_driver/reader.py.
KNOWN_SENSORS = ("LF", "RF", "WB", "LH", "RH")


@dataclass
class TactileSample:
    sensor: str
    raw_sensor_bytes: bytes = b""
    raw_imu_bytes: bytes = b""


class TactileClient(threading.Thread):
    """ZMQ SUB for tactile frames published by ``jq_tactile_driver.server``.

    Mirrors ``SimpleDepthClient``: a daemon thread holding the latest frame per
    sensor. No ROS.
    """

    def __init__(
        self,
        sensor_names: list[str],
        server_address: str = "192.168.123.164",
        port: int = DEFAULT_PORT,
    ):
        super().__init__(daemon=True)
        self.sensor_names = [n for n in sensor_names if n in KNOWN_SENSORS]
        self.server_address = server_address
        self.port = port

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{server_address}:{port}")
        # Subscribe per sensor: the first frame part is the sensor name, so the
        # prefix filter drops sensors this deploy does not care about.
        for sensor in self.sensor_names:
            self.socket.setsockopt(zmq.SUBSCRIBE, sensor.encode("ascii"))
        self.socket.setsockopt(zmq.RCVTIMEO, 500)

        self._lock = threading.Lock()
        self.running = True
        self.frame_count = 0
        self.samples: Dict[str, TactileSample] = {
            sensor: TactileSample(sensor=sensor) for sensor in self.sensor_names
        }

    def run(self) -> None:
        while self.running:
            try:
                parts = self.socket.recv_multipart()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break

            if len(parts) != 3:
                continue
            name, raw, imu = parts
            sensor = name.decode("ascii", "ignore")
            if sensor not in self.samples:
                continue

            with self._lock:
                sample = self.samples[sensor]
                sample.raw_sensor_bytes = raw
                sample.raw_imu_bytes = imu
                self.frame_count += 1

    def stop(self) -> None:
        self.running = False
        self.join(timeout=2.0)
        self.socket.close()
        self.context.term()

    def snapshot(self) -> Dict[str, dict]:
        with self._lock:
            return {
                sensor: {
                    "raw_sensor_bytes": s.raw_sensor_bytes,
                    "raw_imu_bytes": s.raw_imu_bytes,
                }
                for sensor, s in self.samples.items()
            }


def main():
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Print tactile stream rates.")
    ap.add_argument("--server", default="192.168.123.164")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument(
        "--sensor", action="append", dest="sensors", default=None, help="repeatable; default: all"
    )
    args = ap.parse_args()

    client = TactileClient(
        sensor_names=args.sensors or list(KNOWN_SENSORS),
        server_address=args.server,
        port=args.port,
    )
    client.start()
    print(f"Subscribed to tcp://{args.server}:{args.port}")

    last = 0
    try:
        while True:
            time.sleep(1.0)
            new = client.frame_count - last
            last = client.frame_count
            parts = []
            for sensor, s in client.snapshot().items():
                if s["raw_sensor_bytes"]:
                    nz = sum(1 for b in s["raw_sensor_bytes"] if b)
                    parts.append(f"{sensor}(nonzero={nz})")
            print(f"[tactile] {new} Hz total | {' '.join(parts) or 'no data'}")
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()


if __name__ == "__main__":
    main()
