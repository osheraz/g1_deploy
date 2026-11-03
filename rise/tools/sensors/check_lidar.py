"""Read-only LiDAR/robot/depth viewer and stream monitor, with offline replay.

python g1_deploy/rise/tools/sensors/check_lidar.py --replay logs/lidar_live/upright_torso --viser
python g1_deploy/rise/tools/sensors/check_lidar.py --viser  # connect to running preview publisher
"""

if __name__ == "__main__" and not __package__:
    import _bootstrap  # System Python workers use this checkout, not the policy virtualenv.
from rise.paths import ROOT, CONFIG_ROOT, PACKAGE_ROOT
import argparse
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.sensing.lidar_stream import LidarClient
from rise.recording.session import DEFAULT_WS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", default="tcp://127.0.0.1:15560")
    p.add_argument("--viser", action="store_true")
    p.add_argument("--replay", type=Path)
    p.add_argument("--duration-s", type=float, default=0.0)
    p.add_argument(
        "--live", action="store_true", help="Start the local Livox driver and ROS preview bridge"
    )
    p.add_argument("--interface", default="enp132s0")
    p.add_argument(
        "--lidar-hz",
        type=float,
        default=50.0,
        help="Live preview packet rate; does not change laser point rate or planner configuration",
    )
    p.add_argument("--driver-config", type=Path, default=CONFIG_ROOT / "lidar/MID360_bluefin.json")
    p.add_argument("--ros-workspace", type=Path, default=os.environ.get("LIDAR_WS", DEFAULT_WS))
    p.add_argument(
        "--urdf", default=str(ROOT / "humanoidverse2/data/robots/g1/g1_29dof_sonic.urdf")
    )
    p.add_argument("--calibration", type=Path)
    p.add_argument("--depth-server")
    p.add_argument("--depth-port", type=int, default=15556)
    a = p.parse_args()
    if not 0 <= a.duration_s < float("inf"):
        p.error("duration-s must be finite and nonnegative")
    if not 0.5 <= a.lidar_hz <= 100.0:
        p.error("lidar-hz must be between 0.5 and 100")
    if a.live and a.replay:
        p.error("Choose live or replay")
    if a.live and not (a.ros_workspace / "install/setup.bash").exists():
        p.error("Set --ros-workspace or LIDAR_WS to the built ROS workspace")
    if a.live and not a.driver_config.is_file():
        p.error("Livox driver configuration is missing")
    client = LidarClient(a.endpoint)
    children = []
    depth = None
    view = None
    client.start()
    try:
        if a.replay:
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(PACKAGE_ROOT / "tools/sensors/publish_lidar_preview.py"),
                        "--capture",
                        str(a.replay),
                        "--loop",
                        "--endpoint",
                        a.endpoint,
                    ],
                    start_new_session=True,
                )
            )
        if a.live:
            # Positional bash arguments keep paths and endpoints out of shell code.
            env_script = 'set -e; source /opt/ros/humble/setup.bash; source "$1"; shift; export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=7; export PYTHONPATH=${UNITREE_SDK_ROOT:-$HOME/unitree_sdk2_python}:${PYTHONPATH:-}; exec "$@"'

            def ros_process(command):
                return subprocess.Popen(
                    [
                        "bash",
                        "-c",
                        env_script,
                        "lidar-env",
                        str(a.ros_workspace / "install/setup.bash"),
                        *command,
                    ],
                    start_new_session=True,
                )

            children.append(
                ros_process(
                    [
                        "/usr/bin/python3",
                        str(PACKAGE_ROOT / "tools/sensors/publish_lidar_preview.py"),
                        "--interface",
                        a.interface,
                        "--endpoint",
                        a.endpoint,
                    ]
                )
            )
            children.append(
                ros_process(
                    [
                        "ros2",
                        "run",
                        "livox_ros_driver2",
                        "livox_ros_driver2_node",
                        "--ros-args",
                        "-p",
                        "xfer_format:=0",
                        "-p",
                        "multi_topic:=0",
                        "-p",
                        "data_src:=0",
                        "-p",
                        f"publish_freq:={a.lidar_hz}",
                        "-p",
                        "output_data_type:=0",
                        "-p",
                        "frame_id:=lidar",
                        "-p",
                        "user_config_path:=" + str(a.driver_config.resolve()),
                        "-p",
                        "cmdline_input_bd_code:=livox0000000001",
                    ]
                )
            )
        if a.depth_server:
            from core.sensing.depth_client import SimpleDepthClient

            depth = SimpleDepthClient(
                server_address=a.depth_server,
                port=a.depth_port,
                height=480,
                width=640,
                show_debug=False,
            )
            depth.start()
        if a.viser:
            from core.review.display import DeployViser
            from core.review.lidar import LidarView
            from rise.sensing.pelvis_odometry import G1_JOINTS

            display = DeployViser(
                Path(a.urdf),
                with_depth=bool(depth),
                depth_min=0.1,
                depth_max=3.0,
                depth_scale=0.001,
                joint_names=G1_JOINTS,
            )
            view = LidarView(display, a.urdf, drive_robot=True, calibration=a.calibration)
        print("Read-only LiDAR monitor. No motor commands.", flush=True)
        started = time.monotonic()
        last = 0.0
        last_depth = None
        while not a.duration_s or time.monotonic() - started < a.duration_s:
            if any(c.poll() is not None for c in children):
                raise RuntimeError("A LiDAR source process exited; see its output")
            if view is not None:
                view.update(client)
                depth_frame = depth.latest_depth_full if depth else None
                if depth_frame is not None and depth_frame is not last_depth:
                    view.display._update_depth_image(depth_frame)
                    last_depth = depth_frame
            if time.monotonic() - last >= 1:
                print(client.summary(), flush=True)
                if view is not None:
                    print(view.summary(), flush=True)
                last = time.monotonic()
            time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
        if depth:
            depth.running = False
            depth.join(timeout=2)
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
        for child in children:
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if view is not None:
            view.server.stop()


if __name__ == "__main__":
    main()
