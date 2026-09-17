#!/usr/bin/env python3
"""Replay physical C++ robot diagnostics in Viser.

Two old-23-DoF G1 models are shown side-by-side:

* MEASURED (left / negative Y): LowState joint positions and relative IMU pose.
* COMMANDED (right / positive Y): the exact position targets sent to LowCmd.

The log has no global pelvis translation, so both models remain spatially
anchored.  The measured model does show orientation change relative to the
first recorded IMU quaternion.

Examples:

    python tools/replay/replay_robot_diagnostics.py
    python tools/replay/replay_robot_diagnostics.py latest --start policy
    python tools/replay/replay_robot_diagnostics.py /path/to/session --port 8081
"""
from __future__ import annotations

import argparse
import csv
import getpass
import time
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LOG_ROOT = Path(f"/data/{getpass.getuser()}/logs/e_skin/teleop")
DEFAULT_URDF = (
    REPO_ROOT
    / "research/repos/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/old/g1_23dof.urdf"
)

PHYSICAL_MOTOR_INDICES = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
     15, 16, 17, 18, 19, 22, 23, 24, 25, 26],
    dtype=np.int64,
)
JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)
SHORT_NAMES = tuple(name.removesuffix("_joint") for name in JOINT_NAMES)


def resolve_csv(value: str) -> Path:
    if value == "latest":
        candidates = list(
            DEFAULT_LOG_ROOT.glob(
                "*/terminal1_robot_diagnostics/robot_diagnostics.csv"
            )
        )
        if not candidates:
            raise FileNotFoundError(f"no physical diagnostic CSV below {DEFAULT_LOG_ROOT}")
        return max(candidates, key=lambda path: path.stat().st_mtime).resolve()

    path = Path(value).expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        direct = path / "terminal1_robot_diagnostics/robot_diagnostics.csv"
        if direct.is_file():
            return direct
        direct = path / "robot_diagnostics.csv"
        if direct.is_file():
            return direct
        candidates = list(path.glob("**/robot_diagnostics.csv"))
        if len(candidates) == 1:
            return candidates[0].resolve()
        if candidates:
            return max(candidates, key=lambda item: item.stat().st_mtime).resolve()
    raise FileNotFoundError(f"could not resolve robot_diagnostics.csv from {value}")


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or ())
    if not rows:
        raise ValueError(f"empty diagnostic CSV: {path}")

    required = {
        "time_ms", "encoder_mode", "playing", "motion_name",
        "imu_qw", "imu_qx", "imu_qy", "imu_qz",
        *(f"q_{i}" for i in range(29)),
        *(f"q_cmd_{i}" for i in range(29)),
        *(f"q_error_{i}" for i in range(29)),
        *(f"tau_est_{i}" for i in range(29)),
        *(f"temp_winding_{i}" for i in range(29)),
        *(f"temp_driver_{i}" for i in range(29)),
    }
    missing = sorted(required - fieldnames)
    if missing:
        raise ValueError(f"diagnostic CSV is missing fields: {missing}")

    def scalar(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows], dtype=np.float64)

    def motors(prefix: str) -> np.ndarray:
        return np.asarray(
            [[float(row[f"{prefix}_{i}"]) for i in range(29)] for row in rows],
            dtype=np.float64,
        )

    time_s = scalar("time_ms") * 1.0e-3
    time_s -= time_s[0]
    return {
        "time_s": time_s,
        "encoder_mode": scalar("encoder_mode").astype(np.int32),
        "playing": scalar("playing").astype(bool),
        "motion_name": np.asarray([row["motion_name"] for row in rows], dtype=object),
        "imu": np.column_stack(
            tuple(scalar(name) for name in ("imu_qw", "imu_qx", "imu_qy", "imu_qz"))
        ),
        "q": motors("q")[:, PHYSICAL_MOTOR_INDICES],
        "q_cmd": motors("q_cmd")[:, PHYSICAL_MOTOR_INDICES],
        "q_error": motors("q_error")[:, PHYSICAL_MOTOR_INDICES],
        "tau": motors("tau_est")[:, PHYSICAL_MOTOR_INDICES],
        "temp_winding": motors("temp_winding")[:, PHYSICAL_MOTOR_INDICES],
        "temp_driver": motors("temp_driver")[:, PHYSICAL_MOTOR_INDICES],
    }


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    result = np.asarray(q, dtype=np.float64).copy()
    result[1:] *= -1.0
    return result


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dtype=np.float64,
    )


def relative_quaternions(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    normalized = quaternions / np.maximum(norms, 1.0e-12)
    inverse_initial = quat_conjugate(normalized[0])
    result = np.asarray(
        [quat_multiply(inverse_initial, q) for q in normalized], dtype=np.float64
    )
    # Quaternion signs are equivalent, but choosing a continuous sign prevents
    # a browser interpolation from taking a visually distracting long route.
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result


def phase(data: dict[str, np.ndarray], index: int) -> str:
    motion = str(data["motion_name"][index])
    mode = int(data["encoder_mode"][index])
    playing = bool(data["playing"][index])
    if not motion and mode == -2:
        return "PRE-ABXY HOLD"
    if not playing:
        return f"TRANSITION ({motion or 'no motion'})"
    if mode == 2 or motion == "streamed":
        return "POSE"
    if mode == 0 or motion == "planner_motion":
        return "PLANNER"
    return f"mode={mode} {motion}"


def first_index(data: dict[str, np.ndarray], start: str) -> int:
    if start == "beginning":
        return 0
    if start == "policy":
        hits = np.flatnonzero(data["motion_name"] != "")
    else:
        hits = np.flatnonzero(
            (data["encoder_mode"] == 2) | (data["motion_name"] == "streamed")
        )
    return int(hits[0]) if len(hits) else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?", default="latest",
        help="CSV/session directory, or 'latest' (default)",
    )
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Viser bind address (default: localhost only)",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--start", choices=("beginning", "policy", "pose"), default="policy",
        help="initial replay position (default: first policy-transition row)",
    )
    args = parser.parse_args()

    csv_path = resolve_csv(args.path)
    urdf_path = args.urdf.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"old-23-DoF URDF not found: {urdf_path}")
    data = load_csv(csv_path)
    relative_imu = relative_quaternions(data["imu"])
    count = len(data["time_s"])

    server = viser.ViserServer(
        host=args.host, port=args.port, label="G1 command vs measured replay"
    )
    server.scene.add_grid(
        "/ground", width=3.0, height=3.0, position=(0.0, 0.0, -0.78)
    )
    measured_frame = server.scene.add_frame(
        "/measured", position=(0.0, -0.65, 0.0), show_axes=False
    )
    command_frame = server.scene.add_frame(
        "/commanded", position=(0.0, 0.65, 0.0), show_axes=False
    )
    server.scene.add_label(
        "/measured_label", "MEASURED (LowState)", position=(0.0, -0.65, 0.72)
    )
    server.scene.add_label(
        "/commanded_label", "COMMANDED (LowCmd target)", position=(0.0, 0.65, 0.72)
    )
    measured_vis = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        root_node_name="/measured",
        load_meshes=True,
        load_collision_meshes=False,
    )
    command_vis = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        root_node_name="/commanded",
        load_meshes=True,
        load_collision_meshes=False,
    )

    with server.gui.add_folder("Replay"):
        initial = first_index(data, args.start)
        index_handle = server.gui.add_slider(
            "Index", min=0, max=count - 1, step=1, initial_value=initial
        )
        play_handle = server.gui.add_checkbox("Play", False)
        fps_handle = server.gui.add_number("FPS", float(args.fps), min=1.0, max=200.0)
        time_handle = server.gui.add_text("Time", "-")
        phase_handle = server.gui.add_text("Phase", "-")
        motion_handle = server.gui.add_text("Motion", "-")
        worst_handle = server.gui.add_text("Worst tracking error", "-")
        ankle_handle = server.gui.add_text("Ankle pitch L/R", "-")
        torque_handle = server.gui.add_text("Ankle pitch torque L/R", "-")
        temp_handle = server.gui.add_text("Ankle max temp L/R", "-")
        note_handle = server.gui.add_text(
            "Scope", "Joint pose + relative IMU orientation; no global translation"
        )
        note_handle.disabled = True

    def render(index: int) -> None:
        measured_vis.update_cfg(
            {name: float(value) for name, value in zip(JOINT_NAMES, data["q"][index])}
        )
        command_vis.update_cfg(
            {
                name: float(value)
                for name, value in zip(JOINT_NAMES, data["q_cmd"][index])
            }
        )
        measured_frame.wxyz = tuple(float(value) for value in relative_imu[index])
        command_frame.wxyz = (1.0, 0.0, 0.0, 0.0)

        error_deg = np.degrees(np.abs(data["q_error"][index]))
        worst = int(np.argmax(error_deg))
        left_pitch, right_pitch = 4, 10
        time_handle.value = f"{data['time_s'][index]:.3f} s   ({index}/{count - 1})"
        phase_handle.value = phase(data, index)
        motion_handle.value = str(data["motion_name"][index]) or "-"
        worst_handle.value = f"{SHORT_NAMES[worst]}: {error_deg[worst]:.1f} deg"
        ankle_handle.value = (
            f"{np.degrees(data['q_error'][index, left_pitch]):+.1f} / "
            f"{np.degrees(data['q_error'][index, right_pitch]):+.1f} deg"
        )
        torque_handle.value = (
            f"{data['tau'][index, left_pitch]:+.1f} / "
            f"{data['tau'][index, right_pitch]:+.1f} Nm"
        )
        left_temp = max(
            data["temp_winding"][index, left_pitch],
            data["temp_driver"][index, left_pitch],
        )
        right_temp = max(
            data["temp_winding"][index, right_pitch],
            data["temp_driver"][index, right_pitch],
        )
        temp_handle.value = f"{left_temp:.0f} / {right_temp:.0f} C"

    render(initial)
    print(f"Loaded: {csv_path}")
    print(f"Rows: {count}; duration: {data['time_s'][-1]:.3f} s")
    print("Measured robot is on the left; commanded robot is on the right.")
    browser_host = "localhost" if args.host in ("127.0.0.1", "::1") else args.host
    print(f"Viser: http://{browser_host}:{args.port}")

    current = initial
    last_step = time.perf_counter()
    try:
        while True:
            requested = int(index_handle.value)
            if requested != current:
                current = requested
                render(current)
            if play_handle.value:
                now = time.perf_counter()
                target_dt = 1.0 / max(1.0, float(fps_handle.value))
                if now - last_step >= target_dt:
                    last_step = now
                    current = min(current + 1, count - 1)
                    index_handle.value = current
                    render(current)
                    if current == count - 1:
                        play_handle.value = False
            time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
