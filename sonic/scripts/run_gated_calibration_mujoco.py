#!/usr/bin/env python3
"""Review-gated 23-DoF calibration using a two-shoulder MuJoCo fixture."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import sys
import time

import mujoco
from mujoco.glfw import glfw
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sonic.calibration.gated_protocol import (
    DT, MOTOR_NAMES, ReviewBlock, build_review_protocol, save_review_protocol,
)
from sonic.calibration.mujoco_plant import SuspendedMujocoPlant
from sonic.calibration.protocol import PRESENT_23


CAPTURE = "capture"
APPROVE = "approve"
SKIP = "skip"
CANCEL = "cancel"
ABORT = "abort"
KEY_COMMANDS = {
    glfw.KEY_KP_5: CAPTURE,
    glfw.KEY_KP_ENTER: APPROVE,
    glfw.KEY_KP_3: SKIP,
    glfw.KEY_KP_0: CANCEL,
    glfw.KEY_KP_DECIMAL: ABORT,
}


def quat_from_rpy_deg(rpy_deg: list[float]) -> np.ndarray:
    roll, pitch, yaw = np.radians(np.asarray(rpy_deg, dtype=np.float64)) / 2.0
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def quat_distance_rad(q: np.ndarray, reference: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    dots = np.abs(np.sum(q * reference, axis=-1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def block_preview(block: ReviewBlock, index: int, total: int) -> str:
    joints = ", ".join(MOTOR_NAMES[motor] for motor in block.active_motors) or "none"
    peaks = ", ".join(
        f"{MOTOR_NAMES[motor]}={peak:.4f}rad"
        for motor, peak in block.peak_offsets.items()
    ) or "zero offset"
    return (
        f"NEXT [{index + 1}/{total}] {block.name} ({block.duration_s:.1f}s)\n"
        f"  {block.description}\n"
        f"  joints: {joints}\n"
        f"  peak offsets: {peaks}"
    )


class NeutralReturn:
    def __init__(self, start: np.ndarray):
        self.start = np.asarray(start, dtype=np.float64).copy()
        self.elapsed = 0.0
        self.duration = max(2.0, float(np.max(np.abs(self.start))) / 0.08)

    def step(self) -> np.ndarray:
        self.elapsed = min(self.duration, self.elapsed + DT)
        alpha = self.elapsed / self.duration
        blend = alpha**3 * (10.0 + alpha * (-15.0 + 6.0 * alpha))
        return (1.0 - blend) * self.start

    @property
    def done(self) -> bool:
        return self.elapsed >= self.duration


def draw_fixture(viewer, plant: SuspendedMujocoPlant, motors: tuple[int, ...]) -> None:
    scene = viewer.user_scn
    scene.ngeom = 0

    def sphere(position, size, color):
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
            np.asarray([size, 0.0, 0.0]), np.asarray(position),
            np.eye(3).reshape(-1), np.asarray(color, dtype=np.float32),
        )
        scene.ngeom += 1

    for index, motor in enumerate(motors):
        sphere(plant.joint_anchor(motor), 0.025, [1.0, .2 + .1 * (index % 5), .1, .9])
    for anchor, point in zip(plant.shoulder_anchors, plant.shoulder_points):
        sphere(anchor, 0.018, [.1, .9, 1.0, .9])
        sphere(point, 0.015, [1.0, .8, .1, .9])
        if scene.ngeom < scene.maxgeom:
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.asarray([.006, 0.0, 0.0]), np.zeros(3),
                np.eye(3).reshape(-1), np.asarray([.2, .8, 1.0, .8], np.float32),
            )
            mujoco.mjv_connector(
                scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE,
                .006, point, anchor,
            )
            scene.ngeom += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("pilot", "screening"), default="pilot")
    parser.add_argument("--view", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-approve", action="store_true",
                        help="headless validation only; automatically capture and approve")
    parser.add_argument("--initial-rpy-deg", nargs=3, type=float, default=(0.0, 0.0, 0.0),
                        metavar=("ROLL", "PITCH", "YAW"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, default=REPO / "sonic_cpp/g1/scene_23dof.xml")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/g1_23dof_sonic.yaml")
    args = parser.parse_args()
    if args.auto_approve and args.view:
        raise SystemExit("--auto-approve is restricted to --no-view validation")
    if not args.view and not args.auto_approve:
        raise SystemExit("headless mode requires --auto-approve")

    output = args.output.expanduser().absolute()
    protocol = build_review_protocol(args.protocol)
    save_review_protocol(protocol, output)
    plant = SuspendedMujocoPlant(
        mjcf=args.mjcf,
        config=args.config,
        suspension="two_shoulder",
        root_quat_wxyz=quat_from_rpy_deg(args.initial_rpy_deg),
    )
    commands: queue.SimpleQueue[str] = queue.SimpleQueue()

    def key_callback(key: int) -> None:
        command = KEY_COMMANDS.get(key)
        if command:
            commands.put(command)

    viewer = None
    if args.view:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(
            plant.model, plant.data, key_callback=key_callback,
            show_left_ui=True, show_right_ui=True,
        )
        viewer.cam.distance = 3.2
        viewer.cam.elevation = -8
        viewer.cam.azimuth = 135
        print(
            "Review-gated calibration controls (numeric keypad):\n"
            "  Num5          capture steady hanging pose\n"
            "  Numpad Enter approve/run the displayed next block\n"
            "  Num3          skip the displayed block\n"
            "  Num0          cancel the active block and return neutral\n"
            "  NumDecimal    abort, return neutral, record, and close\n"
            "All other MuJoCo viewer controls remain unchanged."
        )

    rows: dict[str, list] = {
        key: [] for key in (
            "time_s", "target_model", "command_hardware", "effective_model_target",
            "q_measured", "dq_measured", "tau_est", "physical_q_measured",
            "physical_dq_measured", "physical_tau_est", "command_offset",
            "captured_neutral", "root_position", "root_quaternion",
            "root_linear_velocity", "root_angular_velocity", "imu_quaternion",
            "imu_gyroscope", "imu_accelerometer", "shoulder_points",
            "shoulder_anchors", "suspension_forces", "foot_link_height",
            "contact_count", "contact_pairs", "block_id", "block_name", "phase",
            "gate_state",
        )
    }
    events: list[dict[str, object]] = []
    block_results: list[dict[str, object]] = []
    captured = False
    captured_neutral = plant.neutral.copy()
    captured_quat = plant.root_quat.copy()
    block_index = 0
    block_sample = 0
    active = False
    offset = np.zeros(29, dtype=np.float64)
    returning: NeutralReturn | None = None
    return_reason = ""
    aborted = False
    collision_detected = False
    finished = False
    auto_settle_samples = 0
    block_row_start = 0
    start_wall = time.perf_counter()
    sim_time = 0.0

    def event(command: str, message: str) -> None:
        events.append({
            "time_s": f"{sim_time:.6f}", "command": command,
            "block_id": block_index if block_index < len(protocol.blocks) else -1,
            "block_name": (
                protocol.blocks[block_index].name
                if block_index < len(protocol.blocks) else "complete"
            ),
            "message": message,
        })
        print(f"[CAL {sim_time:8.2f}s] {command}: {message}")

    def stable_to_capture() -> tuple[bool, str]:
        qvel = np.asarray(plant.data.qvel[plant.vadr])[PRESENT_23]
        max_dq = float(np.max(np.abs(qvel)))
        root_w = float(np.linalg.norm(plant.data.qvel[3:6]))
        foot_min = float(np.min(plant.data.xpos[plant.foot_bodies, 2]))
        okay = max_dq <= 0.20 and root_w <= 0.08 and plant.data.ncon == 0 and foot_min > .10
        return okay, (
            f"max|dq|={max_dq:.3f}rad/s root|w|={root_w:.3f}rad/s "
            f"min_foot_z={foot_min:.3f}m contacts={plant.data.ncon}"
        )

    def summarize_block(block: ReviewBlock, start: int, stop: int, outcome: str) -> None:
        if stop <= start:
            return
        command = np.asarray(rows["command_hardware"][start:stop])[:, PRESENT_23]
        measured = np.asarray(rows["q_measured"][start:stop])[:, PRESENT_23]
        dq = np.asarray(rows["dq_measured"][start:stop])[:, PRESENT_23]
        tau = np.asarray(rows["tau_est"][start:stop])[:, PRESENT_23]
        quat = np.asarray(rows["root_quaternion"][start:stop])
        block_results.append({
            "block_id": block.block_id,
            "name": block.name,
            "outcome": outcome,
            "samples": stop - start,
            "tracking_rmse_rad": float(np.sqrt(np.mean((command - measured) ** 2))),
            "max_abs_tracking_error_rad": float(np.max(np.abs(command - measured))),
            "max_abs_velocity_rad_s": float(np.max(np.abs(dq))),
            "max_abs_torque_nm": float(np.max(np.abs(tau))),
            "max_root_attitude_change_deg": float(np.degrees(np.max(
                quat_distance_rad(quat, captured_quat)
            ))),
            "minimum_foot_link_height_m": float(np.min(
                np.asarray(rows["foot_link_height"][start:stop])
            )),
            "maximum_fixture_force_n": float(np.max(np.linalg.norm(
                np.asarray(rows["suspension_forces"][start:stop]), axis=-1
            ))),
            "contact_samples": int(np.sum(np.asarray(
                rows["contact_count"][start:stop]
            ) > 0)),
        })

    event("initial", "waiting for the two-shoulder fixture to settle and for neutral capture")
    if block_index < len(protocol.blocks):
        print(block_preview(protocol.blocks[block_index], block_index, len(protocol.blocks)))

    while not finished:
        completing_block: ReviewBlock | None = None
        if viewer is not None and not viewer.is_running():
            commands.put(ABORT)

        if args.auto_approve:
            okay, _ = stable_to_capture()
            auto_settle_samples = auto_settle_samples + 1 if okay else 0
            if not captured and auto_settle_samples >= 50 and commands.empty():
                commands.put(CAPTURE)
            elif captured and not active and returning is None and not aborted and commands.empty():
                if block_index < len(protocol.blocks):
                    commands.put(APPROVE)

        while not commands.empty():
            command = commands.get()
            if command == ABORT:
                aborted = True
                if active:
                    summarize_block(
                        protocol.blocks[block_index], block_row_start, len(rows["time_s"]),
                        "aborted",
                    )
                    active = False
                returning = NeutralReturn(offset)
                return_reason = "abort"
                event(command, "abort accepted; returning to captured neutral")
                continue
            if command == CAPTURE:
                if captured or active or returning is not None:
                    event(command, "capture rejected: capture is only allowed before the first block")
                    continue
                okay, detail = stable_to_capture()
                if not okay:
                    event(command, "capture rejected: hanging pose is not steady; " + detail)
                    continue
                captured_neutral = np.asarray(plant.data.qpos[plant.qadr]).copy()
                captured_quat = np.asarray(plant.data.qpos[3:7]).copy()
                captured = True
                offset[:] = 0.0
                event(command, "captured 23 joint positions and root attitude; " + detail)
                print(block_preview(protocol.blocks[block_index], block_index, len(protocol.blocks)))
                continue
            if not captured:
                event(command, "ignored until Num5 captures a steady hanging pose")
                continue
            if command == APPROVE:
                if active or returning is not None:
                    event(command, "ignored while a block/neutral return is active")
                elif block_index >= len(protocol.blocks):
                    event(command, "all blocks are complete")
                else:
                    active = True
                    block_sample = 0
                    block_row_start = len(rows["time_s"])
                    event(command, f"approved {protocol.blocks[block_index].name}")
                continue
            if command == SKIP:
                if active or returning is not None:
                    event(command, "skip is accepted only at a review gate")
                elif block_index < len(protocol.blocks):
                    event(command, f"skipped {protocol.blocks[block_index].name}")
                    block_results.append({
                        "block_id": block_index,
                        "name": protocol.blocks[block_index].name,
                        "outcome": "skipped",
                        "samples": 0,
                    })
                    block_index += 1
                    if block_index < len(protocol.blocks):
                        print(block_preview(
                            protocol.blocks[block_index], block_index, len(protocol.blocks)
                        ))
                    else:
                        finished = True
                continue
            if command == CANCEL:
                if active:
                    summarize_block(
                        protocol.blocks[block_index], block_row_start, len(rows["time_s"]),
                        "cancelled",
                    )
                    active = False
                    returning = NeutralReturn(offset)
                    return_reason = "cancel"
                    event(command, "active block cancelled; returning neutral")
                else:
                    event(command, "already at a review gate")

        phase = "waiting_capture" if not captured else "approval_gate"
        if active:
            block = protocol.blocks[block_index]
            offset = block.offsets[block_sample].copy()
            phase = str(block.phase[block_sample])
            block_sample += 1
            if block_sample >= len(block.local_time_s):
                # Complete only after the final row has actually gone through
                # the plant and has been logged.  Every block's last command is
                # neutral, but preserving its block label/sample makes the
                # physical and simulated datasets line up exactly.
                completing_block = block
        elif returning is not None:
            offset = returning.step()
            phase = "abort_return" if return_reason == "abort" else "cancel_return"
            if returning.done:
                offset[:] = 0.0
                returning = None
                if aborted:
                    finished = True
                else:
                    block_index += 1
                    if block_index < len(protocol.blocks):
                        print(block_preview(protocol.blocks[block_index], block_index, len(protocol.blocks)))
                    else:
                        finished = True

        target = (captured_neutral if captured else plant.neutral) + offset
        state = plant.step(target)
        gate_state = (
            "active" if active else "returning" if returning is not None
            else "waiting_capture" if not captured else "approval_gate"
        )
        rows["time_s"].append(sim_time)
        for key in (
            "target_model", "command_hardware", "effective_model_target",
            "q_measured", "dq_measured", "tau_est", "root_position",
            "root_quaternion", "root_linear_velocity", "root_angular_velocity",
            "imu_quaternion", "imu_gyroscope", "imu_accelerometer",
            "shoulder_points", "shoulder_anchors", "suspension_forces",
            "foot_link_height", "contact_count",
            "contact_pairs",
        ):
            rows[key].append(state[key])
        rows["physical_q_measured"].append(state["q_measured"][PRESENT_23])
        rows["physical_dq_measured"].append(state["dq_measured"][PRESENT_23])
        rows["physical_tau_est"].append(state["tau_est"][PRESENT_23])
        rows["command_offset"].append(offset.copy())
        rows["captured_neutral"].append(captured_neutral.copy())
        rows["block_id"].append(block_index if block_index < len(protocol.blocks) else -1)
        rows["block_name"].append(
            protocol.blocks[block_index].name if block_index < len(protocol.blocks) else "complete"
        )
        rows["phase"].append(phase)
        rows["gate_state"].append(gate_state)

        if captured and int(state["contact_count"]) > 0 and not aborted:
            collision_detected = True
            aborted = True
            if active:
                summarize_block(
                    protocol.blocks[block_index], block_row_start,
                    len(rows["time_s"]), "collision_abort",
                )
            active = False
            completing_block = None
            returning = NeutralReturn(offset)
            return_reason = "abort"
            event(
                "collision_abort",
                "MuJoCo contact detected: " + str(state["contact_pairs"]),
            )

        if completing_block is not None:
            active = False
            offset[:] = 0.0
            summarize_block(
                completing_block, block_row_start, len(rows["time_s"]), "completed"
            )
            event(
                "block_complete",
                f"completed {completing_block.name}; neutral review gate",
            )
            block_index += 1
            if block_index < len(protocol.blocks):
                print(block_preview(
                    protocol.blocks[block_index], block_index, len(protocol.blocks)
                ))
            else:
                event("protocol_complete", "all approved blocks completed")
                finished = True

        if viewer is not None:
            motors = protocol.blocks[block_index].active_motors if block_index < len(protocol.blocks) else ()
            draw_fixture(viewer, plant, motors)
            viewer.sync()
        sim_time += DT
        if args.view:
            remaining = start_wall + sim_time - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)

    if viewer is not None:
        viewer.close()

    output.mkdir(parents=True, exist_ok=True)
    with (output / "state.npz").open("wb") as handle:
        np.savez_compressed(handle, **{key: np.asarray(value) for key, value in rows.items()})
    with (output / "events.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(events[0]))
        writer.writeheader()
        writer.writerows(events)
    summary = {
        "schema": "e_skin.review_gated_calibration_mujoco_result.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "embodiment": "g1_23dof",
        "physical_dof": 23,
        "compatibility_storage_slots": 29,
        "physical_motor_indices": PRESENT_23.tolist(),
        "protocol": protocol.name,
        "blocks_total": len(protocol.blocks),
        "blocks_completed": sum(item.get("outcome") == "completed" for item in block_results),
        "blocks_skipped": sum(item.get("outcome") == "skipped" for item in block_results),
        "captured": captured,
        "captured_neutral_rad_29slot": captured_neutral.tolist(),
        "captured_physical_neutral_rad_23": captured_neutral[PRESENT_23].tolist(),
        "captured_root_quaternion_wxyz": captured_quat.tolist(),
        "initial_rpy_deg": list(args.initial_rpy_deg),
        "suspension": "two_shoulder_point_spring",
        "aborted": aborted,
        "collision_detected": collision_detected,
        "samples": len(rows["time_s"]),
        "simulated_duration_s_including_gates": sim_time,
        "final_max_abs_offset_rad": float(np.max(np.abs(offset))),
        "minimum_foot_link_height_m": float(np.min(rows["foot_link_height"])),
        "contact_samples": int(np.sum(np.asarray(rows["contact_count"]) > 0)),
        "block_results": block_results,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "block_results"}, indent=2))
    if collision_detected:
        raise SystemExit("collision validation failed; see summary.json and events.csv")


if __name__ == "__main__":
    main()
