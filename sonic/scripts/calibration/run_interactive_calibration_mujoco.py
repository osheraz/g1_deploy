#!/usr/bin/env python3
"""Operator-in-the-loop suspended calibration rehearsal in MuJoCo."""

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

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sonic.calibration.interactive import InteractiveCalibrationController, InteractiveCommand
from sonic.calibration.mujoco_plant import SuspendedMujocoPlant
from sonic.calibration.protocol import PRESENT_23

KEY_COMMANDS = {
    glfw.KEY_KP_ENTER: InteractiveCommand.EXECUTE,
    glfw.KEY_KP_0: InteractiveCommand.NEUTRAL,
    glfw.KEY_KP_6: InteractiveCommand.NEXT,
    glfw.KEY_KP_4: InteractiveCommand.PREVIOUS,
    glfw.KEY_KP_1: InteractiveCommand.SINGLE_BANK,
    glfw.KEY_KP_3: InteractiveCommand.GROUP_BANK,
    glfw.KEY_KP_ADD: InteractiveCommand.LOOP_MODE,
    glfw.KEY_KP_SUBTRACT: InteractiveCommand.STATIC_MODE,
    glfw.KEY_KP_8: InteractiveCommand.RANGE_UP,
    glfw.KEY_KP_2: InteractiveCommand.RANGE_DOWN,
    glfw.KEY_KP_9: InteractiveCommand.SCALE_UP,
    glfw.KEY_KP_7: InteractiveCommand.SCALE_DOWN,
    glfw.KEY_KP_MULTIPLY: InteractiveCommand.TEMPO_UP,
    glfw.KEY_KP_DIVIDE: InteractiveCommand.TEMPO_DOWN,
    glfw.KEY_KP_5: InteractiveCommand.CAPTURE_NEUTRAL,
    glfw.KEY_KP_DECIMAL: InteractiveCommand.ABORT,
}

# All bindings are keypad-specific GLFW codes. MuJoCo's ordinary digits,
# letters, arrows, brackets, +/- and F-keys retain their native viewer meaning.
if len(KEY_COMMANDS) != len(set(KEY_COMMANDS)):
    raise RuntimeError("interactive calibration has duplicate key bindings")


SCRIPTED_DEMO = (
    (0.10, InteractiveCommand.CAPTURE_NEUTRAL),
    (0.20, InteractiveCommand.STATUS),
    (0.50, InteractiveCommand.EXECUTE),
    (4.00, InteractiveCommand.EXECUTE),
    (7.20, InteractiveCommand.NEXT),
    (7.40, InteractiveCommand.SCALE_UP),
    (7.60, InteractiveCommand.EXECUTE),
    (11.00, InteractiveCommand.NEUTRAL),
    (14.20, InteractiveCommand.GROUP_BANK),
    (14.40, InteractiveCommand.LOOP_MODE),
    (14.60, InteractiveCommand.TEMPO_UP),
    (14.80, InteractiveCommand.EXECUTE),
    (18.00, InteractiveCommand.RANGE_DOWN),
    (18.20, InteractiveCommand.RANGE_UP),
    (18.40, InteractiveCommand.SCALE_DOWN),
    (18.60, InteractiveCommand.TEMPO_DOWN),
    (21.00, InteractiveCommand.EXECUTE),
    (24.20, InteractiveCommand.NEXT),
    (24.40, InteractiveCommand.EXECUTE),
    (27.40, InteractiveCommand.NEUTRAL),
    (30.60, InteractiveCommand.STATIC_MODE),
    (30.80, InteractiveCommand.SINGLE_BANK),
    (31.00, InteractiveCommand.PREVIOUS),
    (31.20, InteractiveCommand.EXECUTE),
    (34.50, InteractiveCommand.NEUTRAL),
    (37.70, InteractiveCommand.ABORT),
)


def print_controls() -> None:
    print(
        "Interactive suspended calibration (numeric keypad only):\n"
        "  Num4/Num6       previous/next selection\n"
        "  Num1/Num3       single-joint/group bank\n"
        "  Num-/Num+       static/trajectory-loop mode\n"
        "  Num2/Num8       smaller/larger range preset\n"
        "  Num7/Num9       amplitude scale -/+\n"
        "  NumDivide/*     tempo scale -/+\n"
        "  Num5            capture/recapture neutral (only while fully neutral)\n"
        "  Numpad Enter    execute next static phase or start/stop loop\n"
        "  Num0            return smoothly to neutral\n"
        "  NumDecimal      abort (simulation stops after reaching neutral)\n"
        "MuJoCo camera keys and mouse remain unchanged."
    )


def draw_selection(viewer, plant: SuspendedMujocoPlant, controller) -> None:
    scene = viewer.user_scn
    scene.ngeom = 0
    for index, motor in enumerate(controller.selection.motors):
        if scene.ngeom >= scene.maxgeom:
            break
        color = np.asarray([1.0, 0.25 + 0.12 * (index % 4), 0.1, 0.9], np.float32)
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.asarray([0.025, 0.0, 0.0]),
            plant.joint_anchor(motor),
            np.eye(3).reshape(-1),
            color,
        )
        scene.ngeom += 1


def write_recording(output: Path, rows: dict[str, list], events: list[dict], summary: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    with (output / "state.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    with (output / "events.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(events[0]))
        writer.writeheader()
        writer.writerows(events)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scripted-demo", action="store_true")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="optional wall/sim duration; 0 means until abort/window close",
    )
    parser.add_argument("--max-envelope", choices=("screening", "full"), default="screening")
    parser.add_argument("--fault-profile", type=Path)
    parser.add_argument("--calibration-profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, default=REPO / "sonic_cpp/g1/scene_23dof.xml")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/g1_23dof_sonic.yaml")
    args = parser.parse_args()
    if not args.view and not args.scripted_demo and args.duration <= 0:
        raise SystemExit("headless interactive mode needs --scripted-demo or --duration")
    if args.duration < 0:
        raise SystemExit("--duration must be nonnegative")

    plant = SuspendedMujocoPlant(
        mjcf=args.mjcf,
        config=args.config,
        fault_profile=args.fault_profile,
        calibration_profile=args.calibration_profile,
    )
    controller = InteractiveCalibrationController(max_envelope=args.max_envelope)
    command_queue: queue.SimpleQueue[InteractiveCommand] = queue.SimpleQueue()

    def key_callback(key: int) -> None:
        command = KEY_COMMANDS.get(key)
        if command is not None:
            command_queue.put(command)

    viewer = None
    if args.view:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(
            plant.model,
            plant.data,
            key_callback=key_callback,
            show_left_ui=True,
            show_right_ui=True,
        )
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -10
        viewer.cam.azimuth = 135
        print_controls()
    print("[CAL] " + controller.status())

    rows: dict[str, list] = {
        "time_s": [],
        "target_model": [],
        "command_hardware": [],
        "effective_model_target": [],
        "q_measured": [],
        "dq_measured": [],
        "tau_est": [],
        "command_offset": [],
        "captured_neutral": [],
        "bank": [],
        "selection": [],
        "mode": [],
        "range": [],
        "scale": [],
        "tempo": [],
        "running": [],
    }
    events: list[dict] = []
    events.append(
        {
            "time_s": "0.000000",
            "command": "initial",
            **controller.status_dict(),
        }
    )
    demo_index = 0
    start_wall = time.perf_counter()
    sim_time = 0.0
    captured_neutral = plant.neutral.copy()
    while True:
        if viewer is not None and not viewer.is_running():
            command_queue.put(InteractiveCommand.ABORT)
        if args.scripted_demo:
            while demo_index < len(SCRIPTED_DEMO) and sim_time >= SCRIPTED_DEMO[demo_index][0]:
                command_queue.put(SCRIPTED_DEMO[demo_index][1])
                demo_index += 1
        while not command_queue.empty():
            command = command_queue.get()
            if command == InteractiveCommand.CAPTURE_NEUTRAL and controller.can_capture():
                captured_neutral = np.asarray(
                    plant.data.qpos[plant.qadr],
                    dtype=np.float64,
                ).copy()
            status = controller.apply(command)
            print(f"[CAL {sim_time:7.2f}s] {command.value}: {status}")
            events.append(
                {
                    "time_s": f"{sim_time:.6f}",
                    "command": command.value,
                    **controller.status_dict(),
                }
            )
        offset = controller.step()
        state = plant.step(captured_neutral + offset)
        status = controller.status_dict()
        rows["time_s"].append(sim_time)
        for key in (
            "target_model",
            "command_hardware",
            "effective_model_target",
            "q_measured",
            "dq_measured",
            "tau_est",
        ):
            rows[key].append(state[key])
        rows["command_offset"].append(offset)
        rows["captured_neutral"].append(captured_neutral)
        for key in ("bank", "selection", "mode", "range", "scale", "tempo", "running"):
            rows[key].append(status[key])
        if viewer is not None:
            draw_selection(viewer, plant, controller)
            viewer.sync()
        sim_time += controller.dt
        if controller.aborted and np.max(np.abs(offset)) < 1e-5:
            break
        if args.duration > 0 and sim_time >= args.duration:
            controller.apply(InteractiveCommand.ABORT)
        if not args.scripted_demo and args.view:
            target_wall = start_wall + sim_time
            remaining = target_wall - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
        if not args.view and args.duration > 0 and sim_time > args.duration + 10:
            raise RuntimeError("interactive controller did not return neutral after abort")

    if viewer is not None:
        viewer.close()
    error = np.asarray(rows["command_hardware"]) - np.asarray(rows["q_measured"])
    final_max_abs_offset = float(np.max(np.abs(rows["command_offset"][-1])))
    summary = {
        "schema": "e_skin.interactive_suspended_calibration_mujoco.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mjcf": str(args.mjcf.expanduser().absolute()),
        "config": str(args.config.expanduser().absolute()),
        "duration_s": sim_time,
        "samples": len(rows["time_s"]),
        "events": len(events),
        "scripted_demo": args.scripted_demo,
        "max_envelope": args.max_envelope,
        "captured_neutral_rad": captured_neutral.tolist(),
        "fault_profile": (
            str(args.fault_profile.expanduser().absolute()) if args.fault_profile else None
        ),
        "calibration_profile": (
            str(args.calibration_profile.expanduser().absolute())
            if args.calibration_profile
            else None
        ),
        "max_abs_command_offset_rad": float(np.max(np.abs(rows["command_offset"]))),
        "max_abs_tracking_error_rad": float(np.max(np.abs(error[:, PRESENT_23]))),
        "final_offset_norm_rad": float(np.linalg.norm(rows["command_offset"][-1])),
        "final_max_abs_offset_rad": final_max_abs_offset,
        "aborted_to_neutral": bool(controller.aborted and final_max_abs_offset < 1e-5),
    }
    write_recording(args.output.expanduser().absolute(), rows, events, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
