#!/usr/bin/env python3
"""Execute the suspended calibration plan against actual MuJoCo dynamics.

The free root is held by an idealized pelvis harness while gravity, link
inertia, actuator force limits, official SONIC Kp/Kd, and joint dynamics remain
active.  Optional deterministic actuator faults let the inverse-calibration
path be tested before a physical robot is involved.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sonic.calibration.joint_mapping import JointCalibrationProfile
from sonic.calibration.mujoco_plant import set_pd
from sonic.calibration.protocol import MOTOR_NAMES, PRESENT_23, build_plan, load_plan, save_plan
from sonic.model.robot_profile import MUJOCO_JOINT_NAMES


def _load_faults(path: Path | None) -> dict[int, dict[str, float]]:
    if path is None:
        return {}
    payload = json.loads(path.expanduser().read_text())
    if payload.get("schema") != "e_skin.synthetic_joint_faults.v1":
        raise ValueError("unsupported fault profile")
    out = {}
    for key, value in payload.get("joints", {}).items():
        params = {
            "gain": float(value.get("gain", 1.0)),
            "bias_rad": float(value.get("bias_rad", 0.0)),
            "direction_bias_rad": float(value.get("direction_bias_rad", 0.0)),
            "lag_s": float(value.get("lag_s", 0.0)),
        }
        if params["gain"] <= 0 or params["lag_s"] < 0:
            raise ValueError(f"invalid fault parameters for motor {key}")
        out[int(key)] = params
    return out




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--protocol", choices=("screening", "full"), default="screening")
    parser.add_argument("--scope", choices=("legs", "all"), default="all")
    parser.add_argument("--fault-profile", type=Path)
    parser.add_argument("--calibration-profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mjcf", type=Path,
        default=REPO / "sonic_cpp/g1/scene_23dof.xml",
    )
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/g1_23dof_sonic.yaml",
    )
    args = parser.parse_args()

    args.output = args.output.expanduser().absolute()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = load_plan(args.plan) if args.plan else build_plan(
        protocol=args.protocol, scope=args.scope
    )
    save_plan(plan, args.output / "plan.csv")
    profile = (
        JointCalibrationProfile.load(args.calibration_profile)
        if args.calibration_profile else None
    )
    faults = _load_faults(args.fault_profile)

    cfg = yaml.safe_load(args.config.expanduser().read_text())
    default_29 = np.asarray(cfg["default_angles"], dtype=np.float64)
    kp_29 = np.asarray(cfg["kps"], dtype=np.float64)
    kd_29 = np.asarray(cfg["kds"], dtype=np.float64)

    model = mujoco.MjModel.from_xml_path(str(args.mjcf.expanduser()))
    data = mujoco.MjData(model)
    joint_names = [MUJOCO_JOINT_NAMES[i] for i in PRESENT_23]
    joint_ids = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in joint_names
    ])
    if np.any(joint_ids < 0):
        raise RuntimeError("23-DoF MJCF is missing a physical joint")
    qadr = model.jnt_qposadr[joint_ids]
    vadr = model.jnt_dofadr[joint_ids]
    actuator_by_joint = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
    actuators = np.array([actuator_by_joint[int(j)] for j in joint_ids])
    for motor, joint_id, actuator in zip(PRESENT_23, joint_ids, actuators):
        effort = float(model.jnt_actfrcrange[joint_id, 1]) \
            if model.jnt_actfrclimited[joint_id] else 0.0
        set_pd(model, int(actuator), kp_29[motor], kd_29[motor], effort)

    fixed = np.setdiff1d(np.arange(29), PRESENT_23)
    for motor in fixed:
        name = MUJOCO_JOINT_NAMES[motor]
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or joint_id not in actuator_by_joint:
            continue
        actuator = actuator_by_joint[joint_id]
        set_pd(model, actuator, kp_29[motor], kd_29[motor], 0.0)
        data.ctrl[actuator] = default_29[motor]

    neutral_model = default_29.copy()
    root_position = np.array([0.0, 0.0, 1.20])
    root_quat = np.array([1.0, 0.0, 0.0, 0.0])
    data.qpos[:3] = root_position
    data.qpos[3:7] = root_quat
    data.qpos[qadr] = neutral_model[PRESENT_23]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    decimation = max(1, round(.02 / model.opt.timestep))
    filtered_effective = neutral_model.copy()
    previous_desired = neutral_model.copy()
    rows: list[list[object]] = []
    header = ["schema", "sample", "time_s", "phase", "active_motor", "direction"]
    for prefix in ("target_model", "command_hardware", "effective_model_target",
                   "q_measured", "dq_measured", "tau_est"):
        header.extend(f"{prefix}_{i}" for i in range(29))

    for sample in range(len(plan.time_s)):
        desired = neutral_model + plan.offsets[sample]
        direction = np.sign(desired - previous_desired)
        command = desired if profile is None else profile.apply_29(desired, direction)
        effective = command.copy()
        for motor, fault in faults.items():
            delta = command[motor] - neutral_model[motor]
            effective[motor] = (
                neutral_model[motor] + fault["gain"] * delta + fault["bias_rad"]
                + np.sign(direction[motor]) * fault["direction_bias_rad"]
            )
            tau = fault["lag_s"]
            alpha = 1.0 if tau <= 0 else .02 / (tau + .02)
            filtered_effective[motor] += alpha * (
                effective[motor] - filtered_effective[motor]
            )
            effective[motor] = filtered_effective[motor]
        for motor, actuator in zip(PRESENT_23, actuators):
            data.ctrl[actuator] = effective[motor]
        for _ in range(decimation):
            # Ideal pelvis harness: it supplies root wrench but does not alter
            # any joint dynamics or gravity loads.
            data.qpos[:3] = root_position
            data.qpos[3:7] = root_quat
            data.qvel[:6] = 0.0
            mujoco.mj_step(model, data)
        q29 = np.full(29, np.nan)
        dq29 = np.full(29, np.nan)
        tau29 = np.full(29, np.nan)
        q29[PRESENT_23] = data.qpos[qadr]
        dq29[PRESENT_23] = data.qvel[vadr]
        tau29[PRESENT_23] = data.actuator_force[actuators]
        rows.append([
            "e_skin.suspended_calibration_log.v1", sample,
            f"{plan.time_s[sample]:.9f}", plan.phase[sample],
            int(plan.active_motor[sample]), int(plan.direction[sample]),
            *desired, *command, *effective, *q29, *dq29, *tau29,
        ])
        previous_desired = desired

    with (args.output / "state.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

    # Error is evaluated only for the one active joint, which prevents held
    # gravity errors in unrelated joints from diluting the identification.
    state = np.asarray([[float(value) for value in row[6:]] for row in rows])
    target = state[:, :29]
    measured = state[:, 29 * 3:29 * 4]
    active = plan.active_motor
    valid = active >= 0
    error = measured[np.arange(len(active))[valid], active[valid]] \
        - target[np.arange(len(active))[valid], active[valid]]
    phases = plan.phase[valid]
    summary = {
        "schema": "e_skin.suspended_calibration_mujoco_summary.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mjcf": str(args.mjcf.expanduser().absolute()),
        "config": str(args.config.expanduser().absolute()),
        "protocol": plan.protocol,
        "scope": plan.scope,
        "samples": len(plan.time_s),
        "duration_s": plan.duration_s,
        "fault_profile": str(args.fault_profile.expanduser().absolute()) if args.fault_profile else None,
        "calibration_profile": str(args.calibration_profile.expanduser().absolute()) if args.calibration_profile else None,
        "active_rmse_rad": float(np.sqrt(np.mean(error**2))),
        "dwell_rmse_rad": float(np.sqrt(np.mean(error[phases == "dwell"]**2))),
        "dynamic_rmse_rad": float(np.sqrt(np.mean(error[np.char.startswith(phases, "sine_")]**2))),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
