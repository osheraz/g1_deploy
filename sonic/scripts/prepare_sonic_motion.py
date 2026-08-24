#!/usr/bin/env python3
"""Create conservative 50 Hz SONIC motion references for real deployment.

Produces two finite clips:

* ``motion_neutral_default.npz`` — exact configured 29-DoF default pose,
  upright pelvis, zero velocity. This is the first hardware policy test.
* ``motion_walk11_neutral_start.npz`` — the smallest frame-0 joint gap among
  the available walking clips that stay below 10 degrees pelvis tilt, prefixed
  with a smooth default-to-frame-0 reference transition. Use only after the
  neutral hold passes.

The G1 encoder consumes joint positions/velocities and pelvis orientation. The
other copied body transforms remain present for schema compatibility but are
masked out in G1 mode by the official observation configuration.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


MJ_TO_IL = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
     16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
)
FPS = 50.0


def normalize_quat(q):
    q = np.asarray(q, np.float64)
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)


def heading_quat(q):
    w, x, y, z = np.asarray(q, np.float64)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def slerp(q0, q1, amount):
    q0, q1 = normalize_quat(q0), normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        return normalize_quat((1 - amount) * q0 + amount * q1)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    return (
        np.sin((1 - amount) * theta) / np.sin(theta) * q0
        + np.sin(amount * theta) / np.sin(theta) * q1
    )


def finite_difference(q):
    dq = np.zeros_like(q)
    if len(q) > 1:
        dq[:-1] = np.diff(q, axis=0) * FPS
        dq[-1] = dq[-2]
    return dq


def root_tilt_deg(quat):
    w, x, y, z = np.asarray(quat).T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return np.degrees(np.sqrt(roll * roll + pitch * pitch))


def save_clip(path, *, q, body_pos, body_quat, default_il, source, preparation):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dof_il=np.asarray(q, np.float32),
        dof_vel_il=np.asarray(finite_difference(q), np.float32),
        body_pos_il=np.asarray(body_pos, np.float32),
        body_quat_wxyz_il=np.asarray(normalize_quat(body_quat), np.float32),
        source=np.array([source]),
        preparation=np.array([preparation]),
        fps=np.array([FPS], np.float32),
        recommended_loop=np.array([False]),
    )
    print(
        f"wrote {path}: {len(q) / FPS:.1f}s, "
        f"start gap={np.degrees(np.max(np.abs(q[0] - default_il))):.1f}deg, "
        f"max pelvis tilt={root_tilt_deg(body_quat[:, 0]).max():.1f}deg"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/g1_23dof_sonic.yaml")
    ap.add_argument("--seed", default="sonic/model/motion_stand.npz")
    ap.add_argument("--walk-source", default="sonic/model/walk11.npz")
    ap.add_argument("--neutral-out", default="sonic/model/motion_neutral_default.npz")
    ap.add_argument("--walk-out", default="sonic/model/motion_walk11_neutral_start.npz")
    ap.add_argument("--neutral-seconds", type=float, default=6.0)
    ap.add_argument("--transition-seconds", type=float, default=3.0)
    ap.add_argument("--max-walk-tilt-deg", type=float, default=10.0)
    args = ap.parse_args()

    with Path(args.config).open() as handle:
        config = yaml.safe_load(handle)
    default_mj = np.asarray(config["default_angles"], np.float64)
    default_il = default_mj[MJ_TO_IL]
    default_height = float(config.get("sonic_planner_default_height", 0.788740))

    seed = np.load(args.seed)
    n_neutral = max(2, int(round(args.neutral_seconds * FPS)))
    q_neutral = np.repeat(default_il[None], n_neutral, axis=0)
    pos_neutral = np.repeat(seed["body_pos_il"][:1], n_neutral, axis=0).astype(np.float64)
    quat_neutral = np.repeat(seed["body_quat_wxyz_il"][:1], n_neutral, axis=0).astype(np.float64)
    pos_neutral[:, 0, :2] = 0.0
    pos_neutral[:, 0, 2] = default_height
    quat_neutral[:, 0] = heading_quat(quat_neutral[0, 0])
    save_clip(
        args.neutral_out,
        q=q_neutral,
        body_pos=pos_neutral,
        body_quat=quat_neutral,
        default_il=default_il,
        source="synthetic_configured_default_pose",
        preparation="finite neutral hold; exact configured default q; upright root",
    )

    src = np.load(args.walk_source)
    tilt = root_tilt_deg(src["body_quat_wxyz_il"][:, 0])
    if float(tilt.max()) > args.max_walk_tilt_deg:
        raise SystemExit(
            f"refusing {args.walk_source}: pelvis tilt {tilt.max():.1f}deg exceeds "
            f"{args.max_walk_tilt_deg:.1f}deg"
        )
    n = max(2, int(round(args.transition_seconds * FPS)))
    u = np.linspace(0.0, 1.0, n + 1)
    smooth = 10 * u**3 - 15 * u**4 + 6 * u**5
    q_prefix = (
        (1 - smooth[:, None]) * default_il[None]
        + smooth[:, None] * src["dof_il"][0][None]
    )
    pos_prefix = np.repeat(src["body_pos_il"][:1], n + 1, axis=0).astype(np.float64)
    pos_prefix[:, 0, 2] = (
        (1 - smooth) * default_height + smooth * float(src["body_pos_il"][0, 0, 2])
    )
    quat_prefix = np.repeat(src["body_quat_wxyz_il"][:1], n + 1, axis=0).astype(np.float64)
    q0_root = heading_quat(src["body_quat_wxyz_il"][0, 0])
    q1_root = src["body_quat_wxyz_il"][0, 0]
    for i, amount in enumerate(smooth):
        quat_prefix[i, 0] = slerp(q0_root, q1_root, float(amount))

    q_walk = np.concatenate([q_prefix, src["dof_il"][1:]], axis=0)
    pos_walk = np.concatenate([pos_prefix, src["body_pos_il"][1:]], axis=0)
    quat_walk = np.concatenate([quat_prefix, src["body_quat_wxyz_il"][1:]], axis=0)
    source_name = str(src["source"][0]) if "source" in src.files else Path(args.walk_source).name
    save_clip(
        args.walk_out,
        q=q_walk,
        body_pos=pos_walk,
        body_quat=quat_walk,
        default_il=default_il,
        source=source_name,
        preparation=(
            f"{args.transition_seconds:g}s quintic configured-default-to-source-frame0 "
            "transition; 50Hz finite-difference velocities; finite playback"
        ),
    )
    print(
        f"walk start physical gap is 0.0deg; source max pelvis tilt={tilt.max():.1f}deg. "
        "Run neutral hold first and do not loop this clip."
    )


if __name__ == "__main__":
    main()
