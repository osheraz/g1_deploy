#!/usr/bin/env python3
"""Export one motion from a SONIC tracker pack to a plain .npz for deploy.

    python scripts/motion/export_sonic_motion.py --pack <pack.pt> --motion 16 \
        --out sonic/model/motion_clip.npz
    python scripts/motion/export_sonic_motion.py --pack <pack.pt> --list

The pack stores 33 bodies in the g1_bm order and 29 dofs in MuJoCo order. The
tracker wants 14 bodies and 29 dofs in IsaacLab order, quaternions wxyz, 50 fps.
This does that reordering and resampling once, offline, so the deploy only ever
loads numpy.

Name lists are taken verbatim from sonic/motion/bank.py and sonic/robot_g1.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_SUF = "_joint"

IL_DOF = [n + _SUF for n in [
    "left_hip_pitch", "right_hip_pitch", "waist_yaw", "left_hip_roll", "right_hip_roll",
    "waist_roll", "left_hip_yaw", "right_hip_yaw", "waist_pitch", "left_knee", "right_knee",
    "left_shoulder_pitch", "right_shoulder_pitch", "left_ankle_pitch", "right_ankle_pitch",
    "left_shoulder_roll", "right_shoulder_roll", "left_ankle_roll", "right_ankle_roll",
    "left_shoulder_yaw", "right_shoulder_yaw", "left_elbow", "right_elbow",
    "left_wrist_roll", "right_wrist_roll", "left_wrist_pitch", "right_wrist_pitch",
    "left_wrist_yaw", "right_wrist_yaw"]]

MJ_DOF = [n + _SUF for n in [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch",
    "left_ankle_roll", "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll", "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw", "right_shoulder_pitch",
    "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
    "right_wrist_pitch", "right_wrist_yaw"]]

BM33 = [
    "pelvis", "head", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link",
    "right_ankle_pitch_link", "right_ankle_roll_link", "waist_yaw_link", "waist_roll_link",
    "torso_link", "left_shoulder_pitch_link", "left_shoulder_roll_link",
    "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link",
    "left_wrist_pitch_link", "left_wrist_yaw_link", "left_rubber_hand",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link",
    "right_wrist_yaw_link", "right_rubber_hand"]

BODY_NAMES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"]

FPS_OUT = 50.0


def _sample_grid(n_src: int, fps_src: float, fps_out: float):
    """Official C++ grid: floor(duration*out_fps), sampled from t=0."""
    n_out = max(int(np.floor(n_src / fps_src * fps_out)), 1)
    src_f = np.arange(n_out, dtype=np.float64) / fps_out * fps_src
    f0 = np.minimum(np.floor(src_f).astype(np.int64), n_src - 1)
    f1 = np.minimum(f0 + 1, n_src - 1)
    return f0, f1, src_f - np.floor(src_f)


def _resample(arr: np.ndarray, fps_src: float, fps_out: float) -> np.ndarray:
    """Linear position resampling on the same grid as the official planner."""
    n_src = arr.shape[0]
    if abs(fps_src - fps_out) < 1e-6 or n_src < 2:
        return arr.copy()
    f0, f1, w = _sample_grid(n_src, fps_src, fps_out)
    shape = (len(w),) + (1,) * (arr.ndim - 1)
    return (1.0 - w.reshape(shape)) * arr[f0] + w.reshape(shape) * arr[f1]


def _resample_quat(arr: np.ndarray, fps_src: float, fps_out: float) -> np.ndarray:
    """Shortest-path WXYZ SLERP, matching ``quat_slerp_d`` in C++."""
    n_src = arr.shape[0]
    if abs(fps_src - fps_out) < 1e-6 or n_src < 2:
        out = arr.copy()
        return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-8)
    f0, f1, w = _sample_grid(n_src, fps_src, fps_out)
    q0 = arr[f0].astype(np.float64)
    q1 = arr[f1].astype(np.float64)
    q0 /= np.maximum(np.linalg.norm(q0, axis=-1, keepdims=True), 1e-8)
    q1 /= np.maximum(np.linalg.norm(q1, axis=-1, keepdims=True), 1e-8)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    amount = w.reshape((len(w),) + (1,) * (arr.ndim - 1))
    curved = (
        np.sin((1.0 - amount) * theta) / np.where(sin_theta < 1e-8, 1.0, sin_theta) * q0
        + np.sin(amount * theta) / np.where(sin_theta < 1e-8, 1.0, sin_theta) * q1
    )
    linear = (1.0 - amount) * q0 + amount * q1
    out = np.where(dot > 0.9995, linear, curved)
    return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-8)


def _finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    if len(values) > 1:
        out[:-1] = np.diff(values, axis=0) * fps
        out[-1] = out[-2]
    return out


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--motion", type=int, default=None, help="index within the pack")
    ap.add_argument("--out", default="sonic/model/motion_clip.npz")
    ap.add_argument("--list", action="store_true", help="list motions and exit")
    ap.add_argument("--freeze", type=float, default=None, metavar="SECONDS",
                    help="hold frame 0 for this long instead of playing the motion")
    args = ap.parse_args()

    d = torch.load(Path(args.pack).expanduser(), map_location="cpu", weights_only=False)
    names = [Path(str(f)).name for f in d["motion_files"]]
    starts = d["length_starts"].numpy()
    nframes = d["motion_num_frames"].numpy()
    dts = d["motion_dt"].numpy()

    if args.list or args.motion is None:
        for i, (n, f, dt) in enumerate(zip(names, nframes, dts)):
            print(f"{i:3d}  {f:5d}f  {f * dt:6.1f}s  {n}")
        if args.motion is None:
            raise SystemExit("\npick one with --motion N")
        return

    i = args.motion
    s, T = int(starts[i]), int(nframes[i])
    e = s + T
    fps_src = 1.0 / float(dts[i])
    print(f"motion {i}: {names[i]}  {T} frames @ {fps_src:.1f} fps")

    body_idx = np.array([BM33.index(b) for b in BODY_NAMES])
    dof_idx = np.array([MJ_DOF.index(n) for n in IL_DOF])

    dof = d["dps"].numpy()[s:e][:, dof_idx]                 # (T,29) IsaacLab order
    # Source velocities are intentionally not interpolated. The official
    # planner recomputes them from the resampled 50 Hz joint positions.
    bpos = d["gts"].numpy()[s:e][:, body_idx]               # (T,14,3)
    brot_xyzw = d["grs"].numpy()[s:e][:, body_idx]          # (T,14,4)
    bquat = np.concatenate([brot_xyzw[..., 3:4], brot_xyzw[..., :3]], axis=-1)

    dof = _resample(dof, fps_src, FPS_OUT)
    dvel = _finite_difference(dof, FPS_OUT)
    bpos = _resample(bpos, fps_src, FPS_OUT)
    bquat = _resample_quat(bquat, fps_src, FPS_OUT)

    if args.freeze:
        n = max(1, int(args.freeze * FPS_OUT))
        dof = np.repeat(dof[:1], n, axis=0)
        dvel = np.zeros_like(dof)
        bpos = np.repeat(bpos[:1], n, axis=0)
        bquat = np.repeat(bquat[:1], n, axis=0)
        print(f"frozen on frame 0 for {args.freeze:.1f}s")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        dof_il=dof.astype(np.float32),
        dof_vel_il=dvel.astype(np.float32),
        body_pos_il=bpos.astype(np.float32),
        body_quat_wxyz_il=bquat.astype(np.float32),
        source=np.array([names[i]]),
        fps=np.array([FPS_OUT]),
    )
    print(f"wrote {out}")
    print(f"  dof_il            {dof.shape}")
    print(f"  body_pos_il       {bpos.shape}")
    print(f"  {len(dof)} frames @ {FPS_OUT:g} fps = {len(dof) / FPS_OUT:.1f}s")
    print(f"  root height       {bpos[:, 0, 2].min():.3f} .. {bpos[:, 0, 2].max():.3f} m")
    print(f"  dof range         {dof.min():+.3f} .. {dof.max():+.3f} rad")


if __name__ == "__main__":
    main()
