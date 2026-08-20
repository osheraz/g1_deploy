#!/usr/bin/env python3
"""Legacy helper: extract one cyclic gait template from a walking clip.

Withdrawn. Builds npz files for the unused hybrid-gait attempt
(``policy/gait_reference.py`` / ``--gait-templates``). That path is not the
current slow-walk method; keep official Slow Walk 0.4 and use sagittal
scale / ``--slow-gait``.

    python scripts/extract_gait_cycle.py --clip policy/sonic/rand_22.npz \
        --out policy/sonic/gait_templates/slow_walk.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))

from policy.pico_reference import LOWER_BODY_IL_IDX  # noqa: E402

FPS = 50.0
PHASE_SAMPLES = 120


def _yaw(q):
    w, x, y, z = np.asarray(q, np.float64).T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trim", type=float, default=0.15,
                    help="fraction trimmed from each end before cycle search")
    args = ap.parse_args()

    d = np.load(Path(args.clip).expanduser())
    dof = np.asarray(d["dof_il"], np.float64)          # (T,29)
    dvel = np.asarray(d["dof_vel_il"], np.float64)
    root_p = np.asarray(d["body_pos_il"], np.float64)[:, 0]     # (T,3)
    root_q = np.asarray(d["body_quat_wxyz_il"], np.float64)[:, 0]
    T = len(dof)
    lo, hi = int(T * args.trim), int(T * (1 - args.trim))
    steady = slice(lo, hi)

    # Cycle period from autocorrelation of the mean-centred leg signals.
    legs = dof[steady][:, LOWER_BODY_IL_IDX]
    sig = legs - legs.mean(axis=0)
    ac = sum(np.correlate(sig[:, j], sig[:, j], mode="full") for j in range(sig.shape[1]))
    ac = ac[len(ac) // 2:]
    # first local max after the first zero crossing of the derivative
    dmin = int(0.5 * FPS)                                # >= 0.5 s cycles
    dmax = min(int(3.0 * FPS), len(ac) - 2)
    span = ac[dmin:dmax]
    period_frames = dmin + int(np.argmax(span))
    period_s = period_frames / FPS

    # Anchor phase 0 at max left-hip-pitch (IL 0) within the steady region.
    lhip = dof[steady][:, 0]
    starts = np.arange(0, len(lhip) - period_frames)
    anchor = lo + int(starts[np.argmax(lhip[starts])]) if len(starts) else lo
    cyc = slice(anchor, anchor + period_frames + 1)

    # Heading-frame root motion over the cycle.
    yaw = _yaw(root_q[cyc])
    c, s = np.cos(-yaw[0]), np.sin(-yaw[0])
    dpos = root_p[cyc] - root_p[cyc][0]
    fwd = c * dpos[:, 0] - s * dpos[:, 1]
    lat = s * dpos[:, 0] + c * dpos[:, 1]
    stride_m = float(fwd[-1])

    # Resample to a fixed phase grid; enforce cyclicity with an end-blend.
    src = np.linspace(0.0, 1.0, period_frames + 1)
    dst = np.linspace(0.0, 1.0, PHASE_SAMPLES, endpoint=False)

    def resample(x):
        x = np.asarray(x, np.float64)
        flat = x.reshape(len(x), -1)
        out = np.stack([np.interp(dst, src, flat[:, j]) for j in range(flat.shape[1])],
                       axis=1)
        return out.reshape((PHASE_SAMPLES,) + x.shape[1:])

    q_cycle = resample(dof[cyc][:, LOWER_BODY_IL_IDX])          # (P,12)
    dq_cycle = resample(dvel[cyc][:, LOWER_BODY_IL_IDX])
    z_cycle = resample(root_p[cyc][:, 2])
    yaw_cycle = resample(np.unwrap(yaw) - yaw[0])

    # loop blend: cross-fade the last 10% into the first sample
    k = PHASE_SAMPLES // 10
    w = np.linspace(0, 1, k)[:, None]
    for arr in (q_cycle, dq_cycle):
        arr[-k:] = (1 - w) * arr[-k:] + w * arr[0]
    z_cycle[-k:] = (1 - w[:, 0]) * z_cycle[-k:] + w[:, 0] * z_cycle[0]

    stand_q = np.asarray(d["dof_il"], np.float64)[0][LOWER_BODY_IL_IDX]

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        q_cycle=q_cycle.astype(np.float32),
        dq_cycle=dq_cycle.astype(np.float32),
        root_z_cycle=z_cycle.astype(np.float32),
        yaw_cycle=yaw_cycle.astype(np.float32),
        lower_il_idx=np.asarray(LOWER_BODY_IL_IDX, np.int64),
        stand_q=stand_q.astype(np.float32),
        stride_m=np.float32(stride_m),
        period_s=np.float32(period_s),
        lateral_max_m=np.float32(np.abs(lat).max()),
        meta=np.bytes_(json.dumps({
            "source": str(args.clip),
            "phase_samples": PHASE_SAMPLES,
            "speed_mps": stride_m / period_s,
        })),
    )
    print(f"{out.name}: period {period_s:.2f}s, stride {stride_m:.3f}m, "
          f"natural speed {stride_m / period_s:.3f} m/s, "
          f"root z {z_cycle.min():.3f}..{z_cycle.max():.3f}")


if __name__ == "__main__":
    main()
