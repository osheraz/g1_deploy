"""Reference-window sources for the SONIC tracker.

The encoder needs, every tick, a window describing what the robot should be
doing: 10 future frames of joint targets plus current body poses. Two sources,
matching the two deploy modes:

    MotionReferenceWindow             replay an exported clip (standalone)
    policy/pico_reference.py          live 3-point VR teleop

Both return the dict ``sonic.policy.obs.build_encoder_obs`` expects, batched
with a leading dimension of 1. Term shapes and the future stride are taken from
``sonic/motion/bank.py`` in the research repo and must stay identical.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Future window: 10 frames, stride 5 for the G1 terms; stride 1 for the smpl
# wrist term. Same as sonic/motion/bank.py.
G1_N, G1_STEP = 10, 5
SMPL_N, SMPL_STEP = 10, 1

# Wrist IsaacLab dof indices used by the smpl term (== obs.WRIST_IL_IDX).
WRIST_IL_IDX = np.array([23, 24, 25, 26, 27, 28])


class MotionReferenceWindow:
    """Replay an exported motion clip.

    Expects an .npz with the arrays ``MotionBank`` holds, already in IsaacLab
    order and resampled to 50 fps:

        dof_il              (T,29)
        dof_vel_il          (T,29)
        body_pos_il         (T,14,3)
        body_quat_wxyz_il   (T,14,4)

    Export one with ``scripts/export_sonic_motion.py`` -- that step needs the
    research repo's training stack, which is why it is offline and this class
    is not.
    """

    def __init__(self, clip_path, loop: bool = True):
        path = Path(clip_path).expanduser()
        if not path.exists():
            raise SystemExit(
                f"sonic motion clip not found: {path}\n"
                "Export one first:  python scripts/export_sonic_motion.py "
                "--pack <pack.pt> --out <clip.npz>"
            )
        d = np.load(path)
        missing = [k for k in ("dof_il", "dof_vel_il", "body_pos_il",
                               "body_quat_wxyz_il") if k not in d.files]
        if missing:
            raise SystemExit(f"{path} is missing {missing}; re-export it.")

        self.dof_il = d["dof_il"].astype(np.float64)
        self.dof_vel_il = d["dof_vel_il"].astype(np.float64)
        self.body_pos_il = d["body_pos_il"].astype(np.float64)
        self.body_quat_wxyz_il = d["body_quat_wxyz_il"].astype(np.float64)
        self.T = len(self.dof_il)
        self.loop = loop
        self.frame = 0
        print(f"SONIC reference: motion clip {path.name}, {self.T} frames "
              f"({self.T / 50.0:.1f}s @50fps){' looping' if loop else ''}")

    def reset(self) -> None:
        self.frame = 0

    def step(self) -> None:
        self.frame += 1
        if self.frame >= self.T:
            self.frame = 0 if self.loop else self.T - 1

    def _idx(self, n: int, step: int) -> np.ndarray:
        return np.clip(self.frame + np.arange(n) * step, 0, self.T - 1)

    def window(self) -> dict:
        gi = self._idx(G1_N, G1_STEP)
        si = self._idx(SMPL_N, SMPL_STEP)
        ci = min(self.frame, self.T - 1)
        return {
            "dof_fut": self.dof_il[gi][None],                       # (1,10,29)
            "dvel_fut": self.dof_vel_il[gi][None],                  # (1,10,29)
            "rootq_fut": self.body_quat_wxyz_il[gi, 0][None],       # (1,10,4)
            "rootz_fut": self.body_pos_il[gi, 0, 2][None],          # (1,10)
            "body_pos_cur": self.body_pos_il[ci][None],             # (1,14,3)
            "body_quat_cur": self.body_quat_wxyz_il[ci][None],      # (1,14,4)
            "rootz_cur": np.array([self.body_pos_il[ci, 0, 2]]),    # (1,)
            "wrist_dof_smpl": self.dof_il[si][:, WRIST_IL_IDX][None],  # (1,10,6)
        }


# Live teleop lives in policy/pico_reference.py -- it needs the SONIC kinematic
# planner, whose ONNX is ~774 MB and therefore is not vendored here.


def build_reference(config, mode: str):
    """``mode`` is 'motion' or 'pico'."""
    if mode == "motion":
        return MotionReferenceWindow(
            getattr(config, "sonic_motion_clip", "policy/sonic/motion_clip.npz"),
            loop=getattr(config, "sonic_motion_loop", True),
        )
    if mode == "pico":
        from common.pico_client import PicoClient
        from policy.pico_reference import PicoReferenceWindow

        client = PicoClient(
            getattr(config, "pico_server_ip", "127.0.0.1"),
            getattr(config, "pico_server_port", 5560),
        )
        client.start()
        print(f"teleop:  tcp://{client.server_address}:{client.port}")
        return PicoReferenceWindow(config, client)
    raise SystemExit(f"unknown sonic mode {mode!r}; expected 'motion' or 'pico'")
