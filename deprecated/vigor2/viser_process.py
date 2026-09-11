"""Viser in its own process: it reads the plan from the planner's shared buffers, the depth frame from the
DepthProcess's shared double buffer, and the robot state from the control loop as ~35 floats per tick. No
second DDS participant or depth subscription: either one cost the control process ~20 % of its LowState
messages."""
from __future__ import annotations
import math
import os
import signal
import sys
import time
import traceback
import multiprocessing as mp

from types import SimpleNamespace

import numpy as np
import torch

INFO_PLAN_AGE, INFO_HELD, INFO_START_YAW, INFO_YAW_CMD, INFO_TICK, INFO_MODE_PR, INFO_MODE_MACHINE, INFO_IMU = range(8)  # then qj
INFO_QJ = INFO_IMU + 4
CMD_YAW, CMD_SEQ, CMD_HEARTBEAT = range(3)                           # written by the viser process
_RESP_END = 4  # planner_process ctl index of the finished-plan sequence number


def _child_main(repo_root, view_kwargs, depth_frames, depth_seq, states, plan_ctl, plan_stat, info, cmd, err_text):
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # CPU only, single-threaded, on two cores: thread pools spinning on every core pre-empt the control loop
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[k] = "1"
        cores = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, set(cores[-2:]))
        os.chdir(repo_root)
        sys.path.insert(0, repo_root)
        sys.path.insert(0, os.path.join(repo_root, "g1_deploy"))
        torch.set_num_threads(1)
        from deprecated.vigor2.viser_view import PlanDrawer

        def on_cmd(yaw_rad: float) -> None:
            cmd[CMD_YAW] = yaw_rad
            cmd[CMD_SEQ] += 1

        view = PlanDrawer(**view_kwargs, yaw_cmd_deg=math.degrees(float(info[INFO_YAW_CMD])), on_cmd=on_cmd)
        drawn = -1
        depth_np, depth_raw, depth_drawn = depth_frames.numpy(), None, -1
        while True:
            cmd[CMD_HEARTBEAT] = time.monotonic()
            if int(depth_seq[0]) != depth_drawn:
                depth_drawn = int(depth_seq[0])
                depth_raw = depth_np[depth_drawn % 2].copy()
            if int(info[INFO_TICK]) > 0:
                snap = info.numpy().copy()
                msg = SimpleNamespace(tick=int(snap[INFO_TICK]), mode_pr=int(snap[INFO_MODE_PR]), mode_machine=int(snap[INFO_MODE_MACHINE]),
                                      imu_state=SimpleNamespace(quaternion=tuple(snap[INFO_IMU:INFO_IMU + 4])))
                qj = snap[INFO_QJ:].astype(np.float32)
                yaw_now = view.update_robot(msg, qj, depth_raw, float(snap[INFO_YAW_CMD]), float(snap[INFO_START_YAW]),
                                            int(snap[INFO_PLAN_AGE]), float(plan_stat[0]), bool(snap[INFO_HELD]))
                seq = int(plan_ctl[_RESP_END])
                if seq != drawn and seq > 0:
                    view.update_plan(states[0].numpy(), yaw_now, qj, tuple(float(v) for v in snap[INFO_IMU:INFO_IMU + 4]))
                    drawn = seq
            time.sleep(0.05)
    except Exception:
        err_text.value = traceback.format_exc()[-4000:].encode()


class ViserProcess:
    def __init__(self, repo_root: str, view_kwargs: dict, depth_frames: torch.Tensor, depth_seq: torch.Tensor, num_dof: int,
                 states: torch.Tensor, plan_ctl: torch.Tensor, plan_stat: torch.Tensor, yaw_cmd_rad: float) -> None:
        ctx = mp.get_context("spawn")
        self.depth_frames, self.depth_seq = depth_frames, depth_seq  # the DepthProcess's double buffer
        self.info = torch.zeros(INFO_QJ + int(num_dof), dtype=torch.float64).share_memory_()
        self.info[INFO_START_YAW] = float("nan")
        self.info[INFO_YAW_CMD] = float(yaw_cmd_rad)
        self.cmd = torch.zeros(3, dtype=torch.float64).share_memory_()
        self.err_text = ctx.Array("c", 4096)
        self._seen_cmd_seq = 0
        self.proc = ctx.Process(target=_child_main, daemon=True,
                                args=(repo_root, view_kwargs, self.depth_frames, self.depth_seq, states, plan_ctl, plan_stat, self.info, self.cmd, self.err_text))
        self.proc.start()

    def publish(self, low_state, qj: np.ndarray, plan_age: int, held: bool, start_yaw, yaw_cmd_rad: float) -> None:
        """Called once per control tick: ~35 float writes into shared memory."""
        self.info[INFO_PLAN_AGE] = float(plan_age)
        self.info[INFO_HELD] = 1.0 if held else 0.0
        self.info[INFO_START_YAW] = float("nan") if start_yaw is None else float(start_yaw)
        self.info[INFO_YAW_CMD] = float(yaw_cmd_rad)
        self.info[INFO_MODE_PR] = float(low_state.mode_pr)
        self.info[INFO_MODE_MACHINE] = float(low_state.mode_machine)
        self.info[INFO_IMU:INFO_IMU + 4] = torch.as_tensor(np.asarray(low_state.imu_state.quaternion, dtype=np.float64))
        self.info[INFO_QJ:] = torch.as_tensor(np.asarray(qj, dtype=np.float64))
        self.info[INFO_TICK] = float(low_state.tick)

    def slider_command(self):
        """The slider's yaw command if it moved since the last call, else None."""
        seq = int(self.cmd[CMD_SEQ])
        if seq == self._seen_cmd_seq:
            return None
        self._seen_cmd_seq = seq
        return float(self.cmd[CMD_YAW])

    @property
    def error(self) -> str:
        return self.err_text.value.decode(errors="replace")

    def stop(self) -> None:
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=2.0)
