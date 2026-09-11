"""Planner in its own process (SONIC's manager split): sampling never shares a GIL or a CUDA queue with
the control loop. Requests and plans cross through shared CPU tensors guarded by seqlocks."""
from __future__ import annotations
import os
import signal
import sys
import time
import traceback
import torch
import torch.multiprocessing as mp

KEYS = ("body_pos", "body_rot", "body_vel", "body_ang_vel", "dof_pos", "dof_vel")
REQ_BEGIN, REQ_END, REQ_TICK, RESP_BEGIN, RESP_END, RESP_TICK, READY, ERROR, STOP = range(9)


def _child_main(repo_root, planner_kwargs, req_state, req_hist, req_exec, req_depth, req_cmd,
                resp_plan, resp_states, ctl, stat, err_text):
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is the control process's to handle; it stops us
        os.chdir(repo_root)
        sys.path.insert(0, repo_root)
        sys.path.insert(0, os.path.join(repo_root, "g1_deploy"))
        from deprecated.vigor2.planner import DeployPlanner
        planner = DeployPlanner(**planner_kwargs)
        dev = planner.device
        ctl[READY] = 1
        done = 0
        while int(ctl[STOP]) == 0:
            stat[1] = time.monotonic()  # heartbeat
            s = int(ctl[REQ_END])
            if s == done or int(ctl[REQ_BEGIN]) != s:
                time.sleep(0.0005)
                continue
            window = {k: req_state[k].to(dev, non_blocking=False) for k in KEYS}
            planner.hist_action = req_hist.to(dev)
            planner.exec_action = req_exec.to(dev)
            depth, cmd, tick = req_depth.to(dev), req_cmd.to(dev), int(ctl[REQ_TICK])
            t0 = time.perf_counter()
            with torch.no_grad():
                plan, states = planner.plan_from(window, depth, cmd)
            torch.cuda.synchronize()
            ctl[RESP_BEGIN] = s
            resp_plan.copy_(plan.cpu())
            resp_states.copy_(states.cpu())
            ctl[RESP_TICK] = tick
            stat[0] = (time.perf_counter() - t0) * 1000.0
            ctl[RESP_END] = s
            done = s
    except Exception:
        err_text.value = traceback.format_exc()[-4000:].encode()
        ctl[ERROR] = 1


class PlannerProcess:
    def __init__(self, repo_root: str, planner_kwargs: dict, window_shapes: dict, depth_dim: int,
                 cmd_dim: int, H: int, F: int, action_dim: int, num_dof: int, state_dim: int, device: str,
                 ready_timeout_s: float = 120.0):
        ctx = mp.get_context("spawn")
        self.device = device
        self.req_state = {k: torch.zeros(tuple(window_shapes[k])).share_memory_() for k in KEYS}
        self.req_hist = torch.zeros(1, H, action_dim).share_memory_()
        self.req_exec = torch.zeros(1, H, num_dof).share_memory_()
        self.req_depth = torch.zeros(1, depth_dim).share_memory_()
        self.req_cmd = torch.zeros(1, cmd_dim).share_memory_()
        self.resp_plan = torch.zeros(1, F, action_dim).share_memory_()
        self.resp_states = torch.zeros(1, F, state_dim).share_memory_()
        self.ctl = torch.zeros(9, dtype=torch.int64).share_memory_()
        self.stat = torch.zeros(2, dtype=torch.float64).share_memory_()  # [latency_ms, heartbeat]
        self.err_text = ctx.Array("c", 4096)
        self.proc = ctx.Process(
            target=_child_main,
            args=(repo_root, planner_kwargs, self.req_state, self.req_hist, self.req_exec, self.req_depth,
                  self.req_cmd, self.resp_plan, self.resp_states, self.ctl, self.stat, self.err_text),
            daemon=True,
        )
        self.latest_plan = None
        self.latest_states = None
        self.latest_tick = 0
        self.latest_seq = 0
        self.last_latency_ms = 0.0
        self.heartbeat_timeout_s = 1.0
        self.proc.start()
        deadline = time.monotonic() + ready_timeout_s
        while int(self.ctl[READY]) == 0:
            self.check()
            if time.monotonic() > deadline:
                raise RuntimeError(f"Planner process not ready after {ready_timeout_s:.0f} s")
            time.sleep(0.01)

    def check(self) -> None:
        if int(self.ctl[ERROR]) != 0:
            raise RuntimeError("Planner process died:\n" + self.err_text.value.decode(errors="replace"))
        if not self.proc.is_alive():
            raise RuntimeError("Planner process exited")
        hb = float(self.stat[1])
        if hb > 0.0 and time.monotonic() - hb > self.heartbeat_timeout_s:
            raise RuntimeError(f"Planner process silent for {time.monotonic() - hb:.2f} s")

    @property
    def idle(self) -> bool:
        return int(self.ctl[REQ_END]) == int(self.ctl[RESP_END])

    def submit(self, state_window: dict, depth_flat, command, hist_action, exec_action, tick: int) -> None:
        s = int(self.ctl[REQ_END]) + 1
        self.ctl[REQ_BEGIN] = s
        for k in KEYS:
            self.req_state[k].copy_(state_window[k])
        self.req_hist.copy_(hist_action)
        self.req_exec.copy_(exec_action)
        self.req_depth.copy_(depth_flat)
        self.req_cmd.copy_(command)
        self.ctl[REQ_TICK] = int(tick)
        self.ctl[REQ_END] = s

    def poll(self) -> bool:
        """Pull the newest plan if one landed since the last poll."""
        s = int(self.ctl[RESP_END])
        if s == self.latest_seq or int(self.ctl[RESP_BEGIN]) != s:
            return False
        self.latest_plan = self.resp_plan.to(self.device)
        self.latest_states = self.resp_states.clone()
        self.latest_tick = int(self.ctl[RESP_TICK])
        self.last_latency_ms = float(self.stat[0])
        self.latest_seq = s
        return True

    def stop(self) -> None:
        self.ctl[STOP] = 1
        self.proc.join(timeout=2.0)
        if self.proc.is_alive():
            self.proc.kill()
