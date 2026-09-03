"""Depth stream in its own process: ZMQ receive + PNG decode never touch the control loop's GIL.
Frames land in a double-buffered shared array; the control loop and the viser process read the slot
the sequence number points at (a slot is rewritten only two frames later, ~66 ms)."""
from __future__ import annotations
import gc
import multiprocessing as mp
import os
import signal
import time
import traceback

import numpy as np
import torch

DEPTH_SCALE = 0.0010000000474974513


def _child_main(server_address, port, H, W, frames, seq, stamp, stop, err_text):
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        cores = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, set(cores[-4:-2]))  # two cores of its own; viser takes the last two
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[k] = "1"
        import cv2
        import zmq
        cv2.setNumThreads(1)
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.connect(f"tcp://{server_address}:{port}")
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.setsockopt(zmq.RCVTIMEO, 200)
        bufs = frames.numpy()
        gc.collect(); gc.freeze(); gc.disable()  # two arrays per frame; a gen-2 collection stalled delivery ~100 ms
        n = 0
        while int(stop[0]) == 0:
            try:
                parts = sock.recv_multipart()
            except zmq.Again:
                continue
            depth = cv2.imdecode(np.frombuffer(parts[-1], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if depth is None or depth.shape != (H, W):
                continue
            n += 1
            np.multiply(depth, DEPTH_SCALE, out=bufs[n % 2], casting="unsafe")
            stamp[0] = time.monotonic()
            seq[0] = n
    except Exception:
        err_text.value = traceback.format_exc()[-4000:].encode()


class DepthProcess:
    """Same surface the control loop used on the threaded client: latest_seq, latest_depth_full, latest_depth, running."""

    def __init__(self, server_address: str, port: int, height: int, width: int) -> None:
        ctx = mp.get_context("spawn")
        self.H, self.W = int(height), int(width)
        self.frames = torch.zeros(2, self.H, self.W, dtype=torch.float32).share_memory_()
        self.seq = torch.zeros(1, dtype=torch.int64).share_memory_()
        self.stamp = torch.zeros(1, dtype=torch.float64).share_memory_()
        self._stop = torch.zeros(1, dtype=torch.int64).share_memory_()
        self.err_text = ctx.Array("c", 4096)
        self._frames_np = self.frames.numpy()
        self.proc = ctx.Process(target=_child_main, daemon=True,
                                args=(server_address, int(port), self.H, self.W, self.frames, self.seq, self.stamp, self._stop, self.err_text))

    def start(self) -> None:
        self.proc.start()

    @property
    def latest_seq(self) -> int:
        return int(self.seq[0])

    @property
    def latest_depth_full(self):
        n = int(self.seq[0])
        return None if n == 0 else self._frames_np[n % 2]

    @property
    def latest_depth(self):
        full = self.latest_depth_full
        if full is None:
            return None
        crop = min(self.H, self.W)
        y0, x0 = (self.H - crop) // 2, (self.W - crop) // 2
        return full[y0:y0 + crop, x0:x0 + crop]

    @property
    def running(self) -> bool:
        return self.proc.is_alive()

    @running.setter
    def running(self, value: bool) -> None:
        if not value:
            self._stop[0] = 1
            self.proc.join(timeout=2.0)

    @property
    def error(self) -> str:
        return self.err_text.value.decode(errors="replace")
