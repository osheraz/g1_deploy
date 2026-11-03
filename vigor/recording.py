"""VIGOR array recordings for its retained replay format."""

from __future__ import annotations

from typing import Any, Dict, Optional
import threading
import queue
import numpy as np
import time
import os
from pathlib import Path


class RealLogger:
    def __init__(self, save_path: str, max_queue: int = 3000):

        repo_root = Path(__file__).resolve().parents[1]
        self.save_path = os.path.join(repo_root, "logs", "real_rollouts", save_path)
        self.q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max_queue)
        self.dropped_samples = 0
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.data: Dict[str, list] = {}
        self.t0_wall = time.time()

    def start(self):
        self.worker.start()

    def push(self, sample: Dict[str, Any]):
        try:
            self.q.put_nowait(sample)
        except queue.Full:
            self.dropped_samples += 1

    def _run(self):
        while (not self.stop_event.is_set()) or (not self.q.empty()):
            try:
                item = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            for k, v in item.items():
                if k not in self.data:
                    self.data[k] = []
                self.data[k].append(v)

    def stop(self):
        self.stop_event.set()
        self.worker.join(timeout=2.0)
        if self.worker.is_alive():
            raise RuntimeError("VIGOR recorder did not finish draining its queue")

    def save(self, meta: Optional[Dict[str, Any]] = None):
        out: Dict[str, Any] = {}
        for k, lst in self.data.items():
            out[k] = np.asarray(lst)

        m = {"t0_wall": float(self.t0_wall), "dropped_samples": self.dropped_samples}
        if meta is not None:
            m.update(meta)
        out["meta"] = np.array([m], dtype=object)

        ts = int(time.time())
        base, ext = os.path.splitext(self.save_path)
        save_path = f"{base}_{ts}{ext}"

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, **out)
        self.save_path = save_path
        print(f"[RealLogger] Saved: {self.save_path}")
