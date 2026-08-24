"""Episode recording for data collection.

An episode is a directory:

    episodes/episode_001/
        state.npz     one row per control step: robot, tactile, teleop, actions
        frames.bin    camera frames, length-prefixed, stored as received
        meta.json     shapes, rates, schema

Camera frames are kept **still encoded** (JPEG colour, PNG depth) exactly as the
image server sent them. The server encodes before publishing, so decoding and
re-storing them would be ~5x larger for identical pixels, and depth PNG is
lossless either way. Frames are appended as they arrive, so RAM stays flat and
an episode is bounded by disk, not memory.
"""
from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np

from core.common.remote_controller import KeyMap

# frames.bin record: two length-prefixed blobs per frame.
HDR = struct.Struct("<II")  # len(color_jpg), len(depth_png); 0 means absent


class EpisodeWriter:
    """One episode: state accumulated in RAM, camera frames streamed to disk."""

    def __init__(self, root: Path, index: int, meta: dict):
        self.dir = Path(root) / f"episode_{index:03d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.frames = open(self.dir / "frames.bin", "wb")
        self.data: dict[str, list] = {}
        self.offsets: list[int] = []
        self.meta = dict(meta)
        self.t0 = time.time()
        self.n = 0
        self.unique = 0             # camera frames actually written
        self._last_offset = None

    def push(self, sample: dict, color_jpg: Optional[bytes],
             depth_png: Optional[bytes], new_frame: bool = True) -> None:
        """One control step.

        The control loop runs faster than the camera, so most steps see the
        frame they already saw. ``new_frame=False`` records the step but points
        it at the bytes already on disk instead of writing them again -- the
        offsets array simply repeats, and readers are none the wiser.
        """
        if new_frame or self._last_offset is None:
            self._last_offset = self.frames.tell()
            c, d = color_jpg or b"", depth_png or b""
            self.frames.write(HDR.pack(len(c), len(d)))
            self.frames.write(c)
            self.frames.write(d)
            self.unique += 1
        self.offsets.append(self._last_offset)
        for k, v in sample.items():
            self.data.setdefault(k, []).append(v)
        self.n += 1

    @property
    def bytes_on_disk(self) -> int:
        return self.frames.tell()

    def close(self) -> Path:
        self.frames.close()
        arrays = {k: np.asarray(v) for k, v in self.data.items()}
        arrays["frame_offset"] = np.asarray(self.offsets, np.int64)
        np.savez(self.dir / "state.npz", **arrays)

        self.meta.update({
            "frames": int(self.n),
            "unique_camera_frames": int(self.unique),
            "duration_s": float(time.time() - self.t0),
            "t0_wall": float(self.t0),
            "frames_encoding": "color_jpg+depth_png",
            "depth_units": "mm_uint16",
            "color_format": "bgr8",
        })
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2))
        return self.dir


class Recorder:
    """Single-button episode toggle on the gamepad.

    Press to start an episode, press again to stop and write it. Episode numbering continues
    from whatever is already in the output directory.
    """

    def __init__(self, root, meta: dict, button: int | None = KeyMap.X):
        self.root = Path(root).expanduser()
        self.meta = meta
        self.button = button
        self.button_name = (
            "automatic-only"
            if button is None
            else next(
                (
                    k
                    for k, v in vars(KeyMap).items()
                    if not k.startswith("_") and v == button
                ),
                str(button),
            )
        )
        self.episode: Optional[EpisodeWriter] = None
        self.index = self._next_index()
        self._prev = 0

    def _next_index(self) -> int:
        if not self.root.exists():
            return 1
        used = [int(p.name.rsplit("_", 1)[-1]) for p in self.root.glob("episode_*")
                if p.name.rsplit("_", 1)[-1].isdigit()]
        return max(used, default=0) + 1

    def handle_buttons(self, button) -> None:
        """One button toggles: press to start, press again to stop and write."""
        if self.button is None:
            return
        pressed = button[self.button]
        if pressed and not self._prev:
            if self.episode is None:
                self.start()
            else:
                self.stop()
        self._prev = pressed

    def start(self) -> None:
        """Start a new episode unless one is already active."""
        if self.episode is not None:
            return
        self.episode = EpisodeWriter(self.root, self.index, self.meta)
        print(f"\n[rec] START episode_{self.index:03d}")

    def push(self, sample: dict, color_jpg=None, depth_png=None,
             new_frame: bool = True) -> None:
        if self.episode is not None:
            self.episode.push(sample, color_jpg, depth_png, new_frame)

    def stop(self) -> None:
        if self.episode is None:
            return
        n, size = self.episode.n, self.episode.bytes_on_disk
        path = self.episode.close()
        print(f"[rec] STOP  {path}  {n} frames, {size / 1e9:.2f} GB")
        self.episode = None
        self.index += 1

    @property
    def recording(self) -> bool:
        return self.episode is not None

    def status(self) -> str:
        if self.episode is None:
            return f"idle ({self.button_name} to record)"
        return (f"REC ep{self.index:03d} {self.episode.n} frames "
                f"{self.episode.bytes_on_disk / 1e9:.2f} GB")


def load_episode(ep_dir, decode: bool = True):
    """Load an episode.

        state, get_frame = load_episode("episodes/episode_001")
        color, depth = get_frame(0)
    """
    import cv2

    ep_dir = Path(ep_dir).expanduser()
    state = dict(np.load(ep_dir / "state.npz", allow_pickle=True))
    offsets = state["frame_offset"]
    path = ep_dir / "frames.bin"

    def get_frame(i):
        with open(path, "rb") as fh:
            fh.seek(int(offsets[i]))
            nc, nd = HDR.unpack(fh.read(HDR.size))
            cjpg = fh.read(nc) if nc else None
            dpng = fh.read(nd) if nd else None
        if not decode:
            return cjpg, dpng
        color = (cv2.imdecode(np.frombuffer(cjpg, np.uint8), cv2.IMREAD_COLOR)
                 if cjpg else None)
        depth = (cv2.imdecode(np.frombuffer(dpng, np.uint8), cv2.IMREAD_UNCHANGED)
                 if dpng else None)
        return color, depth

    return state, get_frame
