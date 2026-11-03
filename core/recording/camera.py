"""Random access to the external camera dump, using driver exposure timestamps."""

import os
import subprocess
from pathlib import Path

import numpy as np


class RecordedCamera:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.raw = self.directory / "frames.mjpg"
        with np.load(self.directory / "frames.npz", allow_pickle=False) as index:
            self.times = np.asarray(index["ts_monotonic_s"], dtype=float)
            self.offsets = np.asarray(index["offset"], dtype=np.int64)
            self.sizes = np.asarray(index["bytes"], dtype=np.int64)
        n = len(self.times)
        if (
            not n
            or self.times.shape != (n,)
            or self.offsets.shape != (n,)
            or self.sizes.shape != (n,)
            or not np.isfinite(self.times).all()
            or np.any(np.diff(self.times) <= 0)
        ):
            raise ValueError("Camera index requires finite, increasing exposure timestamps")
        if (
            self.offsets[0] != 0
            or np.any(self.sizes <= 0)
            or np.any(self.offsets[1:] != (self.offsets + self.sizes)[:-1])
            or self.offsets[-1] + self.sizes[-1] > self.raw.stat().st_size
        ):
            raise ValueError("Camera byte index does not match the saved MJPEG dump")
        self._cached_index = None
        self._cached_image = None

    def indices_at(self, times, *, max_age_s=0.25):
        """Latest exposure at/before each query; blank outside coverage or in gaps."""
        times = np.asarray(times, dtype=float)
        indices = np.searchsorted(self.times, times, side="right") - 1
        age = times - self.times[np.clip(indices, 0, len(self.times) - 1)]
        valid = (indices >= 0) & np.isfinite(times) & (age >= 0) & (age <= max_age_s)
        return np.where(valid, indices, -1)

    def frame(self, index):
        """Decode one BGR image; memory does not grow with recording length."""
        if index < 0:
            return None
        if index >= len(self.times):
            raise IndexError(index)
        if index != self._cached_index:
            import cv2

            with self.raw.open("rb") as stream:
                stream.seek(int(self.offsets[index]))
                data = stream.read(int(self.sizes[index]))
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot decode external camera frame {index}")
            # Limit browser traffic, preserving aspect ratio. Raw video stays full resolution.
            if image.shape[1] > 960:
                image = cv2.resize(
                    image,
                    (960, round(image.shape[0] * 960 / image.shape[1])),
                    interpolation=cv2.INTER_AREA,
                )
            self._cached_index, self._cached_image = index, image
        return self._cached_image

    def validate(self, query, *, full=True):
        """Check coverage and decode a sample, or all frames when requested."""
        selected = (
            range(len(self.times))
            if full
            else sorted(set([0, len(self.times) // 2, len(self.times) - 1]))
        )
        for i in selected:
            self.frame(i)
        indices = self.indices_at(query)
        valid = indices >= 0
        ages = np.asarray(query)[valid] - self.times[indices[valid]]
        return dict(
            frames=len(self.times),
            decoded_frames=len(selected),
            validation="full" if full else "sampled",
            duration_s=float(self.times[-1] - self.times[0]),
            robot_timeline_coverage=float(valid.mean()),
            exposure_age_p95_ms=None if not len(ages) else float(np.percentile(ages, 95) * 1000),
            alignment="V4L2 monotonic exposure stamps versus recorded robot timestamps; "
            "sensor clock offsets are not independently calibrated",
        )


def camera_report(output, fps, error=None, policy_start_monotonic_s=None):
    """After the policy stopped: index the raw dump from the driver log, cut a truncated tail,
    remux to a playable MKV. Drops are driver sequence-number gaps, so zero is exact."""
    import re
    import numpy as np

    if error:
        return dict(error=error)
    raw, log = output / "camera/frames.mjpg", output / "camera.log"
    if not raw.is_file() or not raw.stat().st_size or not log.is_file():
        return dict(error="no camera frames written")
    text = log.read_text()
    rows = re.findall(CAMERA_FRAME_PATTERN, text)
    if not rows:
        return dict(error="camera.log holds no frames; inspect it")
    seq = np.array([int(r[0]) for r in rows])
    size = np.array([int(r[1]) for r in rows])
    ts = np.array([float(r[2]) for r in rows])
    offset = np.concatenate([[0], np.cumsum(size)[:-1]])
    complete = int(np.searchsorted(offset + size, raw.stat().st_size, side="right"))
    truncated = complete < len(rows)
    if truncated:
        seq, size, ts, offset = seq[:complete], size[:complete], ts[:complete], offset[:complete]
        if complete:
            os.truncate(raw, int(offset[-1] + size[-1]))
    if len(seq) == 0:
        return dict(error="no complete camera frame")
    gaps = np.diff(ts)
    measured = float(1 / gaps[1:].mean()) if len(gaps) > 1 else None
    # The driver clamps the request to a mode it offers; its own message is not retained in the
    # log, so the effective rate is the accepted rate when logged, else the measured cadence.
    rates = re.findall(CAMERA_RATE_PATTERN, text)
    requested_fps = fps
    fps = float(rates[-1]) if rates else (float(round(measured)) if measured else float(fps))
    np.savez(
        output / "camera/frames.npz",
        seq=seq,
        bytes=size,
        offset=offset,
        ts_monotonic_s=ts,
        fps=fps,
        policy_start_monotonic_s=(
            np.nan if policy_start_monotonic_s is None else policy_start_monotonic_s
        ),
    )
    try:
        remux = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "mjpeg",
                "-framerate",
                f"{fps:g}",
                "-i",
                str(raw),
                "-c:v",
                "copy",
                "-f",
                "matroska",
                str(output / "camera/external.mkv"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        remux = subprocess.CompletedProcess([], 1, "", f"{type(exc).__name__}: {exc}")
    index_at_start = (
        None
        if policy_start_monotonic_s is None
        else int(np.searchsorted(ts, policy_start_monotonic_s))
    )
    return dict(
        error=None,
        frames=int(len(seq)),
        dropped=int((np.diff(seq) - 1).sum()) if len(seq) > 1 else 0,
        duration_s=float(ts[-1] - ts[0]),
        fps=fps,
        requested_fps=requested_fps,
        measured_fps=measured,
        warmup_s=float(gaps[0]) if len(gaps) else None,
        interval_ms=dict(
            mean=float(gaps[1:].mean() * 1000) if len(gaps) > 1 else None,
            max=float(gaps[1:].max() * 1000) if len(gaps) > 1 else None,
        ),
        truncated_last_frame=bool(truncated),
        bytes=int(raw.stat().st_size),
        video=None if remux.returncode else "camera/external.mkv",
        remux_error=remux.stderr.strip() if remux.returncode else None,
        frame_index_at_policy_start=index_at_start,
        video_seconds_at_policy_start=None if index_at_start is None else index_at_start / fps,
    )


CAMERA_FRAME_PATTERN = r"dqbuf: \d+ seq:\s+(\d+) bytesused: (\d+) ts: ([\d.]+)"


CAMERA_RATE_PATTERN = r"Frame rate set to ([\d.]+) fps"


DEFAULT_CAMERA = "/dev/v4l/by-id/usb-046d_Brio_101_2540APD0M3H8-video-index0"


def camera_modes(device):
    """MJPG (width, height, fps) triples the driver offers, from v4l2-ctl --list-formats-ext."""
    import re

    text = subprocess.run(
        ["v4l2-ctl", "-d", device, "--list-formats-ext"], capture_output=True, text=True
    ).stdout
    block = text.split("'MJPG'", 1)[1] if "'MJPG'" in text else ""
    block = re.split(r"\n\s*\[\d+\]:", block)[0]
    modes = []
    size = None
    for line in block.splitlines():
        found = re.search(r"Size: Discrete (\d+)x(\d+)", line)
        if found:
            size = (int(found[1]), int(found[2]))
        rate = re.search(r"\(([\d.]+) fps\)", line)
        if rate and size:
            modes.append((size[0], size[1], float(rate[1])))
    return modes


def camera_command(device, size, fps, target):
    """Kernel-only capture: the driver's MJPEG frames go straight to `target` ('-' for stdout);
    each frame's start-of-exposure CLOCK_MONOTONIC stamp and sequence number go to stderr.
    ffmpeg's v4l2 input was measured at ~30% of a core spinning on non-blocking DQBUF."""
    width, height = size.split("x")
    return [
        "bash",
        "-c",
        f'v4l2-ctl -d "$1" -c exposure_dynamic_framerate=0; exec v4l2-ctl -d "$1" '
        f"--set-fmt-video=width={int(width)},height={int(height)},pixelformat=MJPG --set-parm={int(fps)} "
        f'--stream-mmap=4 --stream-to="$2" --verbose',
        "camera",
        device,
        target,
    ]
