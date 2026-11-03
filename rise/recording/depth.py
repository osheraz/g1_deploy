"""Record original depth once; derive video and exact model views on replay.

Compressed network captures reuse the lossless PNG bytes already received.
Raw network and USB captures copy uint16 pixels and encode on the writer thread.
Neither path adds a stream,
GPU work, codec or disk I/O to the control/capture thread. Legacy metric archives
remain readable.
"""

import json
import hashlib
from pathlib import Path
import queue
import threading
from collections import deque

import numpy as np


class DepthFrameRecorder:
    def __init__(
        self, directory, capture_config, *, capacity=120, storage="metric", start_paused=False
    ):
        if storage not in ("metric", "native_png"):
            raise ValueError("Unknown depth storage")
        self.storage = storage
        self.capture_config = capture_config
        self.capture_config_sha256 = hashlib.sha256(
            json.dumps(capture_config, sort_keys=True).encode()
        ).hexdigest()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.queue = queue.Queue(maxsize=capacity)
        self.stop = threading.Event()
        self.frames = 0
        self.dropped = 0
        self.error = None
        self.recording = threading.Event()
        self.pending = deque(maxlen=capacity)
        self.pending_lock = threading.Lock()
        if not start_paused:
            self.recording.set()
        metadata = dict(
            schema="hv2.recorded_metric_depth.v1",
            capture_config=capture_config,
            representation="calibrated metric camera image and surface-valid mask, before model encoding",
            layout="float32 little-endian depth followed by uint8 validity, row-major",
            timestamp="capture_s mapped to workstation monotonic time; timing qualification unchanged",
        )
        if storage == "native_png":
            metadata.update(
                schema="hv2.recorded_native_depth.v2",
                representation="source uint16 optical depth after declared camera processing, native resolution and units; lossless PNG",
                layout="concatenated PNG payloads indexed by byte offset; scale and intrinsics per exposure",
                preprocessing="calibrated_depth_image_v1",
                video_view="same calibrated crop and finite-positive mask, no resampling",
                model_view="same calibrated crop/ray sampling to 64x48, then model range/value/mask transform",
            )
        self.metadata = metadata
        (self.directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        self.thread = threading.Thread(target=self._write, name="depth-recording", daemon=True)
        self.thread.start()

    def offer(self, depth, valid, timing):
        if self.error or self.stop.is_set():
            return
        try:
            self._enqueue(
                (
                    np.asarray(depth, dtype="<f4").copy(),
                    np.asarray(valid, dtype=np.uint8).copy(),
                    dict(timing),
                )
            )
        except queue.Full:
            self.dropped += 1

    def _offer_native(self, payload, kind, depth_scale_m, intrinsics, timing):
        if self.error or self.stop.is_set():
            return
        # Avoid a full USB frame copy when storage is already falling behind.
        if self.queue.full():
            self.dropped += 1
            return
        try:
            if kind == "raw":
                payload = np.asarray(payload).copy()
            else:
                payload = bytes(payload)  # already-owned bytes are not copied
            self._enqueue(
                (
                    payload,
                    kind,
                    dict(
                        timing,
                        depth_scale_m=float(depth_scale_m),
                        intrinsics=list(intrinsics),
                        capture_config_sha256=self.capture_config_sha256,
                    ),
                )
            )
        except queue.Full:
            self.dropped += 1

    def _enqueue(self, item):
        with self.pending_lock:
            if not self.recording.is_set():
                # The first inference can reference exposures before A. Retain
                # only its short causal context in RAM, never the idle video.
                self.pending.append(item)
                cutoff = item[2]["capture_s"] - 0.5
                while self.pending and self.pending[0][2]["capture_s"] < cutoff:
                    self.pending.popleft()
            else:
                self.queue.put_nowait(item)

    def start(self):
        """Queue the bounded history; disk/encoding work stays on the writer."""
        with self.pending_lock:
            if self.recording.is_set():
                return
            for item in self.pending:
                self.queue.put_nowait(item)
            self.pending.clear()
            self.recording.set()

    def offer_png(self, png, *, depth_scale_m, intrinsics, timing, source_processing=None):
        """Offer the validated network packet's immutable PNG; never re-encode."""
        if source_processing is not None:
            timing = dict(timing, _source_processing=source_processing)
        self._offer_native(png, "png", depth_scale_m, intrinsics, timing)

    def offer_raw(self, raw, *, depth_scale_m, intrinsics, timing, source_processing=None):
        """Own raw USB/network pixels; compress only on the recording thread."""
        if source_processing is not None:
            timing = dict(timing, _source_processing=source_processing)
        self._offer_native(raw, "raw", depth_scale_m, intrinsics, timing)

    def _write(self):
        try:
            while not self.recording.wait(0.1):
                if self.stop.is_set():
                    return
            with (self.directory / "frames.bin").open("wb") as raw, (
                self.directory / "frames.jsonl"
            ).open("w") as index:
                while not self.stop.is_set() or not self.queue.empty():
                    try:
                        depth, valid, timing = self.queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    source_processing = timing.pop("_source_processing", None)
                    if source_processing is not None:
                        provenance = source_processing.get("sensor_provenance")
                        if provenance is not None and "exposure" in provenance:
                            # Auto exposure is a per-frame acquisition value,
                            # not a change to the immutable filtering recipe.
                            provenance = dict(provenance)
                            timing["exposure_us"] = provenance.pop("exposure")
                            source_processing = dict(
                                source_processing, sensor_provenance=provenance
                            )
                        saved = self.metadata.get("source_processing")
                        if saved is None:
                            # Save the recipe once, off the capture/control threads.
                            self.metadata["source_processing"] = source_processing
                            (self.directory / "metadata.json").write_text(
                                json.dumps(self.metadata, indent=2) + "\n"
                            )
                        elif saved != source_processing:
                            raise ValueError("Depth processing changed within one recording")
                    if self.storage == "native_png":
                        shape = list(reversed(self.capture_config["base_resolution"]))
                        if valid == "raw":
                            import cv2

                            if depth.dtype != np.uint16 or list(depth.shape) != shape:
                                raise ValueError("Invalid native depth recording shape/type")
                            ok, encoded = cv2.imencode(
                                ".png", depth, [cv2.IMWRITE_PNG_COMPRESSION, 1]
                            )
                            if not ok:
                                raise ValueError("Depth PNG encoding failed")
                            data = encoded.tobytes()
                        elif valid == "png":
                            data = depth
                        else:
                            raise ValueError("Expected native depth recording payload")
                        row = dict(timing, encoding="png_uint16", shape=shape)
                    else:
                        if depth.ndim != 2 or valid.shape != depth.shape:
                            raise ValueError("Invalid depth recording shape")
                        data = depth.tobytes() + valid.tobytes()
                        row = dict(timing, shape=list(depth.shape))
                    row.update(offset=raw.tell(), bytes=len(data))
                    raw.write(data)
                    digest = hashlib.sha256(
                        data + json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest()
                    index.write(json.dumps(dict(row, sha256=digest)) + "\n")
                    self.frames += 1
        except Exception as exc:
            self.error = str(exc)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10.0)
        if self.thread.is_alive():
            self.error = self.error or "Depth writer did not finish before cleanup timeout"
        result = dict(
            frames=self.frames,
            dropped=self.dropped,
            error=self.error,
            complete=not self.error and self.dropped == 0 and self.frames > 0,
        )
        (self.directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        return result


class RecordedDepth:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.raw = self.directory / "frames.bin"
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        self.native = self.metadata["schema"] == "hv2.recorded_native_depth.v2"
        if self.metadata["schema"] not in (
            "hv2.recorded_metric_depth.v1",
            "hv2.recorded_native_depth.v2",
        ):
            raise ValueError("Unknown recorded depth schema")
        if self.native and self.metadata.get("preprocessing") != "calibrated_depth_image_v1":
            raise ValueError("Unknown native depth preprocessing")
        self.capture_config = self.metadata["capture_config"]
        self.rows = [
            json.loads(line)
            for line in (self.directory / "frames.jsonl").read_text().splitlines()
            if line
        ]
        self.times = np.array([r["capture_s"] for r in self.rows], float)
        if (
            not len(self.times)
            or not np.isfinite(self.times).all()
            or np.any(np.diff(self.times) <= 0)
        ):
            raise ValueError("Recorded depth needs finite, increasing capture times")
        size = self.raw.stat().st_size
        end = 0
        config_hash = hashlib.sha256(
            json.dumps(self.capture_config, sort_keys=True).encode()
        ).hexdigest()
        for r in self.rows:
            if (
                len(r["shape"]) != 2
                or min(r["shape"]) <= 0
                or r["bytes"] <= 0
                or r["offset"] != end
            ):
                raise ValueError("Invalid depth frame index")
            if self.native:
                if (
                    r.get("encoding") != "png_uint16"
                    or r["shape"] != list(reversed(self.capture_config["base_resolution"]))
                    or not 0 < float(r.get("depth_scale_m", 0)) < 0.1
                    or r.get("capture_config_sha256") != config_hash
                ):
                    raise ValueError("Native depth index/calibration differs")
            elif r["bytes"] != int(np.prod(r["shape"])) * 5:
                raise ValueError("Invalid metric depth frame size")
            end = r["offset"] + r["bytes"]
            if end > size:
                raise ValueError("Truncated depth frame")
        self._index = None
        self._frame = None
        self._model_frame = None
        self._source = None
        self.by_capture = {float(r["capture_s"]): i for i, r in enumerate(self.rows)}
        height, width = self.rows[0]["shape"]
        dx, dy = self.capture_config["crop_px"] if self.native else (0, 0)
        self.view_shape = (height - 2 * dy, width - 2 * dx)

    def indices_at(self, query, max_age_s=0.25):
        query = np.asarray(query, float)
        idx = np.searchsorted(self.times, query, side="right") - 1
        age = query - self.times[np.maximum(idx, 0)]
        return np.where((idx >= 0) & np.isfinite(query) & (age >= 0) & (age <= max_age_s), idx, -1)

    def _load(self, index):
        if index != self._index:
            r = self.rows[index]
            n = int(np.prod(r["shape"]))
            with self.raw.open("rb") as f:
                f.seek(r["offset"])
                data = f.read(r["bytes"])
            if "sha256" in r:
                body = {k: v for k, v in r.items() if k != "sha256"}
                digest = hashlib.sha256(
                    data + json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                if digest != r["sha256"]:
                    raise ValueError("Recorded depth checksum mismatch")
            if self.native:
                import cv2

                source = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
                if source is None or source.dtype != np.uint16 or list(source.shape) != r["shape"]:
                    raise ValueError("Invalid original depth PNG")
                self._source = source.astype(np.float32) * r["depth_scale_m"]
                self._frame = self._model_frame = None
            else:
                depth = np.frombuffer(data[: n * 4], dtype="<f4").reshape(r["shape"])
                valid = (
                    np.frombuffer(data[n * 4 :], dtype=np.uint8).reshape(r["shape"]).astype(bool)
                )
                self._frame = self._model_frame = (depth, valid)
            self._index = index

    def _calibrated(self, index, *, native_resolution):
        from humanoidverse2.observations.streams import calibrated_depth_image

        result = calibrated_depth_image(
            self._source,
            intrinsics=self.rows[index]["intrinsics"],
            capture_config=self.capture_config,
            native_resolution=native_resolution,
        )
        return tuple(value.numpy() for value in result)

    def frame(self, index):
        """Video view: original pixels after the calibrated crop/validity handling."""
        if index < 0:
            return None, None
        self._load(index)
        if self._frame is None:
            self._frame = self._calibrated(index, native_resolution=True)
        return self._frame

    def model_frame(self, index):
        """Exact model rays, including legacy archives already sampled to 64x48."""
        if index < 0:
            return None, None
        self._load(index)
        if self._model_frame is None:
            self._model_frame = self._calibrated(index, native_resolution=False)
        return self._model_frame


class RecordedModelDepth:
    """Rebuild model image stacks from exact exposure references, not guesses."""

    def __init__(self, directory):
        from collections import OrderedDict

        self.directory = Path(directory)
        self.archive = RecordedDepth(directory)
        summary = json.loads((self.directory / "summary.json").read_text())
        if not summary["complete"] or summary["dropped"] or summary["error"]:
            raise ValueError("Incomplete depth archive for model replay")
        if summary["frames"] != len(self.archive.rows):
            raise ValueError("Depth archive frame count mismatch")
        last = self.archive.rows[-1]
        if last["offset"] + last["bytes"] != self.archive.raw.stat().st_size:
            raise ValueError("Depth archive trailing bytes")
        for i, row in enumerate(self.archive.rows):
            if "sha256" not in row:
                raise ValueError("Referenced model depth requires frame checksums")
            self.archive.model_frame(i)
        self.cache = OrderedDict()

    def restore(self, view, config):
        import torch
        from humanoidverse2.observations.depth import prepare_depth_history

        if view is None or view.get("storage") != "metric_capture_refs_v1":
            return view
        available = np.asarray(view["available"], dtype=bool)
        captures = np.asarray(view["capture_s"])
        if available.shape != (1, config["history_steps"]) or captures.shape != available.shape:
            raise ValueError("Depth exposure reference shape differs from model")
        if not np.array_equal(available, captures >= 0):
            raise ValueError("Depth exposure reference availability mismatch")
        depth = np.zeros((*available.shape, 48, 64), np.float32)
        valid = np.zeros_like(depth, dtype=bool)
        for slot in range(available.shape[1]):
            if not available[0, slot]:
                continue
            stamp = float(captures[0, slot])
            index = self.archive.by_capture.get(stamp)
            if index is None:
                raise ValueError(f"Model-selected depth capture is missing: {stamp}")
            if index not in self.cache:
                self.cache[index] = self.archive.model_frame(index)
                if len(self.cache) > 16:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(index)
            depth[0, slot], valid[0, slot] = self.cache[index]
        # Reuse the model's actual pixel transform. Metadata was already small;
        # retain its exact float32 age/capture-state/relative-attitude values.
        metadata = torch.as_tensor(view["metadata"])
        if metadata.shape != (*available.shape, config["metadata_dim"]):
            raise ValueError("Depth metadata shape mismatch")
        quats = torch.zeros((*available.shape, 4))
        quats[..., 3] = 1.0
        restored = prepare_depth_history(
            torch.from_numpy(depth),
            torch.from_numpy(valid),
            torch.from_numpy(available),
            metadata[..., 0],
            torch.zeros((*available.shape, 64)),
            config=config,
            capture_quat=quats,
            current_heading=torch.zeros(1),
        )
        restored["metadata"] = metadata
        return restored
