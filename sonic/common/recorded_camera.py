"""Random access to the external camera dump, using driver exposure timestamps."""
from pathlib import Path

import numpy as np


class RecordedCamera:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.raw = self.directory / 'frames.mjpg'
        with np.load(self.directory / 'frames.npz', allow_pickle=False) as index:
            self.times = np.asarray(index['ts_monotonic_s'], dtype=float)
            self.offsets = np.asarray(index['offset'], dtype=np.int64)
            self.sizes = np.asarray(index['bytes'], dtype=np.int64)
        n = len(self.times)
        if (not n or self.times.shape != (n,) or self.offsets.shape != (n,)
                or self.sizes.shape != (n,) or not np.isfinite(self.times).all()
                or np.any(np.diff(self.times) <= 0)):
            raise ValueError('Camera index requires finite, increasing exposure timestamps')
        if (self.offsets[0] != 0 or np.any(self.sizes <= 0)
                or np.any(self.offsets[1:] != (self.offsets + self.sizes)[:-1])
                or self.offsets[-1] + self.sizes[-1] > self.raw.stat().st_size):
            raise ValueError('Camera byte index does not match the saved MJPEG dump')
        self._cached_index = None
        self._cached_image = None

    def indices_at(self, times, *, max_age_s=.25):
        """Latest exposure at/before each query; blank outside coverage or in gaps."""
        times = np.asarray(times, dtype=float)
        indices = np.searchsorted(self.times, times, side='right') - 1
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
            with self.raw.open('rb') as stream:
                stream.seek(int(self.offsets[index]))
                data = stream.read(int(self.sizes[index]))
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f'Cannot decode external camera frame {index}')
            # Limit browser traffic, preserving aspect ratio. Raw video stays full resolution.
            if image.shape[1] > 960:
                image = cv2.resize(image, (960, round(image.shape[0] * 960 / image.shape[1])),
                                   interpolation=cv2.INTER_AREA)
            self._cached_index, self._cached_image = index, image
        return self._cached_image

    def validate(self, query, *, full=True):
        """Check coverage and decode a sample, or all frames when requested."""
        selected=(range(len(self.times)) if full else sorted(set([0,len(self.times)//2,len(self.times)-1])))
        for i in selected:
            self.frame(i)
        indices = self.indices_at(query)
        valid = indices >= 0
        ages = np.asarray(query)[valid] - self.times[indices[valid]]
        return dict(frames=len(self.times),decoded_frames=len(selected),validation='full' if full else 'sampled',
                    duration_s=float(self.times[-1] - self.times[0]),
                    robot_timeline_coverage=float(valid.mean()),
                    exposure_age_p95_ms=None if not len(ages) else float(np.percentile(ages,95)*1000),
                    alignment='V4L2 monotonic exposure stamps versus recorded robot timestamps; '
                              'sensor clock offsets are not independently calibrated')
