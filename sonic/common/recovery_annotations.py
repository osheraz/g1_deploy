"""Manual recovery boundaries referenced to the exact replay and camera clocks."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

SCHEMA = 'hv2.recovery_interval.v1'
OUTCOMES = ('First standing', 'No standing', 'Assisted / exclude')


def session_directory(source):
    source = Path(source)
    current = source if source.is_dir() else source.parent
    for folder in (current, *current.parents):
        if (folder/'telemetry.jsonl.gz').is_file() or (folder/'state.npz').is_file():
            return folder
        if folder.name == 'sensors' and (folder/'raw/sensors.npz').is_file():
            return folder.parent
    return current


def playback_times(state):
    for key in ('playback_monotonic_s', 'capture_monotonic_s', 'receipt_monotonic_s'):
        if key in state:
            return np.asarray(state[key], dtype=np.float64)
    return np.asarray(state['t'], dtype=np.float64)


def timeline_signature(state):
    values = playback_times(state)
    if len(values) == 0 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError('Annotation needs a finite, increasing replay timeline')
    return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest()


def validate_interval(document, state):
    if document.get('schema') != SCHEMA:
        raise ValueError('Unrecognized interval format')
    if document.get('timeline_sha256') != timeline_signature(state):
        raise ValueError('Recording timeline changed; mark this recording again')
    if document.get('outcome') not in OUTCOMES:
        raise ValueError('Choose an endpoint type')
    times = playback_times(state)
    for name in ('start', 'end'):
        point = document.get(name)
        if not isinstance(point, dict):
            raise ValueError('Mark both start and standing/end before saving')
        index = point.get('sample_index')
        if type(index) is not int or not 0 <= index < len(times):
            raise ValueError(f'Invalid {name} sample')
        stamp = point.get('playback_monotonic_s')
        if not isinstance(stamp, (int, float)) or not np.isfinite(stamp) or abs(times[index]-stamp) > 1e-7:
            raise ValueError(f'{name.capitalize()} timestamp does not match the recorded sample')
    if document['end']['sample_index'] <= document['start']['sample_index']:
        raise ValueError('Standing/end must be after start')
    return document['start']['sample_index'], document['end']['sample_index']


def next_display_frame(session, index, direction):
    """Skip duplicated camera exposures; fall back to a single robot sample."""
    indices = session.get('camera_indices')
    if indices is not None:
        stop = session['n'] if direction > 0 else -1
        for candidate in range(index+direction, stop, direction):
            if indices[candidate] >= 0 and indices[candidate] != indices[index]:
                return candidate
    return min(max(index+direction, 0), session['n']-1)


class RecoveryInterval:
    def __init__(self, session):
        self.session = session
        self.path = session_directory(session['ep'])/'review/recovery_interval.json'
        self.signature = timeline_signature(session['state'])
        self.start = self.end = None
        self.outcome = OUTCOMES[0]
        self.notes = ''
        self.dirty = False
        self.status = 'Not marked'
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text())
                validate_interval(data, session['state'])
                self.start, self.end = data['start'], data['end']
                self.outcome, self.notes = data['outcome'], data.get('notes', '')
                self.status = 'Saved'
            except (OSError, ValueError, KeyError) as exc:
                self.status = f'Cannot load saved marks: {exc}'

    def point(self, index):
        state = self.session['state']
        if not 0 <= index < self.session['n']:
            raise ValueError('Replay index outside recording')
        point = dict(sample_index=int(index), replay_time_s=float(state['t'][index]),
                     playback_monotonic_s=float(playback_times(state)[index]))
        for key in ('capture_monotonic_s', 'receipt_monotonic_s'):
            if key in state:
                point[key] = float(state[key][index])
        camera = self.session.get('external_camera')
        if camera is not None:
            frame = int(self.session['camera_indices'][index])
            point['camera_frame_index'] = frame
            point['camera_exposure_monotonic_s'] = float(camera.times[frame]) if frame >= 0 else None
        return point

    def mark(self, name, index):
        if name not in ('start', 'end'):
            raise ValueError('Unknown boundary')
        setattr(self, name, self.point(index))
        self.dirty = True
        self.status = 'Unsaved changes'

    def document(self):
        return dict(schema=SCHEMA, timeline_sha256=self.signature,
                    source_name=self.session['ep'].name, recording=self.path.parent.parent.name,
                    robot_observations_present=self.session.get('qj') is not None,
                    start=None if self.start is None else dict(self.start),
                    end=None if self.end is None else dict(self.end), outcome=self.outcome, notes=self.notes,
                    author='operator', saved_at_utc=datetime.now(timezone.utc).isoformat())

    def save(self):
        document = self.document()
        validate_interval(document, self.session['state'])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=self.path.parent, prefix='.interval_', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(json.dumps(document, indent=2, allow_nan=False)+'\n')
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.dirty = False
        self.status = 'Saved'
        return self.path
