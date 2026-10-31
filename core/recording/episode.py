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


class NoControlObservations(ValueError):
    """A readable telemetry log contains no recorded policy-state samples."""


def load_native_recovery(path,*,live_estimates=False):
    """Adapt an existing diffusion telemetry log to the episode player's fields.

    No recording/conversion pass and no invented world position or camera frames.
    Joint order comes from the recorded planner contract, not a 23-DoF default.
    """
    import gzip
    path=Path(path).expanduser()
    if path.is_dir():path=path/'telemetry.jsonl.gz'
    opener=gzip.open if path.suffix=='.gz' else open
    controls=[];contract=None
    with opener(path,'rt') as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('event')=='session_contract':contract=row.get('planner_contract')
            elif row.get('event')=='control':controls.append(row)
    if not controls:raise NoControlObservations('Telemetry has no recorded control observations')
    names=[] if contract is None else contract.get('dof_names',[])
    q=np.asarray([row['q'] for row in controls],dtype=np.float32)
    quat=np.asarray([row['pelvis_quat_xyzw'] for row in controls],dtype=np.float64)
    stamps=np.asarray([row['observation_capture_s'] for row in controls],dtype=float)
    if q.ndim!=2 or len(names)!=q.shape[1] or len(set(names))!=len(names):
        raise ValueError('Telemetry must declare the measured joint order')
    if quat.shape!=(len(q),4) or not np.isfinite(q).all() or not np.isfinite(quat).all() or np.any(np.linalg.norm(quat,axis=1)<1e-6):
        raise ValueError('Invalid recorded joints or pelvis quaternion')
    if not np.isfinite(stamps).all():raise ValueError('Nonfinite control capture times')
    repeated_or_regressed=int(np.count_nonzero(np.diff(stamps)<=0))
    playback=stamps;timeline_source='observation_capture_s'
    if repeated_or_regressed:
        # A controller can reuse an observation during a sensor delay. Preserve
        # every control tick and its original capture stamp; replay on the
        # recorded host tick clock instead of dropping or retiming observations.
        playback=np.asarray([row.get('monotonic_s',np.nan) for row in controls],dtype=float)
        if not np.isfinite(playback).all() or np.any(np.diff(playback)<=0):
            raise ValueError('Repeated/regressed observations require increasing recorded control times')
        timeline_source='monotonic_s'
    quat=quat/np.linalg.norm(quat,axis=1,keepdims=True)
    for i in range(1,len(quat)):
        if np.dot(quat[i-1],quat[i])<0:quat[i]*=-1
    from scipy.spatial.transform import Rotation
    heading=np.asarray([float(np.asarray(row['causal_heading_rad']).reshape(-1)[0]) for row in controls])
    if not np.isfinite(heading).all():raise ValueError('Nonfinite recorded causal heading')
    attitude=Rotation.from_quat(quat)
    displayed=(Rotation.from_euler('z',-heading[0])*attitude).as_quat()
    relative=(attitude[0].inv()*attitude).as_quat()
    state=dict(t=playback-playback[0],qj=q,imu_quat_wxyz=quat[:,[3,0,1,2]].astype(np.float32),
               capture_monotonic_s=stamps,
               playback_monotonic_s=playback,
               receipt_monotonic_s=np.asarray([row['observation_arrival_s'] for row in controls]),
               depth_capture_monotonic_s=np.asarray([(row.get('depth_timing') or {}).get('capture_s',np.nan) for row in controls]),
               causal_heading_rad=heading,yaw_relative_rad=np.arctan2(np.sin(heading-heading[0]),np.cos(heading-heading[0])),
               orientation_relative_xyzw=relative,display_quat_wxyz=displayed[:,[3,0,1,2]])
    meta=dict(source=str(path.resolve()),physical_joint_names=names,duration_s=float(playback[-1]-playback[0]),
        timeline_source=timeline_source,nonincreasing_capture_steps=repeated_or_regressed,
        sensors=[],world_translation_measured=False,
        explanation='Measured joints and pelvis IMU orientation. Pelvis position is fixed: no XYZ source selected.')
    if live_estimates:
        poses=[r.get('pelvis_odometry') or {} for r in controls]
        if not any(p.get('position_reset_m') is not None for p in poses):
            raise ValueError('No A-relative live XYZ in this log; the estimate must be available at policy start')
        positions=np.array([p.get('position_reset_m') if p.get('position_reset_m') is not None else [np.nan]*3 for p in poses])
        geometry_valid=np.isfinite(positions).all(axis=1)&np.array([p.get('available',False) for p in poses])
        valid=geometry_valid.copy()
        # A tilt rejection is not a missing measurement. Preserve the finite
        # contemporaneous estimate for diagnosis, using exactly the logged A
        # origin. This never changes availability or qualification for steering.
        origin=np.asarray(poses[0].get('position_m'),dtype=float)
        first_relative=poses[0].get('position_reset_m')
        if (origin.shape==(3,) and np.isfinite(origin).all() and geometry_valid[0]
                and first_relative is not None and np.allclose(first_relative,0.,atol=1e-8)):
            transform=Rotation.from_euler('z',-heading[0]).as_matrix()
            for i,p in enumerate(poses):
                age=p.get('age_s');raw=np.asarray(p.get('position_m'),dtype=float)
                if (not valid[i] and p.get('invalid_reason') in ('tilt','geometry_invalid')
                        and isinstance(age,(int,float)) and 0<=age<=.05
                        and p.get('stream_id')==poses[0].get('stream_id')
                        and raw.shape==(3,) and np.isfinite(raw).all()):
                    positions[i]=transform@(raw-origin);valid[i]=True
        positions[~valid]=np.nan
        state.update(pelvis_position_relative_m=positions,position_valid=valid,
            position_geometry_valid=geometry_valid,position_tracking_qualified=np.array([p.get('tracking_valid',False) for p in poses]))
        meta.update(world_translation_measured=True,position_origin_delay_s=0.,explanation=
            'Actual live A-relative XYZ, joints and pelvis IMU orientation. No offline interpolation. '
            'Fresh finite estimates rejected for tilt remain visible in orange; these are not valid steering feedback. '
            'Missing, stale or restarted estimates still break the trace. Fixed world camera. '
            'XYZ and timing remain unqualified unless the recorded status says otherwise; no independent position truth.')
    if repeated_or_regressed:
        meta['explanation']+=f' {repeated_or_regressed} reused/regressed observation timestamps; '
        meta['explanation']+='playback uses recorded control times and retains the original observations.'
    return state,lambda _: (None,None),meta


def attach_measured_translation(state,meta,path,*,max_gap_s=.25,allow_late_origin=False):
    """Offline interpolation in the provider's IMU world frame, never extrapolation.

    The bridge/evaluator maps LiDAR stamps onto workstation receipt time. Pair to
    observation *arrival*, not the separately corrected robot capture clock.
    Missing samples hide the robot; geometric disagreement is shown as a diagnostic.
    """
    with open(path) as stream:rows=[json.loads(line) for line in stream if line.strip()]
    rows=[r for r in rows if r.get('frame')=='pelvis_imu_gravity_frame']
    if len(rows)<2:raise ValueError('Need at least two pelvis estimates in pelvis_imu_gravity_frame')
    t=np.asarray([r['capture_monotonic_s'] for r in rows],float)
    p=np.asarray([r['position_m'] for r in rows],float)
    geometry=np.asarray([r.get('geometry_valid',r.get('tracking_valid',False)) for r in rows],bool)
    if p.shape!=(len(t),3) or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Measured pose timestamps must be finite and strictly increasing')
    good=np.isfinite(p).all(axis=1)
    query=np.asarray(state['receipt_monotonic_s'])
    right=np.clip(np.searchsorted(t,query,side='left'),1,len(t)-1);left=right-1
    valid=(query>=t[0])&(query<=t[-1])&good[left]&good[right]&((t[right]-t[left])<=max_gap_s)
    if not valid.any():raise ValueError('No valid measured translation overlaps the robot timeline')
    if not valid[0] and not allow_late_origin:
        raise ValueError('No valid measured translation at policy start; cannot set episode origin')
    origin=int(np.flatnonzero(valid)[0])
    values=np.column_stack([np.interp(query,t,p[:,k]) for k in range(3)])
    values-=values[origin]
    # Match steering's fixed gravity-aligned episode axes, not arbitrary IMU yaw.
    h0=float(state['causal_heading_rad'][0]);c,s=np.cos(h0),np.sin(h0)
    values=values @ np.array([[c,s,0.],[-s,c,0.],[0.,0.,1.]]).T
    values[~valid]=np.nan
    state['pelvis_position_relative_m']=values;state['position_valid']=valid
    state['position_geometry_valid']=geometry[left]&geometry[right]&valid
    delay=float(query[origin]-query[0])
    meta.update(world_translation_measured=True,translation_source=str(path),position_origin_delay_s=delay,
        explanation=('XYZ is an offline LiDAR/FK estimate. '+
        ('Position origin at policy start. ' if origin==0 else
         f'POSITION ORIGIN IS LATE: +{delay:.2f} s after policy start. Displacement before this is unknown; this is NOT XYZ since A. ')+
        'XY aligned to its recorded initial causal heading, Z up. '
        'Joints and orientation come from robot telemetry. Fixed world camera; grid is a ruler, not measured ground. '
        'Missing estimates hide the robot; tilt disagreement is flagged but remains visible for diagnosis. '
        'Absolute clock timing and dynamic position accuracy remain unqualified.'))


def load_raw_motion(path):
    """Replay a sensor-only capture without inventing a policy telemetry log."""
    import torch
    from scipy.spatial.transform import Rotation
    from humanoidverse2.observations.streams import G1_MOTOR_NAMES
    from humanoidverse2.observations.heading import replay_heading
    path=Path(path);data=np.load(path/'sensors.npz')
    report=json.loads((path/'summary.json').read_text())
    times=data['state_receipt_monotonic_s']
    if len(times)<2 or np.any(np.diff(times)<=0):raise ValueError('Need increasing raw state receipt times')
    start=report.get('trigger',{}).get('request_monotonic_s',times[0])
    query=np.arange(max(start,times[0]),times[-1],.02)
    indices=np.maximum(np.searchsorted(times,query,side='right')-1,0)
    if len(query)<2:raise ValueError('Insufficient raw state after the recording trigger')
    quat=data['pelvis_quat_wxyz'][indices][:,[1,2,3,0]]
    quat=quat/np.linalg.norm(quat,axis=1,keepdims=True)
    heading=replay_heading(torch.from_numpy(quat)).numpy()
    attitude=Rotation.from_quat(quat)
    displayed=(Rotation.from_euler('z',-heading[0])*attitude).as_quat()
    relative=(attitude[0].inv()*attitude).as_quat()
    state=dict(t=query-query[0],qj=data['q'][indices],imu_quat_wxyz=quat[:,[3,0,1,2]],
        receipt_monotonic_s=query,causal_heading_rad=heading,
        yaw_relative_rad=np.arctan2(np.sin(heading-heading[0]),np.cos(heading-heading[0])),
        orientation_relative_xyzw=relative,display_quat_wxyz=displayed[:,[3,0,1,2]])
    meta=dict(source=str(path.resolve()),physical_joint_names=list(G1_MOTOR_NAMES),duration_s=float(state['t'][-1]),
        sensors=[],world_translation_measured=False,
        explanation='Sensor-only recording. Joint/IMU samples replayed at 50 Hz using receipt times; '
        'origin is the recording trigger. No policy was run. Translation requires measured pose input.')
    return state,lambda _: (None,None),meta
