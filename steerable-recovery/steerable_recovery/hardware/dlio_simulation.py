"""Actual DLIO process boundary for causal simulation sensors, never pose truth.

The ROS worker runs in its own pinned CPU container. Only raw timed Mid360 points
and IMU samples cross into DLIO. Measured joints/IMU orientation perform waist FK
on the caller side. No simulator position, velocity, floor or future state enters
the estimator. This qualifies a simulated estimator, not dynamic hardware XYZ.
"""
from __future__ import annotations

if __name__ == '__main__' and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steerable_recovery.paths import ROOT

import argparse
from collections import deque
import json
import hashlib
from pathlib import Path
import queue
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid

import numpy as np

DLIO_COMMIT = 'c8acc37100e349d70a9d8432d656cbce7e5072cd'
EPOCH_S = 1_700_000_000.


def _write(stream, header, payload=b''):
    raw = json.dumps(header, allow_nan=False, separators=(',', ':')).encode()
    stream.write(struct.pack('<II', len(raw), len(payload)) + raw + payload)
    stream.flush()


def _read_exact(stream, count):
    result = bytearray()
    while len(result) < count:
        part = stream.read(count-len(result))
        if not part:
            raise EOFError('DLIO IPC closed')
        result.extend(part)
    return bytes(result)


def _read(stream):
    nh, npayload = struct.unpack('<II', _read_exact(stream, 8))
    if not 0 < nh < 65536 or npayload > 16_000_000:
        raise ValueError('Invalid bounded DLIO packet size')
    return json.loads(_read_exact(stream, nh)), _read_exact(stream, npayload)


class G1WaistBody:
    """URDF joint-origin/axis FK for the fixed three-waist-joint sensor chain.

    This avoids loading meshes or a ROS robot model in the Isaac process. The
    transforms implement the same URDF math as master's G1Body; no measured or
    simulated body positions are accepted. Joint indices resolve by names from
    the controller contract; a reordered checkpoint is not assumed motor order.
    """
    def __init__(self, urdf, dof_names=None):
        import xml.etree.ElementTree as ET
        from scipy.spatial.transform import Rotation
        joints = {j.attrib['name']: j for j in ET.parse(urdf).getroot().findall('joint')}
        actuated = [n for n,j in joints.items() if j.attrib['type'] in ('revolute','continuous')]
        self.dof_names = list(actuated if dof_names is None else dof_names)
        if len(self.dof_names) != 29 or len(set(self.dof_names)) != 29 or set(self.dof_names) != set(actuated):
            raise ValueError('Controller joint names must match the native29 URDF')
        self.waist_indices = [self.dof_names.index('waist_'+axis+'_joint') for axis in ('yaw','roll','pitch')]
        self.chain = []
        expected = [('waist_yaw_joint', 'pelvis', 'waist_yaw_link'),
                    ('waist_roll_joint', 'waist_yaw_link', 'waist_roll_link'),
                    ('waist_pitch_joint', 'waist_roll_link', 'torso_link'),
                    ('mid360_joint', 'torso_link', 'mid360_link')]
        for name, parent, child in expected:
            j = joints[name]
            if j.find('parent').attrib['link'] != parent or j.find('child').attrib['link'] != child:
                raise ValueError('G1 pelvis-to-Mid360 chain differs from the supported URDF')
            origin = j.find('origin'); t = np.eye(4)
            t[:3, 3] = np.fromstring(origin.attrib.get('xyz', '0 0 0'), sep=' ')
            t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.attrib.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            axis = j.find('axis')
            self.chain.append((t, None if axis is None else np.fromstring(axis.attrib['xyz'], sep=' ')))
        self.set_joints(np.zeros(29))

    def set_joints(self, q):
        from scipy.spatial.transform import Rotation
        q = np.asarray(q)
        if q.shape != (29,) or not np.isfinite(q).all():
            raise ValueError('Finite native29 measured joints required')
        self.torso = np.eye(4)
        for i, (origin, axis) in enumerate(self.chain[:3]):
            turn = np.eye(4); turn[:3, :3] = Rotation.from_rotvec(axis*q[self.waist_indices[i]]).as_matrix()
            self.torso = self.torso @ origin @ turn

    set_waist_joints = set_joints

    def transform(self, child, parent):
        if (child, parent) == ('torso_link', 'pelvis'):
            return self.torso.copy()
        if (child, parent) == ('mid360_link', 'torso_link'):
            return self.chain[3][0].copy()
        raise ValueError('Only measured waist/fixed Mid360 geometry is supported')


class CausalDlioClient:
    """One estimator per episode, with finite stationary sensor pre-roll first.

    Input/output timestamps are elapsed simulation seconds (including pre-roll),
    mapped by a declared constant epoch inside ROS. Caller must pace physics when
    interpreting age as deployment latency. Wall processing/transport lag is
    reported independently; it is never converted into a hardware clock proof.
    """
    def __init__(self, *, runtime_root, urdf, dof_names, log_dir, ros_domain_id=177,
                 mount_rotation=None, startup_timeout_s=30., connect_address=None):
        from steerable_recovery.hardware.measured_pose import PelvisPoseEstimator
        root = Path(runtime_root).resolve(); logs = Path(log_dir).resolve()
        receipt_bytes=(root/'runtime_receipt.json').read_bytes();receipt=json.loads(receipt_bytes)
        if receipt.get('dlio_commit') != DLIO_COMMIT or receipt.get('schema') != 'hv2.actual_dlio_cpu_runtime.v1':
            raise ValueError('DLIO runtime requires the pinned source/build receipt')
        for rel in ('source/dlio.yaml','source/dlio_initialized_pose.patch',
                    'ws/install/direct_lidar_inertial_odometry/lib/direct_lidar_inertial_odometry/dlio_odom_node'):
            if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=receipt['files'][rel]['sha256']:
                raise ValueError(f'DLIO runtime artifact changed: {rel}')
        logs.mkdir(parents=True, exist_ok=False)
        self.estimator = PelvisPoseEstimator(G1WaistBody(urdf, dof_names), mount_rotation=mount_rotation)
        self.states = deque(maxlen=4096); self.poses = deque(maxlen=1024)
        self.lock = threading.Lock(); self.ready = threading.Event(); self.stopped = threading.Event()
        self.inputs = queue.Queue(maxsize=256); self.error = None; self.latest = None
        self.last_sample_stamp = None; self.last_imu_stamp = -float('inf'); self.last_scan_stamp = -float('inf')
        self.stream_id = uuid.uuid4().hex; self.dropped_inputs = 0; self.pose_count = 0
        self.log = open(logs/'process.log', 'xb')
        repo = ROOT
        script = '/repo/g1_deploy/steerable-recovery/steerable_recovery/hardware/dlio_simulation.py'
        command = ['singularity', 'exec', '--cleanenv', '--containall', '--no-home',
            '--bind', f'{root}:/work', '--bind', f'{repo}:/repo:ro', '--bind', f'{logs}:/output',
            '--pwd', '/work', str(root/'ros_humble_perception.sif'), '/bin/bash', '-c',
            'source /opt/ros/humble/setup.bash; source /work/ws/install/setup.bash; '
            'export ROS_LOCALHOST_ONLY=1 ROS_LOG_DIR=/output/ros OMP_NUM_THREADS=2; '
            'exec python3 "$@"', 'dlio-worker', script, '--worker', '--ros-domain-id', str(ros_domain_id)]
        self.connection = None; self.process = None
        if connect_address is None:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, start_new_session=True)
            self.input_stream, self.output_stream = self.process.stdin, self.process.stdout
        else:
            if connect_address[0] not in ('127.0.0.1', 'localhost'):
                raise ValueError('Simulation estimator IPC is localhost only')
            self.connection = socket.create_connection(tuple(connect_address), timeout=startup_timeout_s)
            self.connection.settimeout(None)
            self.input_stream = self.connection.makefile('wb'); self.output_stream = self.connection.makefile('rb')
        self.writer = threading.Thread(target=self._send_loop, daemon=True)
        self.reader = threading.Thread(target=self._receive_loop, daemon=True)
        self.writer.start(); self.reader.start()
        self.contract = dict(schema='hv2.actual_simulated_dlio.v1', actual_estimator_executed=True,
            estimator='DLIO+measured_G1_waist_FK', dlio_commit=DLIO_COMMIT,
            raw_inputs='full timed Mid360 points and 200Hz local gyro/specific force',
            raw_scan_hz=30, grouped_scans=3, dlio_scan_hz=10, odometry_publish_hz=100,
            truth_position_input=False, hardware_dynamic_accuracy_qualified=False,
            hardware_clock_qualified=False, clock_domain='simulation; fixed exact epoch mapping',
            simulated_mapping_precision_bound_s=1.e-6,
            geometry='native URDF three-waist-joint FK, fixed sensor mount', dof_names=list(dof_names),
            required_stationary_preroll_s=5., command=command, connect_address=connect_address)
        self.contract.update(runtime_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
                             runtime_receipt=str(root/'runtime_receipt.json'),
                             urdf_path=str(Path(urdf).resolve()),
                             urdf_sha256=hashlib.sha256(Path(urdf).read_bytes()).hexdigest(),
                             mount_rotation_override=None if mount_rotation is None else np.asarray(mount_rotation).tolist())
        if not self.ready.wait(startup_timeout_s):
            self.close(); raise RuntimeError(f'DLIO startup failed; inspect {logs}/process.log: {self.error}')

    def _send_loop(self):
        try:
            while True:
                item = self.inputs.get()
                if item is None: return
                _write(self.input_stream, *item)
        except (OSError, ValueError) as exc:
            if not self.stopped.is_set(): self.error = str(exc)
        finally:
            try: self.input_stream.close()
            except OSError: pass

    def _receive_loop(self):
        try:
            while not self.stopped.is_set():
                header, _ = _read(self.output_stream)
                if header['kind'] == 'ready': self.ready.set()
                elif header['kind'] == 'pose':
                    header['received_monotonic_s'] = time.monotonic()
                    with self.lock: self.poses.append(header)
        except (EOFError, OSError, ValueError) as exc:
            if not self.stopped.is_set(): self.error = str(exc)

    def _enqueue(self, header, payload=b''):
        if self.error or (self.process is not None and self.process.poll() is not None):
            raise RuntimeError(f'DLIO worker failed: {self.error}')
        header['sent_monotonic_s'] = time.monotonic()
        try: self.inputs.put_nowait((header, payload))
        except queue.Full:
            self.dropped_inputs += 1
            raise RuntimeError('DLIO raw sensor input queue overran; do not silently discard IMU')

    def push_imu(self, capture_s, gyro_rad_s, specific_force_m_s2):
        values = np.asarray([gyro_rad_s, specific_force_m_s2], dtype=float)
        if values.shape != (2, 3) or not np.isfinite(values).all() or not np.isfinite(capture_s) or capture_s <= self.last_imu_stamp:
            raise ValueError('Finite, strictly ordered raw gyro/specific force required')
        self._enqueue(dict(kind='imu', capture_s=float(capture_s), gyro=values[0].tolist(), accel=values[1].tolist()))
        self.last_imu_stamp = float(capture_s)

    def push_scan(self, capture_s, points_lidar, point_capture_s):
        p = np.asarray(points_lidar, dtype='<f4'); stamps = np.asarray(point_capture_s, dtype='<f8')
        if (p.ndim != 2 or p.shape[1] != 3 or not 0 < len(p) <= 250000 or stamps.shape != (len(p),)
                or not np.isfinite(p).all() or not np.isfinite(stamps).all()
                or not np.isfinite(capture_s) or capture_s <= self.last_scan_stamp or stamps.max() > capture_s+1.e-9):
            raise ValueError('A complete raw scan needs finite points and causal original firing times')
        self._enqueue(dict(kind='scan', capture_s=float(capture_s), count=len(p)), p.tobytes()+stamps.tobytes())
        self.last_scan_stamp = float(capture_s)

    def record_robot_state(self, capture_s, q29, pelvis_quat_xyzw, torso_quat_xyzw=None):
        state = dict(q=np.asarray(q29, dtype=float).copy(), quat_xyzw=np.asarray(pelvis_quat_xyzw, dtype=float).copy())
        if (state['q'].shape != (29,) or state['quat_xyzw'].shape != (4,)
                or not all(np.isfinite(v).all() for v in state.values())
                or not np.isfinite(capture_s) or np.linalg.norm(state['quat_xyzw']) < 1.e-6):
            raise ValueError('Finite native29 joints and measured pelvis xyzw required')
        if self.states and capture_s <= self.states[-1][0]: raise ValueError('Robot sensor timestamps must increase')
        if torso_quat_xyzw is not None: state['torso_quat_xyzw'] = np.asarray(torso_quat_xyzw, dtype=float).copy()
        self.states.append((float(capture_s), state, time.monotonic()))

    def sample(self, now_s, reset=False):
        if self.error or (self.process is not None and self.process.poll() is not None): raise RuntimeError(f'DLIO worker failed: {self.error}')
        with self.lock:
            # Never use a pose beyond the consumer's current sensor clock.
            eligible = []
            # Even sub-microsecond future timestamps stay queued: the shared
            # consumer's causal clock gate must not be bypassed by IPC jitter.
            while self.poses and self.poses[0]['capture_s'] <= now_s:
                eligible.append(self.poses.popleft())
        for pose in eligible:
            stamp = pose['capture_s']
            if self.latest is not None and stamp <= self.latest['capture_monotonic_s']: continue
            past = next((r for r in reversed(self.states) if r[0] <= stamp+1.e-6), None)
            if past is None or stamp-past[0] > .02: continue
            try:
                result = self.estimator.update(EPOCH_S+stamp, pose['position_m'], pose['quat_xyzw'], past[1])
            except ValueError as exc:
                raise RuntimeError(f'Actual DLIO pose/FK became invalid: {exc}') from exc
            # Public packet flags are Python bool, not NumPy scalar booleans:
            # packets must survive strict JSON logging/transport unchanged.
            valid = bool(result['tilt_disagreement_rad'] <= np.deg2rad(5.))
            reason = None if valid else 'tilt'
            if self.latest is not None:
                if stamp-self.latest['capture_monotonic_s'] > .1: valid = False; reason = 'gap'
                if np.linalg.norm(np.asarray(result['position_m'])-self.latest['position_m']) > .5: valid = False; reason = 'jump'
            self.latest = dict(schema='hv2.pelvis_odometry.v1', frame='pelvis_imu_gravity_frame',
                estimator='actual_simulated_DLIO+G1_waist_FK', stream_id=self.stream_id,
                capture_monotonic_s=stamp, clock_uncertainty_s=1.e-6,
                geometry_valid=valid, tracking_valid=valid, position_m=result['position_m'],
                invalid_reason=reason, **{k:v for k,v in result.items() if k != 'position_m'},
                state_pair_error_s=stamp-past[0], received_monotonic_s=pose['received_monotonic_s'],
                sensor_to_pose_wall_s=pose['received_monotonic_s']-past[2],
                actual_estimator_executed=True, hardware_clock_qualified=False,
                clock_domain='simulation', hardware_dynamic_accuracy_qualified=False)
            self.pose_count += 1
        if self.latest is None:
            return dict(schema='hv2.pelvis_odometry.v1', frame='pelvis_imu_gravity_frame',
                estimator='actual_simulated_DLIO+G1_waist_FK', stream_id=self.stream_id,
                position_m=None, capture_monotonic_s=None, tracking_valid=False, geometry_valid=False,
                clock_uncertainty_s=1.e-6, available=False, updated=False)
        result = dict(self.latest, available=True,
                      updated=bool(reset or self.latest['capture_monotonic_s'] != self.last_sample_stamp))
        result['position_m'] = [self.latest['position_m']]  # batch-one shared consumer
        self.last_sample_stamp = self.latest['capture_monotonic_s']
        return result

    def close(self):
        self.stopped.set()
        if self.connection is not None:
            try: self.connection.shutdown(socket.SHUT_RDWR)
            except OSError: pass
            self.connection.close()
        try: self.inputs.put_nowait(None)
        except queue.Full: pass
        if getattr(self, 'process', None) is not None:
            try: self.process.wait(timeout=3.)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                try: self.process.wait(timeout=3.)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL); self.process.wait()
        if getattr(self, 'log', None) is not None: self.log.close()


def _worker(domain, listen_port=None, crop_box_size=None):
    """Runs under ROS Humble system Python, with no HV2/Torch imports."""
    import os
    os.environ['ROS_DOMAIN_ID'] = str(domain)
    import rclpy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2, PointField, Imu
    from rclpy.qos import qos_profile_sensor_data
    rclpy.init(); node = rclpy.create_node('hv2_causal_dlio_input')
    points = node.create_publisher(PointCloud2, '/hv2/dlio/points', 10)
    imus = node.create_publisher(Imu, '/hv2/dlio/imu', qos_profile_sensor_data)
    server = connection = None
    input_stream, output_stream = sys.stdin.buffer, sys.stdout.buffer
    if listen_port is not None:
        server=socket.socket(); server.bind(('127.0.0.1', listen_port)); server.listen(1)
        print(f'DLIO raw IPC listening on localhost:{server.getsockname()[1]}',file=sys.stderr,flush=True)
        connection,_=server.accept(); server.close()
        input_stream,output_stream=connection.makefile('rb'),connection.makefile('wb')
    lock = threading.Lock(); incoming = queue.Queue(maxsize=1024); done = threading.Event()
    def emit(header):
        with lock: _write(output_stream, header)
    def pose(msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        stamp = msg.header.stamp.sec-EPOCH_S+msg.header.stamp.nanosec*1.e-9
        emit(dict(kind='pose', capture_s=stamp, position_m=[p.x,p.y,p.z], quat_xyzw=[q.x,q.y,q.z,q.w]))
    node.create_subscription(Odometry, '/hv2/dlio/odom', pose, 32)
    binary='/work/ws/install/direct_lidar_inertial_odometry/lib/direct_lidar_inertial_odometry/dlio_odom_node'
    overrides=[]
    if crop_box_size is not None:
        if not .1<=crop_box_size<=2.:raise ValueError('Bounded explicit ego crop required')
        overrides=['-p',f'odom/preprocessing/cropBoxFilter/size:={crop_box_size}']
    child = subprocess.Popen([binary, '--ros-args', '--params-file', '/work/source/dlio.yaml',*overrides,
        '-r', 'pointcloud:=/hv2/dlio/points', '-r', 'imu:=/hv2/dlio/imu', '-r', 'odom:=/hv2/dlio/odom'],
        stdout=sys.stderr, stderr=sys.stderr)
    def receive():
        try:
            while not done.is_set(): incoming.put(_read(input_stream), timeout=1.)
        except (EOFError, OSError, queue.Full): done.set()
    thread = threading.Thread(target=receive, daemon=True); thread.start()
    def stamp(header, t):
        ns = int(round(t*1.e9)); header.stamp.sec = int(EPOCH_S)+ns//1_000_000_000
        header.stamp.nanosec = ns%1_000_000_000; header.frame_id = 'mid360_link'
    pending_scans=[]
    try:
        emit(dict(kind='ready', estimator='actual_DLIO', dlio_commit=DLIO_COMMIT))
        while not done.is_set():
            if child.poll() is not None: raise RuntimeError('DLIO process exited')
            for _ in range(64):
                try: h, payload = incoming.get_nowait()
                except queue.Empty: break
                if h['kind'] == 'imu':
                    msg = Imu(); stamp(msg.header, h['capture_s']); msg.orientation_covariance[0] = -1.
                    msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z = h['gyro']
                    msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z = h['accel']
                    imus.publish(msg)
                elif h['kind'] == 'scan':
                    n=h['count']; p=np.frombuffer(payload, dtype='<f4', count=n*3).reshape(n,3)
                    ts=np.frombuffer(payload, dtype='<f8', count=n, offset=n*12)
                    pending_scans.append((p,ts))
                    if len(pending_scans)<3: continue
                    p=np.concatenate([v[0] for v in pending_scans]);ts=np.concatenate([v[1] for v in pending_scans])
                    pending_scans.clear();n=len(p)
                    msg=PointCloud2(); stamp(msg.header, float(ts.min())); msg.height=1; msg.width=n
                    fields=[('x',0,7),('y',4,7),('z',8,7),('intensity',12,7),('tag',16,2),('line',17,2),('timestamp',18,8)]
                    msg.fields=[PointField(name=a,offset=b,datatype=c,count=1) for a,b,c in fields]
                    msg.point_step=26; msg.row_step=n*26; msg.is_dense=True
                    dtype=np.dtype(dict(names=[v[0] for v in fields], formats=['f4']*4+['u1','u1','f8'],offsets=[v[1] for v in fields],itemsize=26))
                    data=np.zeros(n,dtype=dtype);data['x'],data['y'],data['z']=p.T
                    data['intensity']=100.;data['timestamp']=EPOCH_S*1.e9+np.round(ts*1.e9)
                    msg.data=data.tobytes();points.publish(msg)
            rclpy.spin_once(node, timeout_sec=.001)
    finally:
        done.set(); child.terminate()
        try: child.wait(timeout=3.)
        except subprocess.TimeoutExpired: child.kill(); child.wait()
        node.destroy_node(); rclpy.shutdown()
        if connection is not None: connection.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true')
    parser.add_argument('--ros-domain-id',type=int,default=177)
    parser.add_argument('--listen-port',type=int);parser.add_argument('--crop-box-size',type=float);args=parser.parse_args()
    if not args.worker: parser.error('Use CausalDlioClient; direct mode is the ROS worker only')
    _worker(args.ros_domain_id,args.listen_port,args.crop_box_size)
