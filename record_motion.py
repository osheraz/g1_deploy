"""Record recovery sensors and measured pelvis motion alongside normal deployment.

Read-only robot subscriptions. Owns only its sensor/estimator child processes.
The planner keeps depth. Optional live odometry publishes to the runtime receiver.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
RESULTS_ROOT=ROOT/'logs/real_world'
DEFAULT_WS=str(Path.home()/'.local/share/hv2/lidar/ros2_ws')


class RecorderProcesses:
    """Shared process ownership for manual recording and deployment-triggered capture."""
    def __init__(self):
        self.children=[];self.logs=[]

    def launch(self,plan,output,workspace,*,controlled=False):
        env_script='set -e; source /opt/ros/humble/setup.bash; source "$1"; shift; export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; export PYTHONPATH=/home/osher/unitree_sdk2_python:${PYTHONPATH:-}; exec "$@"'
        for name,command in plan:
            log=open(output/(name+'.log'),'w');self.logs.append(log)
            proc=subprocess.Popen(['nice','-n','10','bash','-c',env_script,'motion-env',str(workspace/'install/setup.bash'),*command],
                stdin=subprocess.PIPE if controlled and name in ('capture','camera','pelvis') else subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            self.children.append((name,proc))

    def check(self,optional=()):
        """Raise for a required child that exited; return the optional ones that did."""
        exited=[]
        for name,proc in self.children:
            if proc.poll() is None:continue
            if name in optional:exited.append(name)
            else:raise RuntimeError(f'Sensor {name} exited ({proc.returncode}); inspect its recording log')
        return exited

    def close(self):
        # The policy has already closed before deployment calls this. A second
        # Ctrl+C must not interrupt the recorder's final compressed write.
        previous=signal.signal(signal.SIGINT,signal.SIG_IGN)
        try:self._close()
        finally:signal.signal(signal.SIGINT,previous)

    def _close(self):
        # Stop video/pose now; raw sensor compression below can take seconds.
        for name,proc in self.children:
            if name in ('camera','pelvis') and proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGINT)
                except ProcessLookupError:pass
        for name,proc in reversed(self.children):
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGINT)
                except ProcessLookupError:pass
                if name=='capture':
                    elapsed=0
                    while True:
                        try:proc.wait(timeout=2);break
                        except subprocess.TimeoutExpired:
                            elapsed+=2
                            print(f'Saving sensors ({elapsed}s; recorder PID {proc.pid}). '
                                  'The policy is already stopped.' if elapsed<=4 else
                                  f'Sensor recorder still saving ({elapsed}s; PID {proc.pid}); preserving its unsaved data.',flush=True)
                else:
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            if proc.stdin is not None:proc.stdin.close()
        for log in self.logs:log.close()


class PolicyRecording:
    """Prepare off-thread sensor capture; policy start writes one small pipe message."""
    def __init__(self,output,*,interface,seconds,existing_driver=False,label='supported_hoist',live_odometry=None,
                 automatic_start=False,camera=None):
        self.output=Path(output);self.started=False;self.processes=RecorderProcesses()
        self.error=None;self.monitor_stop=threading.Event();self.monitor=None
        self.args=SimpleNamespace(interface=interface,seconds=seconds,label=label,
            existing_driver=existing_driver,record_only=True,mount_calibration=None,controlled=True,
            driver_config=ROOT/'g1_deploy/config/lidar/MID360_bluefin.json',
            camera_device=None if camera is None else camera['device'],
            camera_size=None if camera is None else camera['size'],camera_fps=None if camera is None else camera['fps'])
        self.live_odometry=live_odometry
        self.automatic_start=automatic_start
        self.camera=camera;self.camera_error=None;self.trigger=None
        if live_odometry is not None:
            self.args.record_only=False;self.args.scan_packets=3
            self.args.dlio_config=ROOT/'g1_deploy/config/lidar/dlio.yaml'
            self.args.mount_calibration=Path(live_odometry['mount_calibration'])
            self.args.odometry_port=int(live_odometry['port'])
            self.args.clock_calibration=live_odometry.get('clock_calibration')
        self.workspace=Path(os.environ.get('LIDAR_WS',DEFAULT_WS))

    def prepare(self):
        """Launch only, overlapping sensor discovery with model loading."""
        try:self._launch()
        except Exception as exc:
            self.error=str(exc)
            print(f'Recorder unavailable: {exc}. Policy startup continues.',flush=True)

    def wait_until_ready(self,check):
        """One pre-start pause before policy/positioning and the episode clock.

        check keeps robot/depth health and operator cancellation responsive.
        This method is never called from the running policy or start trigger.
        """
        if not self.processes.children:
            raise RuntimeError(f'Recorder cannot warm up: {self.error or "capture process unavailable"}')
        def status(path):
            try:return json.loads(path.read_text())
            except (FileNotFoundError,json.JSONDecodeError):return {}
        began=time.monotonic();ready_since=None;previous=None
        print('Startup: waiting for sensor streams, then 5 seconds of warmup (no media saved). '
              'Keep still. '+('Ctrl+C cancels.' if self.automatic_start else
              'A/START are ignored until ready; SELECT or Ctrl+C cancels.'),flush=True)
        while True:
            check()
            self.processes.check()  # failed capture cannot satisfy requested readiness
            now=time.monotonic()
            raw=status(self.output/'raw/status.json')
            missing=list(raw.get('missing',[]))
            if not raw.get('ready') or now-raw.get('monotonic_s',-float('inf'))>2.:
                if not missing:missing=['raw sensor stream status']
            if self.camera is not None:
                camera=status(self.output/'camera/status.json')
                if not camera.get('ready') or now-camera.get('arrival_monotonic_s',-float('inf'))>2.:
                    missing.append('external camera frames')
            if self.live_odometry is not None:
                pose=status(self.output/'measured_pose.status.json')
                if not pose.get('geometry_valid') or now-pose.get('arrival_monotonic_s',-float('inf'))>=1.:
                    missing.append('live XYZ initialization')
            if missing:
                ready_since=None
                message='Startup waiting for '+', '.join(missing)+'. Keep still; policy has not started.'
            else:
                if ready_since is None:ready_since=now
                remaining=max(0.,5.-(now-ready_since))
                if remaining==0.:
                    (self.output/'startup_warmup.json').write_text(json.dumps(dict(
                        ready_monotonic_s=now,streams_ready_since_s=ready_since,
                        warmup_s=now-ready_since,pre_roll_s=0.,wait_s=now-began),indent=2)+'\n')
                    print('Recording warmup complete. Checking policy readiness...',flush=True)
                    return
                message=f'Startup sensor warmup: {math.ceil(remaining)} s remaining. Keep still.'
            if message!=previous:print(message,flush=True);previous=message
            time.sleep(.1)

    def _launch(self):
        if not (self.workspace/'install/setup.bash').is_file():
            raise RuntimeError(f'Missing ROS workspace {self.workspace}; set LIDAR_WS or use --no-record-sensors')
        self.output.mkdir(parents=True,exist_ok=False)
        if self.camera is not None:(self.output/'camera').mkdir()
        plan=commands(self.args,self.output);plan[-1][1].append('--controlled')
        camera=None if self.camera is None else dict(self.camera,raw='camera/frames.mjpg',index='camera/frames.npz',
            video='camera/external.mkv (remuxed after stop; uniform frame timeline, use frames.npz for timing)',
            timestamps='V4L2 start-of-exposure CLOCK_MONOTONIC, same domain as time.monotonic() in this session',
            launch_monotonic_s=time.monotonic())
        (self.output/'session.json').write_text(json.dumps(dict(commands=plan,label=self.args.label,
            motor_publisher=False,trigger='sensor-only inference start' if self.automatic_start else 'accepted policy A',planner_terrain='depth',
            policy_session=str(self.output.parent),raw_only=self.args.record_only,ros_workspace=str(self.workspace.resolve()),
            live_odometry=self.live_odometry,camera=camera),indent=2)+'\n')
        print('Starting recorder in background; loading policy without waiting for it.',flush=True)
        self.processes.launch(plan,self.output,self.workspace,controlled=True)
        for _,proc in self.processes.children:
            if proc.stdin is not None:os.set_blocking(proc.stdin.fileno(),False)
        self.monitor=threading.Thread(target=self._monitor,daemon=True);self.monitor.start()
        print(('Recording starts with sensor-only inference.' if self.automatic_start else 'A starts policy + recording.')+
            f' Raw sensors: {self.output}',flush=True)

    def _camera_state(self,exited,previous_size):
        raw=self.output/'camera/frames.mjpg'
        size=raw.stat().st_size if raw.is_file() else 0
        if 'camera' in exited:
            self.camera_error=self.camera_error or 'capture process exited; inspect camera.log'
            return 'External camera stopped: '+self.camera_error,size
        if not self.started:
            path=self.output/'camera/status.json'
            status=json.loads(path.read_text()) if path.is_file() else {}
            ready=status.get('ready') and time.monotonic()-status.get('arrival_monotonic_s',0)<2.
            return ('External camera ready; recording waits for A.' if ready else 'External camera opening.'),size
        if size>previous_size:return f'External camera recording ({size>>20} MB).',size
        return ('External camera opening.' if size==0 else 'External camera waiting for file flush (buffering or stalled input).'),size

    def _monitor(self):
        previous=None;camera_previous=None;camera_size=0;camera_message=''
        while not self.monitor_stop.wait(.5):
            try:
                if self.trigger is not None and not (self.output/'policy_start.json').is_file():
                    (self.output/'policy_start.json').write_text(json.dumps(self.trigger,indent=2)+'\n')
                exited=self.processes.check(optional=('camera',))
                if self.camera is not None:
                    camera_message,camera_size=self._camera_state(exited,camera_size)
                    if camera_message.split('(')[0]!=(camera_previous or '').split('(')[0]:
                        print(camera_message,flush=True);camera_previous=camera_message
                path=self.output/'raw/status.json'
                if path.is_file():
                    status=json.loads(path.read_text())
                    missing=status['missing']
                    message=('Recorder: robot + LiDAR streams ready.' if not missing else
                        'Recorder waiting for '+', '.join(missing)+'. Policy is unaffected; incomplete sensors cannot validate XYZ.')
                else:message='Recorder initializing in background; policy is unaffected.'
                if self.live_odometry is not None:
                    path=self.output/'measured_pose.status.json'
                    pose=json.loads(path.read_text()) if path.is_file() else {}
                    fresh=time.monotonic()-pose.get('arrival_monotonic_s',0)<1.
                    if fresh and pose.get('geometry_valid'):
                        message+=' Live XYZ ready'+(' (qualified).' if pose.get('tracking_valid') else ' for monitoring (clock unqualified).')
                    else:message+=' Live XYZ initializing; keep the robot still.'
            except Exception as exc:
                self.error=str(exc)
                message=f'Recorder error: {exc}. Policy is unaffected.'
            if message!=previous:print(message,flush=True);previous=message

    def start(self):
        if self.started:return
        try:
            if not self.processes.children or self.processes.children[-1][0]!='capture':
                raise RuntimeError(self.error or 'capture process unavailable')
            proc=self.processes.children[-1][1]
            if proc.poll() is not None:raise RuntimeError('capture process exited; inspect capture.log')
            now=time.monotonic();wall=time.time()
            message=(json.dumps(dict(command='start',monotonic_s=now))+'\n').encode()
            os.write(proc.stdin.fileno(),message)
            for name,child in self.processes.children:
                if name in ('camera','pelvis') and child.stdin is not None:
                    try:os.write(child.stdin.fileno(),message)
                    except OSError as exc:
                        if name=='camera':self.camera_error=str(exc)
                        print(f'{name} recording trigger failed: {exc}',flush=True)
            self.trigger=dict(monotonic_s=now,wall_s=wall)  # written by the monitor thread, not here
            self.started=True
            print('Recording triggered with policy.',flush=True)
        except Exception as exc:
            self.error=str(exc)
            print(f'Recording could not start: {exc}. Policy continues; raw sensor recording is unavailable.',flush=True)

    def close(self):
        self.monitor_stop.set()
        if self.monitor is not None:self.monitor.join(timeout=1.)
        if self.started:print('Policy stopped. Saving raw sensors; wait for Saved and verified...',flush=True)
        self.processes.close()
        if not self.output.exists():return
        # A quick start/stop can finish before the monitor's first wakeup.
        if self.trigger is not None:
            (self.output/'policy_start.json').write_text(json.dumps(self.trigger,indent=2)+'\n')
        error=self.error or (capture_saved(self.output) if self.started else None)
        if self.started and error is None:
            capture=self.processes.children[-1][1]
            if capture.returncode:error=f'Capture exited with status {capture.returncode}; inspect capture.log'
        camera=None
        if self.camera is not None and self.started:
            camera=camera_report(self.output,self.camera['fps'],self.camera_error,
                None if self.trigger is None else self.trigger['monotonic_s'])
            if camera['error'] and error is None:error='External camera: '+camera['error']
        (self.output/'recording_result.json').write_text(json.dumps(dict(
            started=self.started,complete=self.started and error is None,error=error,camera=camera),indent=2)+'\n')
        if error:print(f'INCOMPLETE sensor recording: {error}. See {self.output}/capture.log',flush=True)
        elif self.started:print(f'Saved and verified raw sensors: {self.output}',flush=True)
        else:print('Recorder closed: policy A was not pressed; no motion capture started.',flush=True)
        if camera is not None and not camera['error']:
            print(f"External camera: {camera['frames']} frames over {camera['duration_s']:.1f} s, "
                  f"{camera['dropped']} dropped, interval {camera['interval_ms']['mean']:.1f} ms "
                  f"(max {camera['interval_ms']['max']:.1f}); video {camera['video'] or 'not remuxed'}.",flush=True)


def camera_report(output,fps,error=None,policy_start_monotonic_s=None):
    """After the policy stopped: index the raw dump from the driver log, cut a truncated tail,
    remux to a playable MKV. Drops are driver sequence-number gaps, so zero is exact."""
    import re
    import numpy as np
    if error:return dict(error=error)
    raw,log=output/'camera/frames.mjpg',output/'camera.log'
    if not raw.is_file() or not raw.stat().st_size or not log.is_file():return dict(error='no camera frames written')
    text=log.read_text()
    rows=re.findall(CAMERA_FRAME_PATTERN,text)
    if not rows:return dict(error='camera.log holds no frames; inspect it')
    seq=np.array([int(r[0]) for r in rows]);size=np.array([int(r[1]) for r in rows]);ts=np.array([float(r[2]) for r in rows])
    offset=np.concatenate([[0],np.cumsum(size)[:-1]])
    complete=int(np.searchsorted(offset+size,raw.stat().st_size,side='right'))
    truncated=complete<len(rows)
    if truncated:
        seq,size,ts,offset=seq[:complete],size[:complete],ts[:complete],offset[:complete]
        if complete:os.truncate(raw,int(offset[-1]+size[-1]))
    if len(seq)==0:return dict(error='no complete camera frame')
    gaps=np.diff(ts)
    measured=float(1/gaps[1:].mean()) if len(gaps)>1 else None
    # The driver clamps the request to a mode it offers; its own message is not retained in the
    # log, so the effective rate is the accepted rate when logged, else the measured cadence.
    rates=re.findall(CAMERA_RATE_PATTERN,text)
    requested_fps=fps
    fps=float(rates[-1]) if rates else (float(round(measured)) if measured else float(fps))
    np.savez(output/'camera/frames.npz',seq=seq,bytes=size,offset=offset,ts_monotonic_s=ts,fps=fps,
        policy_start_monotonic_s=np.nan if policy_start_monotonic_s is None else policy_start_monotonic_s)
    try:
        remux=subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-f','mjpeg','-framerate',f'{fps:g}',
            '-i',str(raw),'-c:v','copy','-f','matroska',str(output/'camera/external.mkv')],capture_output=True,text=True,timeout=120)
    except (OSError,subprocess.TimeoutExpired) as exc:
        remux=subprocess.CompletedProcess([],1,'',f'{type(exc).__name__}: {exc}')
    index_at_start=None if policy_start_monotonic_s is None else int(np.searchsorted(ts,policy_start_monotonic_s))
    return dict(error=None,frames=int(len(seq)),dropped=int((np.diff(seq)-1).sum()) if len(seq)>1 else 0,
        duration_s=float(ts[-1]-ts[0]),fps=fps,requested_fps=requested_fps,measured_fps=measured,
        warmup_s=float(gaps[0]) if len(gaps) else None,
        interval_ms=dict(mean=float(gaps[1:].mean()*1000) if len(gaps)>1 else None,
                         max=float(gaps[1:].max()*1000) if len(gaps)>1 else None),
        truncated_last_frame=bool(truncated),bytes=int(raw.stat().st_size),
        video=None if remux.returncode else 'camera/external.mkv',
        remux_error=remux.stderr.strip() if remux.returncode else None,
        frame_index_at_policy_start=index_at_start,
        video_seconds_at_policy_start=None if index_at_start is None else index_at_start/fps)


ARTIFACTS=[
    ('session.json','policy session receipt: model, backend, options, hashes'),
    ('resolved.json','fully resolved runtime configuration'),
    ('controller_release.json','low-level control was taken from the Unitree controller'),
    ('telemetry.jsonl.gz','50 Hz policy telemetry: state, actions, plans, timing, live XYZ'),
    ('model_trace','exact neural inputs/noise/outputs, issued commands, source snapshot and replay provenance'),
    ('COMPLETE.json','run finished: tick counts, control/latency percentiles, telemetry status'),
    ('STOPPED.json','run stopped by the operator (A, SELECT, Ctrl+C)'),
    ('FAILED.json','run aborted with an error'),
    ('live_evaluation/evaluation.json','offline evaluation of the recorded motion'),
    ('sensors/session.json','recorder plan, trigger and camera contract'),
    ('sensors/policy_start.json','A press in monotonic and wall clocks'),
    ('sensors/startup_warmup.json','completed sensor pre-roll before operator readiness or sensor-only inference'),
    ('sensors/recording_result.json','recorder completeness, errors, camera frame report'),
    ('sensors/raw/sensors.npz','raw LiDAR, Livox IMU, low state q/dq/tau_est, pelvis and torso IMUs with receipt times'),
    ('sensors/raw/summary.json','sensor counts, gaps, capture errors, trigger times'),
    ('sensors/raw/bag','timestamped LiDAR and Livox IMU ROS bag for offline odometry'),
    ('sensors/raw/odometry.npz','live DLIO odometry samples, if live odometry ran'),
    ('sensors/measured_pose.jsonl','live pelvis XYZ estimates delivered to the policy'),
    ('sensors/camera/frames.mjpg','external camera raw MJPEG frames straight from the driver'),
    ('sensors/camera/frames.npz','per-frame driver sequence, byte offset and start-of-exposure monotonic time'),
    ('sensors/camera/external.mkv','playable remux of the raw frames (uniform frame timeline)'),
    ('sensors/camera.log','camera driver log with one line per frame'),
    ('sensors/depth/metadata.json','metric depth frame contract'),
    ('sensors/depth/frames.bin','calibrated float32 depth and validity masks'),
    ('sensors/depth/frames.jsonl','depth capture times and byte index'),
    ('sensors/depth/summary.json','depth recording completeness and queue drops'),
    ('sensors/capture.log','raw recorder log'),
    ('sensors/driver.log','Livox driver log'),
    ('sensors/raw/capture.json','raw recorder configuration and clock semantics'),
    ('sensors/raw/status.json','last recorder stream status'),
    ('sensors/raw/ready.json','when the sensor streams became ready'),
]


def session_manifest(session):
    """List what a policy session saved, in one place, and write it as manifest.json."""
    session=Path(session)
    def size(path):
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) if path.is_dir() else path.stat().st_size
    rows=[dict(path=rel,bytes=size(session/rel),what=what) for rel,what in ARTIFACTS if (session/rel).exists()]
    known={r['path'] for r in rows}
    for path in sorted(session.rglob('*')):
        rel=str(path.relative_to(session))
        if path.is_file() and not any(rel==k or rel.startswith(k+'/') for k in known) and rel!='manifest.json':
            rows.append(dict(path=rel,bytes=path.stat().st_size,what=''))
    (session/'manifest.json').write_text(json.dumps(dict(session=str(session),artifacts=rows),indent=2)+'\n')
    width=max(len(r['path']) for r in rows)
    lines=[f'Session saved: {session}']+[f"  {r['path']:{width}s}  {r['bytes']/1e6:8.1f} MB  {r['what']}" for r in rows]
    lines.append(f'  {"manifest.json":{width}s}  {"":8s}     this list')
    print('\n'.join(lines),flush=True)


def capture_saved(output):
    """Success means finalized artifacts, not merely an existing output folder."""
    required=[output/'raw/sensors.npz',output/'raw/summary.json',output/'raw/bag/metadata.yaml']
    missing=[str(p) for p in required if not p.is_file() or not p.stat().st_size]
    if missing:return 'Missing capture artifacts: '+', '.join(missing)
    report=json.loads((output/'raw/summary.json').read_text())
    if report.get('error'):return 'Capture reported: '+report['error']
    for name in ('scans','lidar_imu','lowstate'):
        if not report.get(name,{}).get('samples',0):return f'No recorded {name} samples'
    return None


CAMERA_FRAME_PATTERN=r'dqbuf: \d+ seq:\s+(\d+) bytesused: (\d+) ts: ([\d.]+)'
CAMERA_RATE_PATTERN=r'Frame rate set to ([\d.]+) fps'
DEFAULT_CAMERA='/dev/v4l/by-id/usb-046d_Brio_101_2540APD0M3H8-video-index0'


def camera_modes(device):
    """MJPG (width, height, fps) triples the driver offers, from v4l2-ctl --list-formats-ext."""
    import re
    text=subprocess.run(['v4l2-ctl','-d',device,'--list-formats-ext'],capture_output=True,text=True).stdout
    block=text.split("'MJPG'",1)[1] if "'MJPG'" in text else ''
    block=re.split(r"\n\s*\[\d+\]:",block)[0]
    modes=[];size=None
    for line in block.splitlines():
        found=re.search(r'Size: Discrete (\d+)x(\d+)',line)
        if found:size=(int(found[1]),int(found[2]))
        rate=re.search(r'\(([\d.]+) fps\)',line)
        if rate and size:modes.append((size[0],size[1],float(rate[1])))
    return modes


def camera_command(device,size,fps,target):
    """Kernel-only capture: the driver's MJPEG frames go straight to `target` ('-' for stdout);
    each frame's start-of-exposure CLOCK_MONOTONIC stamp and sequence number go to stderr.
    ffmpeg's v4l2 input was measured at ~30% of a core spinning on non-blocking DQBUF."""
    width,height=size.split('x')
    return ['bash','-c',
        f'v4l2-ctl -d "$1" -c exposure_dynamic_framerate=0; exec v4l2-ctl -d "$1" '
        f'--set-fmt-video=width={int(width)},height={int(height)},pixelformat=MJPG --set-parm={int(fps)} '
        f'--stream-mmap=4 --stream-to="$2" --verbose','camera',device,target]


def commands(args, output):
    hw=ROOT/'motion_diffusion/behavioral/hardware'
    validation=ROOT/'scripts/fall_recovery/validation'
    result=[]
    if not args.existing_driver:
        result.append(('driver',['ros2','run','livox_ros_driver2','livox_ros_driver2_node','--ros-args',
            '-p','xfer_format:=0','-p','multi_topic:=0','-p','data_src:=0','-p','publish_freq:=30.0',
            '-p','output_data_type:=0','-p','frame_id:=lidar','-p','user_config_path:='+str(args.driver_config.resolve()),
            '-p','cmdline_input_bd_code:=livox0000000001']))
    if not args.record_only:
        result.extend([
            ('imu_si',['/usr/bin/python3',str(hw/'livox_imu_si.py'),'--input-unit','g']),
            ('sweeps',['/usr/bin/python3',str(hw/'livox_cloud_window.py'),'--input-topic','/livox/lidar',
                '--packets',str(args.scan_packets)]),
            ('dlio',['ros2','run','direct_lidar_inertial_odometry','dlio_odom_node','--ros-args',
                '--params-file',str(args.dlio_config.resolve()),'-r','pointcloud:=/livox/lidar_dlio',
                '-r','imu:=/livox/imu_si','-r','odom:=/dlio/odom_node/odom']),
            ('pelvis',['/usr/bin/python3',str(hw/'pelvis_odometry.py'),'bridge','--interface',args.interface,
                '--record',str(output/'measured_pose.jsonl')]),
        ])
    if args.mount_calibration is not None and not args.record_only:
        result[-1][1].extend(['--mount-calibration',str(args.mount_calibration.resolve())])
    if getattr(args,'odometry_port',None) is not None and not args.record_only:
        result[-1][1].extend(['--odometry-port',str(args.odometry_port),'--publish-diagnostics'])
    if getattr(args,'clock_calibration',None) is not None and not args.record_only:
        result[-1][1].extend(['--clock-calibration',str(Path(args.clock_calibration).resolve())])
    if getattr(args,'controlled',False) and not args.record_only:
        result[-1][1].append('--controlled-record')
    if args.camera_device:
        if getattr(args,'controlled',False):
            result.append(('camera',['/usr/bin/python3',str(ROOT/'g1_deploy/record_camera.py'),
                '--device',args.camera_device,'--size',args.camera_size,'--fps',str(args.camera_fps),
                '--output',str(output/'camera')]))
        else:
            result.append(('camera',camera_command(args.camera_device,args.camera_size,args.camera_fps,
                str(output/'camera/frames.mjpg'))))
    result.append(('capture',['/usr/bin/python3',str(validation/'record_lidar_live.py'),
        '--interface',args.interface,'--seconds',str(args.seconds),'--pose',args.label,'--output',str(output/'raw')]))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=float,default=90.);p.add_argument('--label',default='diffusion_recovery')
    p.add_argument('--output',type=Path);p.add_argument('--interface',default='enp132s0')
    p.add_argument('--ros-workspace',type=Path,default=os.environ.get('LIDAR_WS',DEFAULT_WS))
    p.add_argument('--driver-config',type=Path,default=ROOT/'g1_deploy/config/lidar/MID360_bluefin.json')
    p.add_argument('--dlio-config',type=Path,default=ROOT/'g1_deploy/config/lidar/dlio.yaml')
    p.add_argument('--mount-calibration',type=Path,help='Explicit fixed-mount calibration candidate; recorded for comparison')
    p.add_argument('--existing-driver',action='store_true',help='Subscribe to an already-running driver; do not start a second one')
    p.add_argument('--scan-packets',type=int,default=3,help='3 for 30 Hz raw input; 5 for 50 Hz input gives 100 ms sweeps')
    p.add_argument('--record-only',action='store_true',help='Raw capture only; replay DLIO offline later')
    p.add_argument('--camera-device',nargs='?',const=DEFAULT_CAMERA,
        help=f'External USB camera; raw MJPEG dump plus driver timestamps. No value selects {DEFAULT_CAMERA}')
    p.add_argument('--camera-size',default='1920x1080');p.add_argument('--camera-fps',type=int,default=30)
    p.add_argument('--dry-run',action='store_true',help='Print child commands without starting processes or contacting hardware')
    a=p.parse_args()
    if not 0<a.seconds<=300 or not 1<=a.scan_packets<=10:p.error('Seconds must be 0..300; scan-packets 1..10')
    output=(a.output or RESULTS_ROOT/f'sensors_{time.time_ns()}').resolve()
    plan=commands(a,output)
    if a.dry_run:
        print(json.dumps(dict(output=str(output),commands=plan,motor_publisher=False),indent=2));return
    required=[a.ros_workspace/'install/setup.bash']
    if not a.existing_driver:required.append(a.driver_config)
    if not a.record_only:required.append(a.dlio_config)
    for path in required:
        if not path.is_file():p.error(f'Missing {path}; set --ros-workspace / LIDAR_WS if the ROS workspace moved')
    output.mkdir(parents=True,exist_ok=False)
    if a.camera_device:(output/'camera').mkdir()
    (output/'session.json').write_text(json.dumps(dict(label=a.label,commands=plan,
        start_monotonic_s=time.monotonic(),motor_publisher=False,head_mount='fixed',ros_workspace=str(a.ros_workspace.resolve()),
        clock_qualified=False,planner_terrain='depth',
        estimator_input='raw scans with existing DLIO crop; raw recordings retained for self-filter comparisons'),indent=2)+'\n')
    processes=RecorderProcesses();children=processes.children;error=None
    print(f'Sensor recording: {output}\n'
          'Terminal 1: load your normal diffusion deployment and wait at the operator-button prompt.\n'
          'Terminal 2: this recorder. Once streams appear, keep still for 5 seconds, then start recovery with the normal operator control.\n'
          'Keep still for 5 seconds afterward. Stop the policy before ending recording (final compression uses CPU).\n'
          +('Raw capture only: no DLIO, pose conversion, viewer, or GPU computation. CPU, network and disk load remain.\n'
            if a.record_only else 'Live DLIO/pose estimation enabled: additional CPU load; use --record-only during latency-sensitive evaluation.\n')
          +'No motor commands from this recorder.',flush=True)
    try:
        processes.launch(plan,output,a.ros_workspace)
        recorder=children[-1][1];last_print=0.
        while recorder.poll() is None:
            for name,proc in children[:-1]:
                if proc.poll() is not None:raise RuntimeError(f'{name} exited; see {output/(name+".log")}')
            if time.monotonic()-last_print>=5:
                log=(output/'capture.log').read_text(errors='replace').splitlines()
                print(log[-1] if log else 'Waiting for sensor streams...',flush=True);last_print=time.monotonic()
            time.sleep(.2)
        if recorder.returncode:raise RuntimeError(f'Capture failed; see {output/"capture.log"}')
    except KeyboardInterrupt:
        print('Stopping sensor recording and saving partial capture...',flush=True)
    except Exception as exc:error=str(exc)
    finally:
        processes.close()
    incomplete=capture_saved(output)
    (output/'recording_result.json').write_text(json.dumps(dict(complete=incomplete is None and error is None,
        error=error or incomplete),indent=2)+'\n')
    if incomplete:
        print(f'INCOMPLETE recording at {output}. {incomplete}. See capture.log.',flush=True)
        if error is None:error=incomplete
    else:print(f'Saved and verified {output}',flush=True)
    if error:raise SystemExit(error)


if __name__=='__main__':main()
