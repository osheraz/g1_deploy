"""Record recovery sensors and measured pelvis motion alongside normal deployment.

Read-only robot subscriptions. Owns only its sensor/estimator child processes.
The planner keeps depth. Optional live odometry publishes to the runtime receiver.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_WS='/tmp/claude-114471805/-home-osher-hv2/3ab304f2-651f-4602-b50a-69406401ca49/scratchpad/eth_stack/ros2_ws'


class RecorderProcesses:
    """Shared process ownership for manual recording and deployment-triggered capture."""
    def __init__(self):
        self.children=[];self.logs=[]

    def launch(self,plan,output,workspace,*,controlled=False):
        env_script='set -e; source /opt/ros/humble/setup.bash; source "$1"; shift; export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; export PYTHONPATH=/home/osher/unitree_sdk2_python:${PYTHONPATH:-}; exec "$@"'
        for name,command in plan:
            log=open(output/(name+'.log'),'w');self.logs.append(log)
            proc=subprocess.Popen(['nice','-n','10','bash','-c',env_script,'motion-env',str(workspace/'install/setup.bash'),*command],
                stdin=subprocess.PIPE if controlled and name=='capture' else subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            self.children.append((name,proc))

    def check(self):
        for name,proc in self.children:
            if proc.poll() is not None:raise RuntimeError(f'Sensor {name} exited ({proc.returncode}); inspect its recording log')

    def close(self):
        # The policy has already closed before deployment calls this. A second
        # Ctrl+C must not interrupt the recorder's final compressed write.
        previous=signal.signal(signal.SIGINT,signal.SIG_IGN)
        try:self._close()
        finally:signal.signal(signal.SIGINT,previous)

    def _close(self):
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
    def __init__(self,output,*,interface,seconds,existing_driver=False,label='supported_hoist',live_odometry=None,automatic_start=False):
        self.output=Path(output);self.started=False;self.processes=RecorderProcesses()
        self.error=None;self.monitor_stop=threading.Event();self.monitor=None
        self.args=SimpleNamespace(interface=interface,seconds=seconds,label=label,
            existing_driver=existing_driver,record_only=True,mount_calibration=None,
            driver_config=ROOT/'g1_deploy/config/lidar/MID360_bluefin.json')
        self.live_odometry=live_odometry
        self.automatic_start=automatic_start
        if live_odometry is not None:
            self.args.record_only=False;self.args.scan_packets=3
            self.args.dlio_config=ROOT/'g1_deploy/config/lidar/dlio.yaml'
            self.args.mount_calibration=Path(live_odometry['mount_calibration'])
            self.args.odometry_port=int(live_odometry['port'])
            self.args.clock_calibration=live_odometry.get('clock_calibration')
        self.workspace=Path(os.environ.get('LIDAR_WS',DEFAULT_WS))

    def prepare(self):
        """Launch only. Sensor discovery must not delay policy loading or startup."""
        try:self._launch()
        except Exception as exc:
            self.error=str(exc)
            print(f'Recorder unavailable: {exc}. Policy startup continues.',flush=True)

    def _launch(self):
        if not (self.workspace/'install/setup.bash').is_file():
            raise RuntimeError(f'Missing ROS workspace {self.workspace}; set LIDAR_WS or use --no-record-sensors')
        self.output.mkdir(parents=True,exist_ok=False)
        plan=commands(self.args,self.output);plan[-1][1].append('--controlled')
        (self.output/'session.json').write_text(json.dumps(dict(commands=plan,label=self.args.label,
            motor_publisher=False,trigger='sensor-only inference start' if self.automatic_start else 'accepted policy A',planner_terrain='depth',
            policy_session=str(self.output.parent),raw_only=self.args.record_only,
            live_odometry=self.live_odometry),indent=2)+'\n')
        print('Starting recorder in background; loading policy without waiting for it.',flush=True)
        self.processes.launch(plan,self.output,self.workspace,controlled=True)
        os.set_blocking(self.processes.children[-1][1].stdin.fileno(),False)
        self.monitor=threading.Thread(target=self._monitor,daemon=True);self.monitor.start()
        print(('Recording starts with sensor-only inference.' if self.automatic_start else 'A starts policy + recording.')+
            f' Raw sensors: {self.output}',flush=True)

    def _monitor(self):
        previous=None
        while not self.monitor_stop.wait(.5):
            try:
                self.processes.check()
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
            message=(json.dumps(dict(command='start',monotonic_s=time.monotonic()))+'\n').encode()
            os.write(proc.stdin.fileno(),message)
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
        error=self.error or (capture_saved(self.output) if self.started else None)
        if self.started and error is None:
            capture=self.processes.children[-1][1]
            if capture.returncode:error=f'Capture exited with status {capture.returncode}; inspect capture.log'
        (self.output/'recording_result.json').write_text(json.dumps(dict(
            started=self.started,complete=self.started and error is None,error=error),indent=2)+'\n')
        if error:print(f'INCOMPLETE sensor recording: {error}. See {self.output}/capture.log',flush=True)
        elif self.started:print(f'Saved and verified raw sensors: {self.output}',flush=True)
        else:print('Recorder closed: policy A was not pressed; no motion capture started.',flush=True)


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
    p.add_argument('--dry-run',action='store_true',help='Print child commands without starting processes or contacting hardware')
    a=p.parse_args()
    if not 0<a.seconds<=300 or not 1<=a.scan_packets<=10:p.error('Seconds must be 0..300; scan-packets 1..10')
    output=(a.output or ROOT/'logs/measured_motion'/time.strftime('%Y%m%d_%H%M%S')).resolve()
    plan=commands(a,output)
    if a.dry_run:
        print(json.dumps(dict(output=str(output),commands=plan,motor_publisher=False),indent=2));return
    required=[a.ros_workspace/'install/setup.bash']
    if not a.existing_driver:required.append(a.driver_config)
    if not a.record_only:required.append(a.dlio_config)
    for path in required:
        if not path.is_file():p.error(f'Missing {path}; set --ros-workspace / LIDAR_WS if the ROS workspace moved')
    output.mkdir(parents=True,exist_ok=False)
    (output/'session.json').write_text(json.dumps(dict(label=a.label,commands=plan,
        start_monotonic_s=time.monotonic(),motor_publisher=False,head_mount='fixed',
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
