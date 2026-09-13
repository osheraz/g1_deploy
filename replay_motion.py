"""Prepare relative pelvis motion offline and launch the existing Viser player.

Run after deployment stops. Never connects to the robot or starts a sensor driver.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from record_motion import DEFAULT_WS


def save_timing_report(telemetry,output):
    """Named measured endpoints; no assumption that asynchronous stages add up."""
    import gzip
    import numpy as np
    with gzip.open(telemetry,'rt') as stream:rows=[json.loads(line) for line in stream]
    controls=[r for r in rows if r.get('event')=='control']
    issued=[r for r in controls if r.get('command_publish_s') is not None]
    def stats(seconds):
        x=np.asarray(seconds,dtype=float)*1000
        return dict(samples=len(x),p50_ms=None if not len(x) else float(np.median(x)),
            p95_ms=None if not len(x) else float(np.percentile(x,95)),max_ms=None if not len(x) else float(x.max()))
    result=dict(control_step=stats([r['control_wall_s'] for r in controls]),
        planner_pass=stats([r['sample_wall_s'] for r in rows if r.get('event')=='plan']),
        robot_observation_to_command=stats([r['command_publish_s']-r['observation_capture_s'] for r in issued]),
        latest_depth_age_at_command=stats([r['command_publish_s']-r['depth_timing']['capture_s'] for r in issued if r.get('depth_timing')]),
        note='Milliseconds. Depth age describes the latest logged frame, not a traced planner-to-command dependency. '
             'Offline DLIO replay does not establish live odometry latency.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print('Deployment timing (p95): '+', '.join(f'{name} {v["p95_ms"]:.2f} ms'
        for name,v in result.items() if isinstance(v,dict) and v['p95_ms'] is not None),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('session',type=Path,help='The Session directory printed by deploy_diffusion.py')
    p.add_argument('--port',type=int,default=8081)
    p.add_argument('--pose-only',action='store_true',help='Only joints and IMU orientation; no estimated translation')
    p.add_argument('--live-estimates',action='store_true',help='Use the XYZ actually logged during deployment; do not rerun DLIO')
    p.add_argument('--prepare-only',action='store_true',help='Compute offline motion without starting Viser')
    p.add_argument('--capture-only',action='store_true',help='Session is a standalone sensor recording, with raw/ and no policy telemetry')
    p.add_argument('--mount-calibration',type=Path,help='Optional explicit fixed-mount calibration candidate')
    a=p.parse_args();session=a.session.expanduser().resolve()
    if a.live_estimates and (a.capture_only or a.pose_only or a.mount_calibration):p.error('--live-estimates requires a policy session and no offline pose options')
    telemetry=session/'telemetry.jsonl.gz'
    if not a.capture_only and not telemetry.is_file():p.error(f'Missing {telemetry}')
    if not a.capture_only:save_timing_report(telemetry,session/'motion_replay/deployment_timing.json')
    viewer=[sys.executable,str(ROOT/'g1_deploy/sonic/scripts/replay_episode.py'),
        *(['--capture',str(session/'raw')] if a.capture_only else ['--telemetry',str(telemetry)]),
        '--urdf',str(ROOT/'humanoidverse2/data/robots/g1/g1_29dof_sonic.urdf'),
        '--port',str(a.port)]
    if a.live_estimates:viewer+=['--live-estimates']
    if not a.pose_only and not a.live_estimates:
        raw=session/('raw' if a.capture_only else 'sensors/raw');replay=session/'motion_replay';odom=replay/'dlio'
        pose=replay/('pose_calibrated' if a.mount_calibration else 'pose_nominal')
        if not (raw/'sensors.npz').is_file() or not (raw/'bag/metadata.yaml').is_file():
            p.error('This session lacks the complete raw sensor capture needed for XYZ. '
                    'Use --pose-only for the old joint/IMU replay. New A-triggered sessions save both together.')
        replay.mkdir(exist_ok=True)
        if not (odom/'summary.json').is_file():
            if odom.exists():p.error(f'Incomplete offline replay; move {odom} aside before retrying')
            workspace=Path(os.environ.get('LIDAR_WS',DEFAULT_WS))
            setup=workspace/'install/setup.bash'
            if not setup.is_file():p.error(f'Missing {setup}; set LIDAR_WS')
            print('Replaying saved LiDAR offline; no live sensors or motor connection.',flush=True)
            shell='set -e; source /opt/ros/humble/setup.bash; source "$1"; shift; export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=91; exec "$@"'
            subprocess.run(['bash','-c',shell,'replay-env',str(setup),'/usr/bin/python3',
                str(ROOT/'scripts/fall_recovery/validation/replay_dlio.py'),str(raw/'bag'),
                '--config',str(ROOT/'g1_deploy/config/lidar/dlio.yaml'),'--output',str(odom),
                '--imu-unit','g','--scan-packets','3','--timeout-s','360'],check=True)
        report=json.loads((odom/'summary.json').read_text())
        if report.get('error'):p.error(f'Offline DLIO failed: {report["error"]}; inspect {odom}')
        calibration=None if a.mount_calibration is None else str(a.mount_calibration.resolve())
        signature=dict(telemetry=str(telemetry),mount_calibration=calibration,
            calibration_contents=None if calibration is None else Path(calibration).read_text())
        if pose.exists() and not (pose/'source.json').is_file():
            p.error(f'Incomplete pose evaluation; move {pose} aside before retrying')
        if (pose/'source.json').is_file() and json.loads((pose/'source.json').read_text())!=signature:
            p.error(f'Calibration changed; move {pose} aside before regenerating')
        if not (pose/'measured_pose.jsonl').is_file():
            command=[sys.executable,str(ROOT/'scripts/fall_recovery/validation/evaluate_measured_motion.py'),
                str(odom/'odometry.npz'),'--capture',str(raw),'--output',str(pose)]
            if not a.capture_only:command+=['--deployment-log',str(telemetry)]
            if calibration:command+=['--mount-calibration',calibration]
            subprocess.run(command,check=True)
            (pose/'source.json').write_text(json.dumps(signature,indent=2)+'\n')
        viewer+=['--measured-pose',str(pose/'measured_pose.jsonl')]
        print(f'Offline motion: {pose}\nTranslation is a LiDAR estimate, not ground truth.',flush=True)
    subprocess.run([*viewer,'--validate-only','--validation-output',str(session/'motion_replay/replay_validation.json')],check=True)
    if not a.prepare_only:os.execv(sys.executable,viewer)


if __name__=='__main__':main()
