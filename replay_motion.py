"""Review a stopped hardware run: video, sensor checks, XYZ and synchronized Viser.

Run after deployment stops. Never connects to the robot or starts a sensor driver.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from record_motion import DEFAULT_WS, camera_report, capture_saved


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def resolve_session(path):
    path=path.expanduser().resolve()
    # Accept the "Saved and verified raw sensors" path as well as "Session".
    if path.name=='sensors' and (path.parent/'telemetry.jsonl.gz').is_file():path=path.parent
    if not path.is_dir():raise ValueError(f'Session directory not found: {path}')
    return path


def finalize_camera(recording,review,*,export_video=True):
    camera=recording/'camera'
    result=dict(available=False,mp4=None,error=None)
    if not (camera/'frames.mjpg').is_file():return result
    result['available']=True
    config=read_json(recording/'session.json').get('camera') or {}
    saved=read_json(recording/'recording_result.json').get('camera') or {}
    if not (camera/'frames.npz').is_file():
        trigger=read_json(recording/'policy_start.json').get('monotonic_s')
        saved=camera_report(recording,config.get('fps',30),policy_start_monotonic_s=trigger)
        (review/'camera_finalize.json').write_text(json.dumps(saved,indent=2)+'\n')
    result['error']=saved.get('error') or saved.get('remux_error')
    if not (camera/'frames.npz').is_file():
        result.update(available=False,error=result['error'] or 'Camera timestamp index is missing')
        return result
    # MP4 is for sharing. Viser uses indexed MJPEG exposure times instead.
    video=review/'external.mp4';raw=camera/'frames.mjpg'
    if export_video and (not video.is_file() or video.stat().st_mtime_ns<raw.stat().st_mtime_ns):
        import numpy as np
        with np.load(camera/'frames.npz') as index:fps=float(index['fps'])
        pending=review/'external.pending.mp4'
        print('Preparing external.mp4 after recording stopped...',flush=True)
        with (review/'video_export.log').open('w') as log:
            try:
                subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y',
                    '-f','mjpeg','-framerate',str(fps),'-i',str(raw),'-an',
                    '-c:v','libx264','-threads','2','-preset','fast','-crf','20',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(pending)],
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
                pending.replace(video)
            except (OSError,subprocess.SubprocessError) as exc:
                result['error']=f'Video export failed: {exc}; see {review}/video_export.log'
    if video.is_file():result['mp4']=str(video)
    return result


def has_live_odometry(telemetry):
    import gzip
    with gzip.open(telemetry,'rt') as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('event')=='control' and row.get('pelvis_odometry') is not None:return True
    return False


def preserve_attempt(path):
    if path.exists():
        archived=path.with_name(path.name+f'.previous_{time.time_ns()}')
        path.rename(archived)
        print(f'Previous processing attempt preserved: {archived}',flush=True)


def prepare_xyz(session,raw,telemetry,calibration,*,capture_only=False):
    replay=session/'motion_replay';odom=replay/'dlio'
    pose=replay/('pose_calibrated' if calibration else 'pose_nominal')
    if not (raw/'sensors.npz').is_file() or not (raw/'bag/metadata.yaml').is_file():
        raise ValueError('Complete raw LiDAR/IMU capture is missing; XYZ cannot be reconstructed')
    replay.mkdir(exist_ok=True)
    report=read_json(odom/'summary.json')
    if not report or report.get('error') or not (odom/'odometry.npz').is_file():
        workspace=Path(os.environ.get('LIDAR_WS',DEFAULT_WS));setup=workspace/'install/setup.bash'
        if not setup.is_file():raise ValueError(f'Missing {setup}; set LIDAR_WS for offline XYZ')
        preserve_attempt(odom)
        print('Replaying saved LiDAR offline; no live sensors or motor connection.',flush=True)
        shell='set -e; source /opt/ros/humble/setup.bash; source "$1"; shift; export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=91; exec "$@"'
        subprocess.run(['bash','-c',shell,'replay-env',str(setup),'/usr/bin/python3',
            str(ROOT/'scripts/fall_recovery/validation/replay_dlio.py'),str(raw/'bag'),
            '--config',str(ROOT/'g1_deploy/config/lidar/dlio.yaml'),'--output',str(odom),
            '--imu-unit','g','--scan-packets','3','--timeout-s','360'],check=True)
        report=read_json(odom/'summary.json')
    if report.get('error'):raise ValueError(f'Offline DLIO failed: {report["error"]}')
    calibration=None if calibration is None else str(calibration.resolve())
    signature=dict(telemetry=None if capture_only else str(telemetry),mount_calibration=calibration,
        calibration_contents=None if calibration is None else Path(calibration).read_text(),
        odometry_mtime_ns=(odom/'odometry.npz').stat().st_mtime_ns)
    if not (pose/'measured_pose.jsonl').is_file() or read_json(pose/'source.json')!=signature:
        preserve_attempt(pose)
        command=[sys.executable,str(ROOT/'scripts/fall_recovery/validation/evaluate_measured_motion.py'),
            str(odom/'odometry.npz'),'--capture',str(raw),'--output',str(pose)]
        if calibration:command+=['--mount-calibration',calibration]
        with (replay/'pose_evaluation.log').open('w') as log:
            run=subprocess.run(command+([] if capture_only else ['--deployment-log',str(telemetry)]),
                               stdout=log,stderr=subprocess.STDOUT)
        if run.returncode:
            # Keep late estimates for display, never invent an A-relative origin.
            if 'Sensor recording does not cover deployment episode start' not in (replay/'pose_evaluation.log').read_text():
                raise RuntimeError(f'Pelvis evaluation failed; see {replay}/pose_evaluation.log')
            print('XYZ at policy start is unavailable. Preparing later motion with an explicitly late origin.',flush=True)
            preserve_attempt(pose)
            with (replay/'pose_evaluation_late.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        (pose/'source.json').write_text(json.dumps(signature,indent=2)+'\n')
    return pose/'measured_pose.jsonl'


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
        control_ticks=len(controls),commands_issued=len(issued),
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
    p.add_argument('session',type=Path,help='Session directory or its sensors/ directory')
    p.add_argument('--port',type=int,default=8081)
    p.add_argument('--pose-only',action='store_true',help='Joints, orientation and video; skip XYZ processing')
    p.add_argument('--live-estimates',action='store_true',help='Use XYZ actually logged during deployment; no offline DLIO')
    p.add_argument('--offline-xyz',action='store_true',help='Explicitly reconstruct XYZ from raw LiDAR; default reuses live estimates or cached offline motion')
    p.add_argument('--export-video',action='store_true',help='Also encode a shareable MP4; Viser reads the original camera frames directly')
    p.add_argument('--full-validation',action='store_true',help='Decode all video frames; default validates index and samples three frames')
    p.add_argument('--prepare-only',action='store_true',help='Save checks/requested exports without starting Viser')
    p.add_argument('--capture-only',action='store_true',help='Standalone raw capture without policy telemetry (also autodetected)')
    p.add_argument('--mount-calibration',type=Path,help='Optional explicit fixed-mount calibration candidate')
    a=p.parse_args();session=resolve_session(a.session)
    telemetry=session/'telemetry.jsonl.gz'
    if not telemetry.is_file() and (session/'raw/sensors.npz').is_file():a.capture_only=True
    if a.live_estimates and (a.capture_only or a.pose_only or a.mount_calibration or a.offline_xyz):
        p.error('--live-estimates requires a policy session and no offline pose options')
    if not a.capture_only and not telemetry.is_file():p.error(f'Missing {telemetry}')
    recording=session if a.capture_only else session/'sensors'
    depth_only=(recording/'depth/summary.json').is_file() and not (recording/'raw').exists()
    if recording.is_dir() and not (recording/'recording_result.json').is_file() and not depth_only:
        p.error('Recorder has not finalized this run. Wait for "Saved and verified raw sensors" before reviewing.')
    review=session/'review';review.mkdir(exist_ok=True)
    result=dict(session=str(session),recording=read_json(recording/'recording_result.json'),
        model=read_json(session/'session.json').get('model'),
        policy_end={name:read_json(session/name) for name in ('FAILED.json','STOPPED.json','COMPLETE.json')
                    if (session/name).is_file()},
        capture_error=capture_saved(recording) if recording.is_dir() else 'No raw recording',
        xyz_source='none',xyz_error=None)
    result['camera']=finalize_camera(recording,review,export_video=a.export_video)
    if not a.capture_only:
        result['timing']=save_timing_report(telemetry,session/'motion_replay/deployment_timing.json')
    viewer=[sys.executable,str(ROOT/'g1_deploy/sonic/scripts/replay_episode.py'),
        *(['--capture',str(session/'raw')] if a.capture_only else ['--telemetry',str(telemetry)]),
        '--urdf',str(ROOT/'humanoidverse2/data/robots/g1/g1_29dof_sonic.urdf'),'--port',str(a.port)]
    if result['camera']['available']:viewer+=['--external-camera',str(recording/'camera')]
    if (recording/'depth/frames.jsonl').is_file():viewer+=['--depth-recording',str(recording/'depth')]
    use_live=a.live_estimates or (not a.pose_only and not a.offline_xyz and not a.mount_calibration
                                  and not a.capture_only and has_live_odometry(telemetry))
    if use_live:
        from sonic.common.episode import load_native_recovery
        try:
            load_native_recovery(telemetry,live_estimates=True)
            viewer+=['--live-estimates'];result['xyz_source']='recorded_live'
        except ValueError as exc:
            result['xyz_error']=str(exc)
            print(f'Live XYZ unavailable: {exc}. Showing joints, orientation and video.',flush=True)
    if not a.pose_only and not use_live:
        try:
            calibration=a.mount_calibration or (session/'odometry_mount.json' if (session/'odometry_mount.json').is_file() else None)
            selection=read_json(review/'xyz_selection.json')
            pose=(session/selection['pose'] if selection.get('pose') and not a.mount_calibration else
                  session/'motion_replay'/('pose_calibrated' if calibration else 'pose_nominal')/'measured_pose.jsonl')
            if a.offline_xyz or a.mount_calibration:
                pose=prepare_xyz(session,recording/'raw',telemetry,calibration,capture_only=a.capture_only)
            elif not pose.is_file():
                raise ValueError('No live XYZ or cached offline estimate. Use --offline-xyz to reconstruct; record with --live-odometry next time')
            from sonic.common.episode import load_native_recovery, load_raw_motion, attach_measured_translation
            state,_,meta=(load_raw_motion(recording/'raw') if a.capture_only else load_native_recovery(telemetry))
            attach_measured_translation(state,meta,pose,allow_late_origin=True)
            result['xyz_evaluation']=read_json(pose.parent/'evaluation.json')
            viewer+=['--measured-pose',str(pose),'--allow-late-origin'];result['xyz_source']='offline_dlio'
            (review/'xyz_selection.json').write_text(json.dumps(dict(pose=str(pose.relative_to(session))),indent=2)+'\n')
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
            result['xyz_error']=str(exc)
            print(f'XYZ unavailable: {exc}. Continuing with recorded joints, orientation and video.',flush=True)
    validation=session/'motion_replay/replay_validation.json'
    subprocess.run([*viewer,'--validate-only',*(['--full-validation'] if a.full_validation else []),
                    '--validation-output',str(validation)],check=True)
    result['replay']=read_json(validation)
    (review/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    ending=next(iter(result['policy_end']),'no policy end marker')
    outcome=result['policy_end'].get(ending,{})
    detail=outcome.get('error') or outcome.get('reason') or ''
    lines=['# Run review','',f'Session: `{session}`','',
        '- [External video](external.mp4)' if result['camera']['mp4'] else
        ('- External video available directly in Viser; use --export-video for MP4.' if result['camera']['available'] else '- External video not recorded.'),
        '- [Checks and timing](review.json)',
        f'- Model: {result["model"] or "not recorded"}.',
        f'- Policy outcome: {ending} {detail}.',
        f'- XYZ source: {result["xyz_source"]}. Estimated motion, not ground truth.',
        f'- Position origin delay: {result["replay"].get("position_origin_delay_s")} s (0 means policy start).',
        f'- XYZ coverage: {100*result["replay"]["translation_coverage"]:.1f}%.',
        f'- Recording error: {result["capture_error"] or result["recording"].get("error") or "none reported"}.',
        f'- XYZ error: {result["xyz_error"] or "none reported"}.',
        f'- Camera error: {result["camera"]["error"] or "none reported"}.','',
        f'- Depth error: {result["replay"].get("depth_error") or "none reported"}.',
        'The MP4 includes pre/post-policy footage on a uniform frame timeline. Viser synchronizes indexed exposures.',
        ('Depth: metric frames recorded.' if result['replay'].get('depth') else 'Depth: not recorded in this run.'),
        'Camera/odometry clocks and dynamic XYZ accuracy are not independently qualified.','']
    (review/'README.md').write_text('\n'.join(lines))
    print(f'Review ready: {review}/README.md',flush=True)
    if result['camera']['mp4']:print(f'Video: {result["camera"]["mp4"]}',flush=True)
    if result['capture_error']:print(f'Recording warning: {result["capture_error"]}',flush=True)
    if result['camera']['error']:print(f'Camera warning: {result["camera"]["error"]}',flush=True)
    if result['replay'].get('depth_error'):print(f'Depth warning: {result["replay"]["depth_error"]}',flush=True)
    if not a.prepare_only:os.execv(sys.executable,viewer)


if __name__=='__main__':main()
