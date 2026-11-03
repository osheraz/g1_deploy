"""Evaluate the estimates actually available to deployment, or replay that path without a robot."""
import argparse
import gzip
import json
import os
from pathlib import Path
import signal
import re
import subprocess
import sys
import time

import numpy as np

from steerable_recovery.paths import ROOT, CONFIG_ROOT, PACKAGE_ROOT
sys.path.insert(0,str(ROOT))
from steerable_recovery.hardware.runtime import PelvisOdometryReceiver


def stats(values,scale=1.):
    values=np.asarray([v for v in values if v is not None and np.isfinite(v)],float)*scale
    if not len(values):return dict(samples=0)
    return dict(samples=len(values),p50=float(np.median(values)),p95=float(np.percentile(values,95)),max=float(values.max()))


def summarize(rows):
    controls=[r for r in rows if r.get('event')=='control']
    poses=[r['pelvis_odometry'] for r in controls if r.get('pelvis_odometry')]
    if not poses:raise ValueError('No live odometry in this log; use --replay for an older raw recording')
    reasons={}
    for p in poses:
        reason=p.get('invalid_reason') or 'qualified'
        reasons[reason]=reasons.get(reason,0)+1
    positions=[p['position_reset_m'] for p in poses if p.get('position_reset_m') is not None]
    return dict(control_ticks=len(controls),odometry_ticks=len(poses),
        available_fraction=float(np.mean([p['available'] for p in poses])),
        qualified_fraction=float(np.mean([p['tracking_valid'] for p in poses])),invalid_reasons=reasons,
        estimated_pose_age_ms=stats([p.get('age_s') for p in poses],1000),
        receiver_cost_ms=stats([p.get('poll_s') for p in poses],1000),
        bridge_geometry_cost_ms=stats([p.get('processing_s') for p in poses],1000),
        state_pair_error_ms=stats([p.get('state_pair_error_s') for p in poses],1000),
        heading_disagreement_deg=stats([abs(p['yaw_disagreement_rad']) if p.get('yaw_disagreement_rad') is not None else None for p in poses],180/np.pi),
        tilt_disagreement_deg=stats([p.get('tilt_disagreement_rad') for p in poses],180/np.pi),
        full_orientation_disagreement_deg=stats([p.get('orientation_disagreement_rad') for p in poses],180/np.pi),
        control_step_ms=stats([r.get('control_wall_s') for r in controls],1000),
        planner_ms=stats([r.get('sample_wall_s') for r in rows if r.get('event')=='plan'],1000),
        observation_to_command_ms=stats([r['command_publish_s']-r['observation_capture_s']
            for r in controls if r.get('command_publish_s') is not None and r.get('observation_capture_s') is not None],1000),
        max_relative_displacement_m=None if not positions else float(np.linalg.norm(positions,axis=1).max()),
        translation_accuracy='Not measured: requires independent position reference',
        timing_qualification='Receipt-derived ages do not certify device clock offsets')


def replay(session,output,workspace):
    raw=session/'sensors/raw'
    if not (raw/'sensors.npz').is_file():raise ValueError(f'Missing {raw}/sensors.npz')
    receiver=PelvisOdometryReceiver(0,background=True)
    port=receiver.socket.getsockname()[1]
    command=['/usr/bin/python3',str(PACKAGE_ROOT / 'tools/sensors/replay_dlio.py'),
        str(raw/'bag'),'--config',str(CONFIG_ROOT / 'lidar/dlio.yaml'),
        '--output',str(output/'dlio'),'--imu-unit','g','--scan-packets','3','--timeout-s','360',
        '--hardware-capture',str(raw),'--mount-calibration',
        str(CONFIG_ROOT / 'lidar/mount_candidate.json'),'--odometry-port',str(port)]
    setup='set -e; source /opt/ros/humble/setup.bash; source "$1"; shift; export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=91 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; exec "$@"'
    rows=[];proc=None;cpu_start=time.process_time();wall_start=time.monotonic()
    try:
        with open(output/'replay.log','w') as log:
            proc=subprocess.Popen(['bash','-c',setup,'replay-env',str(workspace/'install/setup.bash'),*command],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            next_tick=time.monotonic();last_print=next_tick
            while proc.poll() is None:
                now=time.monotonic();started=time.perf_counter()
                pose=receiver.sample(now)
                pose['poll_s']=time.perf_counter()-started
                rows.append(dict(event='control',observation_arrival_s=now,pelvis_odometry=pose))
                if now-last_print>=5:
                    print(f'Replay: {len(rows)} ticks; XYZ available={pose["available"]}; {pose["invalid_reason"]}',flush=True)
                    last_print=now
                next_tick=max(next_tick+.02,time.monotonic())
                time.sleep(max(0,next_tick-time.monotonic()))
            if proc.returncode:raise RuntimeError(f'Replay failed; see {output}/replay.log')
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGINT)
            try:proc.wait(timeout=8)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        receiver.close()
        (output/'telemetry.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    context=dict(receiver_cpu_seconds=time.process_time()-cpu_start,wall_seconds=time.monotonic()-wall_start,
        scope='1x ROS bag + live bridge + production receiver; no planner, camera, DDS or motors; clocks remain unqualified')
    replay_report=json.loads((output/'dlio/summary.json').read_text())
    shift=replay_report.get('recording_to_replay_monotonic_shift_s')
    if shift is not None:
        with gzip.open(session/'telemetry.jsonl.gz','rt') as f:
            original=[json.loads(line) for line in f]
        times=[r['observation_arrival_s'] for r in original if r.get('event')=='control']
        active=[r for r in rows if times[0]+shift<=r['observation_arrival_s']<=times[-1]+shift]
        if active:context['recorded_policy_interval']=summarize(active)
    costs=[float(v) for v in re.findall(r'Computation Time\s*::\s*([\d.]+) ms',
        (output/'dlio/dlio.log').read_text(errors='replace'))]
    context['dlio_reported_registration_ms']=stats(costs)
    return rows,context


def plot(rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[r for r in rows if r.get('event')=='control' and r.get('pelvis_odometry')]
    t=np.array([r['observation_arrival_s'] for r in rows]);t-=t[0]
    poses=[r['pelvis_odometry'] for r in rows]
    # Real telemetry stores A-relative position. Replay has no policy A: use
    # absolute estimate and label it accordingly, without inventing an origin.
    relative=any(p.get('position_reset_m') is not None for p in poses)
    key='position_reset_m' if relative else 'position_m'
    xyz=np.array([p.get(key) if p.get(key) is not None and p['available'] else [np.nan]*3 for p in poses])
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for i,label in enumerate('XYZ'):axes[0].plot(t,xyz[:,i],label=label)
    axes[0].set_ylabel('A-relative m' if relative else 'Estimator position m');axes[0].legend()
    axes[1].plot(t,[np.nan if p.get('age_s') is None else p['age_s']*1000 for p in poses])
    axes[1].axhline(50,color='r',ls='--');axes[1].set_ylabel('Estimated age ms')
    for key,label in [('yaw_disagreement_rad','heading'),('tilt_disagreement_rad','tilt')]:
        axes[2].plot(t,[np.nan if p.get(key) is None else np.degrees(p[key]) for p in poses],label=label)
    axes[2].set_ylabel('Disagreement deg');axes[2].set_xlabel('Seconds');axes[2].legend()
    fig.suptitle('Live measurements; gaps are unavailable estimates; XYZ accuracy is unverified')
    fig.tight_layout();fig.savefig(path);plt.close(fig)


def main():
    from steerable_recovery.recording.session import DEFAULT_WS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('session',type=Path);p.add_argument('--replay',action='store_true')
    p.add_argument('--output',type=Path)
    p.add_argument('--baseline',type=Path,help='Compare timing with another real deployment session; not a causal benchmark')
    p.add_argument('--ros-workspace',type=Path,default=os.environ.get('LIDAR_WS',DEFAULT_WS))
    a=p.parse_args();output=a.output or a.session/('live_replay_'+time.strftime('%Y%m%d_%H%M%S') if a.replay else 'live_evaluation')
    output.mkdir(parents=True,exist_ok=not a.replay)
    context={}
    if a.replay:rows,context=replay(a.session,output,a.ros_workspace)
    else:
        path=a.session/'telemetry.jsonl.gz'
        if not path.exists():path=a.session/'telemetry.jsonl'
        with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:rows=[json.loads(line) for line in f]
    report=summarize(rows);report.update(context)
    if a.baseline:
        from steerable_recovery.review.session import save_timing_report
        report['baseline_timing']=save_timing_report(a.baseline/'telemetry.jsonl.gz',output/'baseline_timing.json')
        report['baseline_caveat']='Separate run: pose, scheduling and model outputs may differ; compare distributions, not a proven incremental latency'
    (output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n');plot(rows,output/'motion.png')
    print(json.dumps(report,indent=2));print(f'Plot and report: {output}')


if __name__=='__main__':main()
