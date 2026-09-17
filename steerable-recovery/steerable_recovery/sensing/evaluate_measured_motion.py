"""Replay the actual pelvis provider on DLIO output, with sim truth or live states.

Writes measured_pose.jsonl, motion.npz, evaluation.json and motion.png. No ROS,
robot connection, planner or motor publisher. Live pose accuracy needs an external
reference; IMU disagreement and stationary drift are not dynamic ground truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from motion_diffusion.behavioral.hardware.measured_pose import PelvisPoseEstimator
from motion_diffusion.behavioral.hardware.pelvis_odometry import G1Body


def nearest_indices(times, query):
    right = np.clip(np.searchsorted(times, query), 0, len(times)-1)
    left = np.maximum(right-1, 0)
    return np.where(np.abs(times[left]-query) < np.abs(times[right]-query), left, right)


def deployment_measurements(path, times, positions):
    """Use the actual runtime's episode start and saved heading, without replaying it.

    Provider pairing currently uses workstation state receipt time, so use the
    logged observation arrival time on that same clock. Capture-time accuracy is
    still separately unqualified; this is an offline diagnostic join.
    """
    import gzip
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as stream:
        controls=[r for line in stream if (r:=json.loads(line)).get('event')=='control']
    if not controls or controls[0]['tick']!=0:raise ValueError('Need telemetry containing the episode-start control tick')
    query=np.array([r['observation_arrival_s'] for r in controls])
    heading=np.array([float(np.asarray(r['causal_heading_rad']).reshape(-1)[0]) for r in controls])
    if not times[0]<=query[0]<=times[-1]:raise ValueError('Sensor recording does not cover deployment episode start')
    if np.any(np.diff(query)<=0):raise ValueError('Deployment observation times must increase')
    keep=(query>=times[0])&(query<=times[-1]);query=query[keep];heading=heading[keep]
    idx=nearest_indices(times,query)
    pair_error=np.abs(times[idx]-query)
    sampled=np.column_stack([np.interp(query,times,positions[:,k]) for k in range(3)])
    relative=(sampled-sampled[0]) @ Rotation.from_euler('z',-heading[0]).as_matrix().T
    yaw=np.arctan2(np.sin(heading-heading[0]),np.cos(heading-heading[0]))
    result=dict(observation_arrival_s=query,episode_time_s=query-query[0],
        position_reset_m=relative,height_rel_m=relative[:,2],yaw_rel_rad=yaw,
        odometry_pair_error_s=pair_error,paired=pair_error<=.02)
    # Full 3D rotation from the starting pelvis pose is distinct from steering's
    # fall-aware heading difference. Export both, with their conventions explicit.
    attitudes=Rotation.from_quat(np.asarray([r['pelvis_quat_xyzw'] for r in controls])[keep])
    result.update(orientation_relative_xyzw=(attitudes[0].inv()*attitudes).as_quat(),
        pelvis_quat_in_start_heading_xyzw=(Rotation.from_euler('z',-heading[0])*attitudes).as_quat(),
        initial_heading_rad=np.array(heading[0]))
    return result


def episode_trace(times, positions, quaternions, *, start_s=None):
    """Diagnostic display only: replay the shared heading at the control's 50 Hz.

    Interpolation here is OFFLINE, with both endpoints known. Live deployment keeps
    its own causal heading update and episode origins in AsyncNativeRecovery.
    """
    import torch
    from humanoidverse2.agents.action_vae.heading import replay_heading
    start = times[0] if start_s is None else start_s
    if not times[0] <= start < times[-1]:raise ValueError('Reset time is outside the recording')
    t = np.arange(start, times[-1]+1e-9, .02)
    q = Slerp(times, Rotation.from_quat(quaternions))(t).as_quat()
    heading = replay_heading(torch.from_numpy(q)).numpy()
    p = np.column_stack([np.interp(t, times, positions[:, k]) for k in range(3)])
    relative = (p-p[0]) @ Rotation.from_euler('z', -heading[0]).as_matrix().T
    yaw = np.arctan2(np.sin(heading-heading[0]), np.cos(heading-heading[0]))
    return t, relative, yaw


def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    d = np.load(args.odometry)
    stamps, unique = np.unique(d['stamp_s'], return_index=True)
    lp, lq = d['position'][unique], d['quat_xyzw'][unique]
    finite = np.isfinite(lp).all(1) & np.isfinite(lq).all(1) & (np.linalg.norm(lq,axis=1)>1e-6) & (stamps>0)
    stamps, lp, lq = stamps[finite], lp[finite], lq[finite]
    if len(stamps)<2:raise ValueError('No usable DLIO trajectory')
    source = np.load(args.truth if args.truth else args.capture/'sensors.npz')
    clock_note = 'Known simulation stamps; offline replay, not workstation arrival timing'
    if args.truth:
        st = source['t'].astype(float)
        qj, pq = source['dof_pos'][:,args.env], source['pelvis_quat_xyzw'][:,args.env]
        tq = source['torso_quat_xyzw'][:,args.env]
        query = stamps-args.epoch_s
        sample_times = st
    else:
        # Estimate offset only for pairing. Constant unknown transport delay remains
        # unqualified, even when the arrival jitter is tiny.
        offset = np.min(source['imu_receipt_monotonic_s']-source['imu_stamp_ns']*1e-9)
        query = stamps+offset
        st = source['state_receipt_monotonic_s']
        qj, pq = source['q'], source['pelvis_quat_wxyz'][:,[1,2,3,0]]
        tq = None
        sample_times = st
        clock_note = 'Minimum-arrival-offset pairing; absolute sensor timing is UNQUALIFIED'
    keep = (query >= st[0]) & (query <= st[-1])
    if args.stationary:keep &= query >= query[0]+5.
    stamps, lp, lq, query = stamps[keep], lp[keep], lq[keep], query[keep]
    if len(stamps)<2:raise ValueError('No overlapping sensor and odometry interval')
    indices = nearest_indices(st, query)
    mount = None if args.mount_calibration is None else json.loads(args.mount_calibration.read_text())['rotation_lidar_to_torso']
    estimator = PelvisPoseEstimator(G1Body(load_body_boxes=False), mount_rotation=mount)
    rows=[]; accepted_times=[]; pair_errors=[]
    for stamp, pos, quat, capture, idx in zip(stamps,lp,lq,query,indices):
        if abs(st[idx]-capture) > .02:continue
        state = dict(q=qj[idx],quat_xyzw=pq[idx])
        if tq is not None:state['torso_quat_xyzw']=tq[idx]
        elif 'torso_receipt_monotonic_s' in source:
            tt=source['torso_receipt_monotonic_s']; ti=int(nearest_indices(tt,np.array([capture]))[0])
            if abs(tt[ti]-capture)<=.02:state['torso_quat_xyzw']=source['torso_quat_wxyz'][ti,[1,2,3,0]]
        info=estimator.update(stamp,pos,quat,state)
        geometry_valid=bool(info['tilt_disagreement_rad']<=np.radians(5.))
        rows.append(dict(info,schema='hv2.pelvis_odometry.v1',frame='pelvis_imu_gravity_frame',
            estimator='dlio_mid360+g1_waist_fk',lidar_stamp_s=float(stamp),capture_monotonic_s=float(capture),
            clock_uncertainty_s=0. if args.truth else None, geometry_valid=geometry_valid,
            tracking_valid=False,offline_replay=True))
        accepted_times.append(capture);pair_errors.append(abs(st[idx]-capture))
    if len(rows)<2:raise ValueError('Too few timestamp-matched states')
    t=np.array(accepted_times);pos=np.array([r['position_m'] for r in rows])
    eq=np.array([r['pelvis_quat_xyzw'] for r in rows]); iq=np.array([r['imu_quat_xyzw'] for r in rows])
    # A single, user-selected episode origin. Never reinitialize heading at each point.
    joined=deployment_measurements(args.deployment_log,t,pos) if args.deployment_log else None
    start=t[0]+args.reset_after_s if joined is None else joined['observation_arrival_s'][0]
    ticks,relative,estimated_yaw=episode_trace(t,pos,eq,start_s=start)
    _,_,imu_yaw=episode_trace(t,pos,iq,start_s=start)
    if joined is not None:
        imu_yaw=np.interp(ticks,joined['observation_arrival_s'],np.unwrap(joined['yaw_rel_rad']))
        imu_yaw=np.arctan2(np.sin(imu_yaw),np.cos(imu_yaw))
    # Position reset axes must follow runtime's IMU heading, not estimated heading.
    from humanoidverse2.agents.action_vae.heading import initialize_heading
    import torch
    initial_q=Slerp(t,Rotation.from_quat(iq))([ticks[0]]).as_quat()
    h0=float(initialize_heading(torch.from_numpy(initial_q))[0])
    if joined is not None:h0=float(joined['initial_heading_rad'])
    sampled=np.column_stack([np.interp(ticks,t,pos[:,k]) for k in range(3)])
    relative=(sampled-sampled[0]) @ Rotation.from_euler('z',-h0).as_matrix().T
    arrays=dict(t=ticks-ticks[0],position_reset_m=relative,estimated_yaw_rel_rad=estimated_yaw,
                imu_yaw_rel_rad=imu_yaw)
    yaw_error=np.arctan2(np.sin(estimated_yaw-imu_yaw),np.cos(estimated_yaw-imu_yaw))
    report=dict(samples=len(rows),control_ticks=len(ticks),clock=clock_note,position_accuracy_reference='none',
        geometry_valid_fraction=float(np.mean([r['geometry_valid'] for r in rows])),
        timestamp_pair_error_max_ms=float(max(pair_errors)*1000),
        odometry_gap_max_ms=float(np.diff(t).max()*1000),
        yaw_disagreement_rms_deg=float(np.degrees(np.sqrt(np.mean(yaw_error**2)))),
        yaw_disagreement_max_deg=float(np.degrees(np.abs(yaw_error).max())),
        runtime_clock_qualified=False,offline_replay=True,
        scope='All finite matched estimates, including geometric failures; no filtering to improve accuracy numbers')
    if args.truth:
        truth=np.column_stack([np.interp(ticks,sample_times,source['pelvis_pos'][:,args.env,k]) for k in range(3)])
        truth=(truth-truth[0]) @ Rotation.from_euler('z',-h0).as_matrix().T
        arrays['truth_position_reset_m']=truth
        error=relative-truth
        report.update(position_accuracy_reference='simulator truth',
            position_rmse_m=float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),
            position_max_error_m=float(np.linalg.norm(error,axis=1).max()),
            height_rmse_m=float(np.sqrt(np.mean(error[:,2]**2))),height_max_error_m=float(np.abs(error[:,2]).max()))
    elif args.stationary:
        report.update(position_accuracy_reference='operator-declared stationary interval',
                      position_max_drift_m=float(np.linalg.norm(relative,axis=1).max()))
    (args.output/'measured_pose.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output/'motion.npz',**arrays)
    if args.deployment_log:
        np.savez_compressed(args.output/'deployment_measurements.npz',**joined)
        (args.output/'deployment_measurements.json').write_text(json.dumps(dict(
            position_reset_m='Rz(-heading_at_policy_start) @ (pelvis_position_t - pelvis_position_at_policy_start)',
            orientation_relative_xyzw='quaternion of inverse(R_pelvis_start) @ R_pelvis_t; xyzw, not Euler yaw',
            pelvis_quat_in_start_heading_xyzw='quaternion of Rz(-heading_at_policy_start) @ R_pelvis_t',
            yaw_rel_rad='wrapped change of actual runtime causal_forward_spine_heading_v1; not raw Euler yaw',
            interpolation='offline: both temporal endpoints available; not a live latency measurement',
            position_accuracy='unvalidated on dynamic hardware; compare to independent reference',
            timestamp_alignment=clock_note),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    axes[0,0].plot(relative[:,0],relative[:,1],label='estimated pelvis')
    if 'truth_position_reset_m' in arrays:
        truth=arrays['truth_position_reset_m'];axes[0,0].plot(truth[:,0],truth[:,1],label='truth')
    axes[0,0].set(xlabel='reset X (m)',ylabel='reset Y (m)',aspect='equal');axes[0,0].legend()
    for k,name in enumerate('XYZ'):axes[0,1].plot(arrays['t'],relative[:,k],label=name)
    axes[0,1].set(ylabel='displacement (m)',xlabel='episode time (s)');axes[0,1].legend()
    axes[1,0].plot(arrays['t'],np.degrees(imu_yaw),label='pelvis IMU shared heading')
    axes[1,0].plot(arrays['t'],np.degrees(estimated_yaw),label='DLIO-derived shared heading')
    axes[1,0].set(ylabel='heading change (deg)',xlabel='episode time (s)');axes[1,0].legend()
    axes[1,1].plot(t-ticks[0],np.degrees([r['tilt_disagreement_rad'] for r in rows]),label='tilt disagreement')
    axes[1,1].set(ylabel='degrees',xlabel='episode time (s)');axes[1,1].legend()
    fig.suptitle('Measured motion diagnostics — '+report['position_accuracy_reference']);fig.tight_layout()
    fig.savefig(args.output/'motion.png');plt.close(fig)
    print(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('odometry',type=Path);p.add_argument('--output',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--truth',type=Path);mode.add_argument('--capture',type=Path)
    p.add_argument('--env',type=int,default=0);p.add_argument('--epoch-s',type=float,default=1.7e9)
    p.add_argument('--stationary',action='store_true');p.add_argument('--reset-after-s',type=float,default=0.)
    p.add_argument('--mount-calibration',type=Path)
    p.add_argument('--deployment-log',type=Path,help='Actual telemetry.jsonl[.gz]: join measurements to its episode reset and saved heading')
    args=p.parse_args()
    evaluate(args)


if __name__=='__main__':main()
