"""Record real Livox ROS topics and G1 LowState without any motor publisher.

Run with system Python after sourcing ROS 2 and the Livox workspace. ROS_DOMAIN_ID
must be nonzero; Unitree LowState is subscribed separately on DDS domain 0.
Raw message capture times and local receipt times are retained independently.
"""
from __future__ import annotations

if __name__ == '__main__' and not __package__:
    import _bootstrap  # System Python workers use this checkout, not the policy virtualenv.
from steerable_recovery.paths import ROOT

import argparse
import json
import os
import signal
from pathlib import Path
import sys
import threading
import time
from collections import deque

import numpy as np

# The dedicated capture process owns these DDS/ROS handles. Keep them alive until
# process exit: deleting a Unitree listener while a DDS callback is active can
# deadlock inside the SDK before Python gets a chance to save the recording.
_PROCESS_HANDLES=[]


def cloud_array(msg):
    """Decode PointCloud2 including byte order, organized rows and row padding."""
    types={1:'i1',2:'u1',3:'i2',4:'u2',5:'i4',6:'u4',7:'f4',8:'f8'}
    endian='>' if msg.is_bigendian else '<'
    dtype=np.dtype(dict(names=[f.name for f in msg.fields],
        formats=[np.dtype(endian+types[f.datatype]) if f.count==1 else
                 (np.dtype(endian+types[f.datatype]),(f.count,)) for f in msg.fields],
        offsets=[f.offset for f in msg.fields],itemsize=msg.point_step))
    return np.ndarray((msg.height,msg.width),dtype=dtype,buffer=bytes(msg.data),
        strides=(msg.row_step,msg.point_step)).reshape(-1)


def rate_report(times):
    t=np.asarray(times,dtype=np.float64)
    if len(t)<2:return dict(samples=len(t),hz=None)
    dt=np.diff(t)
    return dict(samples=len(t),hz=float((len(t)-1)/(t[-1]-t[0])) if t[-1]>t[0] else None,
        interval_p50_ms=float(np.median(dt)*1000),interval_p99_ms=float(np.quantile(dt,.99)*1000),
        interval_max_ms=float(dt.max()*1000),nonincreasing=int((dt<=0).sum()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface',default='enp132s0')
    parser.add_argument('--seconds',type=float,default=20.)
    parser.add_argument('--pose',default='unknown')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--controlled',action='store_true',help='Arm streams; START on stdin begins capture, EOF/signal stops it')
    args=parser.parse_args()
    if not np.isfinite(args.seconds) or not 0<args.seconds<=300:
        parser.error('--seconds must be positive and at most 300')
    if int(os.environ.get('ROS_DOMAIN_ID','0'))==0:
        parser.error('Set a nonzero ROS_DOMAIN_ID for Livox; Unitree uses domain 0')
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.serialization import serialize_message
    import rosbag2_py
    from sensor_msgs.msg import Imu,PointCloud2
    from nav_msgs.msg import Odometry
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize,ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_,IMUState_
    from steerable_recovery.hardware.fast_crc import CRC

    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    metadata=dict(schema='hv2.lidar_live_capture.v1',pose=args.pose,interface=args.interface,
        requested_seconds=args.seconds,motor_publisher=False,ros_domain=int(os.environ['ROS_DOMAIN_ID']),
        clocks='capture stamps are unaligned sensor clocks; receipt_monotonic_s is workstation time',
        imu_acceleration_units='driver values, nominal Livox g; preserved without scaling',
        motor_torque_source='LowState motor_state.tau_est, SDK estimate; not a contact-force measurement',
        contacts='not estimated; raw signals retained for offline estimation and validation')
    (out/'capture.json').write_text(json.dumps(metadata,indent=2)+'\n')
    bag=rosbag2_py.SequentialWriter()
    bag.open(rosbag2_py.StorageOptions(uri=str(out/'bag'),storage_id='mcap'),
             rosbag2_py.ConverterOptions('cdr','cdr'))
    for topic,kind in [('/livox/lidar','sensor_msgs/msg/PointCloud2'),('/livox/imu','sensor_msgs/msg/Imu'),('/dlio/odom_node/odom','nav_msgs/msg/Odometry')]:
        bag.create_topic(rosbag2_py.TopicMetadata(name=topic,type=kind,serialization_format='cdr'))
    clouds=deque();imus=deque();states=deque();torsos=deque();odometry=[];lock=threading.Lock();bad_crc=0
    frozen=threading.Event()
    recording=threading.Event();stop_requested=threading.Event()
    seen={};trigger={}
    if not args.controlled:recording.set()
    pending_bag=deque()
    def accept(name):
        seen[name]=time.monotonic()
    def append(rows,row):
        rows.append(row)
        if not recording.is_set():
            while rows and rows[0][0]<row[0]-5.:rows.popleft()
    def write_bag(topic,msg,wall):
        data=serialize_message(msg)
        if recording.is_set():
            while pending_bag:
                _,old_topic,old_data,old_wall=pending_bag.popleft()
                bag.write(old_topic,old_data,old_wall)
            bag.write(topic,data,wall)
        else:
            now=time.monotonic();pending_bag.append((now,topic,data,wall))
            while pending_bag and pending_bag[0][0]<now-5.:pending_bag.popleft()
    def control():
        # Blocking pipe read in this process only; never runs on the policy thread.
        for line in sys.stdin:
            command=json.loads(line)
            if command.get('command')=='start' and not recording.is_set():
                trigger.update(request_monotonic_s=float(command['monotonic_s']),
                    received_monotonic_s=time.monotonic())
                recording.set()
        stop_requested.set()  # Parent exited or deliberately closed its pipe.
    crc=CRC()
    def on_state(msg):
        nonlocal bad_crc
        if frozen.is_set():return
        arrival=time.monotonic()
        if int(msg.crc)!=int(crc.Crc(msg)):
            bad_crc+=1;return
        accept('lowstate')
        row=(arrival,int(msg.tick),int(msg.mode_machine),
             [m.q for m in msg.motor_state[:29]],[m.dq for m in msg.motor_state[:29]],
             list(msg.imu_state.quaternion),list(msg.imu_state.gyroscope),
             list(msg.imu_state.accelerometer),[m.tau_est for m in msg.motor_state[:29]])
        with lock:
            if not frozen.is_set():append(states,row)
    def on_torso(msg):
        if frozen.is_set():return
        accept('torso')
        row=(time.monotonic(),list(msg.quaternion),list(msg.gyroscope),list(msg.accelerometer))
        with lock:
            if not frozen.is_set():append(torsos,row)
    def stamp(msg):return int(msg.header.stamp.sec)*1_000_000_000+int(msg.header.stamp.nanosec)
    def on_cloud(msg):
        accept('scans')
        arrival=time.monotonic();wall=time.time_ns()
        data=cloud_array(msg)
        xyz=np.stack([data[k] for k in ('x','y','z')],axis=-1).astype(np.float32)
        point_time=data['timestamp'].astype(np.float64) if 'timestamp' in data.dtype.names else np.full(len(data),np.nan)
        append(clouds,(arrival,stamp(msg),xyz,point_time))
        write_bag('/livox/lidar',msg,wall)
    def on_imu(msg):
        accept('lidar_imu')
        arrival=time.monotonic();wall=time.time_ns()
        append(imus,(arrival,stamp(msg),[msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z],
                     [msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z]))
        write_bag('/livox/imu',msg,wall)

    def on_odom(msg):
        if not recording.is_set():return
        arrival=time.monotonic();pose=msg.pose.pose
        if stamp(msg)<=0:return
        odometry.append((arrival,stamp(msg)*1e-9,[pose.position.x,pose.position.y,pose.position.z],
                         [pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w]))
        bag.write('/dlio/odom_node/odom',serialize_message(msg),time.time_ns())

    # Own SIGINT so it requests a graceful stop rather than interrupting the
    # final NPZ write or letting ROS shut itself down ahead of cleanup.
    from rclpy.signals import SignalHandlerOptions
    signal.signal(signal.SIGINT,lambda *_:stop_requested.set())
    signal.signal(signal.SIGTERM,lambda *_:stop_requested.set())
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node=rclpy.create_node('hv2_lidar_read_only_capture')
    _PROCESS_HANDLES.append(node)
    node.create_subscription(PointCloud2,'/livox/lidar',on_cloud,qos_profile_sensor_data)
    node.create_subscription(Imu,'/livox/imu',on_imu,qos_profile_sensor_data)
    node.create_subscription(Odometry,'/dlio/odom_node/odom',on_odom,2048)
    ChannelFactoryInitialize(0,args.interface)
    subscriber=ChannelSubscriber('rt/lowstate',LowState_)
    _PROCESS_HANDLES.append(subscriber)
    subscriber.Init(on_state)
    torso_subscriber=ChannelSubscriber('rt/secondary_imu',IMUState_)
    _PROCESS_HANDLES.append(torso_subscriber)
    torso_subscriber.Init(on_torso)
    if args.controlled:threading.Thread(target=control,daemon=True).start()
    started=time.monotonic();deadline=started+20.;record_start=None;last_print=0.;error=None
    ready=False;last_status=0.
    try:
        while not stop_requested.is_set():
            rclpy.spin_once(node,timeout_sec=.05)
            now=time.monotonic()
            if now-last_status>=.5:
                missing=[k for k in ('scans','lidar_imu','lowstate') if now-seen.get(k,-float('inf'))>2.]
                pending=out/'status.partial.json'
                pending.write_text(json.dumps(dict(monotonic_s=now,missing=missing,ready=not missing,
                    recording=recording.is_set(),counts=dict(scans=len(clouds),lidar_imu=len(imus),lowstate=len(states),torso_imu=len(torsos)))))
                pending.replace(out/'status.json');last_status=now
            if not ready and all(now-seen.get(k,-float('inf'))<1. for k in ('scans','lidar_imu','lowstate')):
                ready=True
                (out/'ready.json').write_text(json.dumps(dict(monotonic_s=now,controlled=args.controlled)))
                print('Sensor streams ready; waiting for policy A.' if args.controlled else 'Sensor streams ready.',flush=True)
            if recording.is_set() and clouds and imus and states and record_start is None:
                record_start=now;print('Required streams present; recording sensors (torso IMU optional).',flush=True)
            if recording.is_set() and now-last_print>=2.:
                print(f'scans={len(clouds)} lidar_imu={len(imus)} lowstate={len(states)} torso_imu={len(torsos)}',flush=True);last_print=now
            # A controlled capture saves only after the parent has stopped motors.
            if not args.controlled and record_start is not None and now-record_start>=args.seconds:break
            if not args.controlled and not ready and now>deadline:
                raise RuntimeError('Missing sensor stream after 20 s; saving partial capture')
    except (KeyboardInterrupt,Exception) as exc:
        error=f'{type(exc).__name__}: {exc}'
        if args.controlled:
            (out/'capture_error.json').write_text(json.dumps(dict(error=error)))
            print(f'Capture error: {error}. Waiting for deployment stop before compression.',flush=True)
            while not stop_requested.wait(.1):pass
    finally:
        # First stop mutations, including callbacks already building a row. No
        # subscriber destruction or unbounded SDK cleanup before durable saving.
        with lock:frozen.set()
        cleanup_errors=[]
        print('Saving sensor arrays; capture frozen. No motor commands.',flush=True)
    if args.controlled and not recording.is_set():
        clouds.clear();imus.clear();states.clear();torsos.clear()
    arrays={}
    if clouds:
        arrays.update(scan_receipt_monotonic_s=np.array([r[0] for r in clouds]),
            scan_stamp_ns=np.array([r[1] for r in clouds],dtype=np.int64),
            scan_offsets=np.r_[0,np.cumsum([len(r[2]) for r in clouds])],
            points=np.concatenate([r[2] for r in clouds]),point_timestamp_ns=np.concatenate([r[3] for r in clouds]))
    if imus:
        arrays.update(imu_receipt_monotonic_s=np.array([r[0] for r in imus]),
            imu_stamp_ns=np.array([r[1] for r in imus],dtype=np.int64),
            imu_gyro=np.array([r[2] for r in imus]),imu_accel=np.array([r[3] for r in imus]))
    if states:
        arrays.update(state_receipt_monotonic_s=np.array([r[0] for r in states]),
            state_tick_ms=np.array([r[1] for r in states],dtype=np.int64),
            mode_machine=np.array([r[2] for r in states]),q=np.array([r[3] for r in states]),
            dq=np.array([r[4] for r in states]),pelvis_quat_wxyz=np.array([r[5] for r in states]),
            pelvis_gyro=np.array([r[6] for r in states]),pelvis_accel=np.array([r[7] for r in states]),
            tau_est=np.array([r[8] for r in states],dtype=np.float32))
    if torsos:
        arrays.update(torso_receipt_monotonic_s=np.array([r[0] for r in torsos]),
            torso_quat_wxyz=np.array([r[1] for r in torsos]),
            torso_gyro=np.array([r[2] for r in torsos]),torso_accel=np.array([r[3] for r in torsos]))
    temporary=out/'sensors.partial.npz'
    np.savez_compressed(temporary,**arrays)
    temporary.replace(out/'sensors.npz')
    if odometry:
        np.savez_compressed(out/'odometry.npz',receipt_monotonic_s=np.array([r[0] for r in odometry]),
            stamp_s=np.array([r[1] for r in odometry]),position=np.array([r[2] for r in odometry]),
            quat_xyzw=np.array([r[3] for r in odometry]))
    print('Sensor arrays saved. Finalizing LiDAR bag...',flush=True)
    if recording.is_set():
        while pending_bag:
            _,topic,data,wall=pending_bag.popleft();bag.write(topic,data,wall)
    del bag
    report=dict(metadata,error=error,cleanup_errors=cleanup_errors,
        controlled=args.controlled,trigger=trigger,pre_roll_s=5. if args.controlled else 0.,
        cancelled_before_start=args.controlled and not recording.is_set(),
        stop_reason='operator_signal' if stop_requested.is_set() else 'duration_or_error',
        odometry_samples=len(odometry),elapsed_s=time.monotonic()-started,rejected_lowstate_crc=bad_crc,
        scans=rate_report([r[0] for r in clouds]),lidar_imu=rate_report([r[0] for r in imus]),
        lowstate=rate_report([r[0] for r in states]),torso_imu=rate_report([r[0] for r in torsos]),
        torso_clock='receipt time only; secondary IMU message has no capture timestamp')
    if clouds:
        points=arrays['points'];ranges=np.linalg.norm(points,axis=-1)
        report.update(scan_capture=rate_report(arrays['scan_stamp_ns']*1.e-9),
            points_per_scan_median=float(np.median(np.diff(arrays['scan_offsets']))),
            valid_range_fraction=float(np.mean(np.isfinite(ranges)&(ranges>.1))),
            range_quantiles_m=np.nanquantile(ranges,[0,.1,.5,.9,1]).tolist())
    if imus:report['imu_accel_norm_median']=float(np.median(np.linalg.norm(arrays['imu_accel'],axis=1)))
    if states:
        report.update(mode_machine_values=np.unique(arrays['mode_machine']).tolist(),
            joint_speed_abs_max_rad_s=float(np.max(np.abs(arrays['dq']))))
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    print(f'Saved {out}',flush=True)
    return 1 if error else 0


if __name__=='__main__':
    # All capture artifacts are explicitly finalized above. Let the OS reclaim
    # process-owned DDS listeners instead of running the SDK's blocking destructor.
    import traceback
    try:exit_code=main()
    except SystemExit as exc:
        exit_code=exc.code if isinstance(exc.code,int) else 1
        if isinstance(exc.code,str):print(exc.code,file=sys.stderr)
    except BaseException:
        traceback.print_exc();exit_code=1
    sys.stdout.flush();sys.stderr.flush()
    os._exit(exit_code)
