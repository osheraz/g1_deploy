"""Run DLIO on a saved ROS bag and record its output. Never starts a sensor driver.

Source ROS and the built DLIO workspace first. Use an isolated nonzero ROS
domain. Optional UDP output is for an isolated monitoring receiver only.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

import numpy as np
# Keep ROS system Python independent of the training package dependencies.
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'motion_diffusion/behavioral/hardware'))
from livox_imu_si import imu_to_si
from livox_cloud_window import join_cloud_packets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bag',type=Path);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--rate',type=float,default=1.)
    p.add_argument('--timeout-s',type=float,default=90.)
    p.add_argument('--imu-unit',choices=['g','si'],required=True,help='Raw bag units: Livox hardware g, simulator si')
    p.add_argument('--scan-packets',type=int,default=1,help='Diagnostic: concatenate N consecutive raw scans before DLIO')
    p.add_argument('--sim-state',type=Path,help='Also exercise the real pelvis bridge with recorded simulation joint/IMU states')
    p.add_argument('--sim-env',type=int,default=0)
    p.add_argument('--hardware-capture',type=Path,help='Replay recorded low state causally, without DDS or robot access')
    p.add_argument('--mount-calibration',type=Path)
    p.add_argument('--odometry-port',type=int)
    a=p.parse_args()
    if int(os.environ.get('ROS_DOMAIN_ID','0'))==0:p.error('Use nonzero ROS_DOMAIN_ID')
    if not a.bag.exists() or not a.config.exists():p.error('Bag and configuration must exist')
    if not 0<a.rate<=2 or not 0<a.timeout_s<=900:p.error('Invalid rate or timeout')
    if not 1<=a.scan_packets<=10:p.error('scan-packets must be 1..10')
    if a.hardware_capture and (a.sim_state or a.rate!=1.):p.error('Hardware state replay requires rate 1 and no sim state')
    if a.odometry_port and not a.hardware_capture:p.error('UDP diagnostic output requires hardware-capture')
    import rclpy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu,PointCloud2
    from rclpy.qos import qos_profile_sensor_data
    a.output.mkdir(parents=True,exist_ok=False)
    config_copy=a.output/'dlio_config.yaml';shutil.copyfile(a.config,config_copy)
    rows=[];children=[];logs=[];error=None
    rclpy.init();node=rclpy.create_node('hv2_offline_dlio_recorder')
    pub=node.create_publisher(Imu,'/validation/imu_si',qos_profile_sensor_data)
    node.create_subscription(Imu,'/livox/imu',lambda msg:pub.publish(imu_to_si(msg,a.imu_unit)),qos_profile_sensor_data)
    cloud_pub=node.create_publisher(PointCloud2,'/validation/lidar',10)
    pending=[]
    def cloud_callback(msg):
        pending.append(msg)
        if len(pending)<a.scan_packets:return
        cloud_pub.publish(join_cloud_packets(pending));pending.clear()
    node.create_subscription(PointCloud2,'/livox/lidar',cloud_callback,qos_profile_sensor_data)
    def callback(msg):
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        if stamp<=0:return
        pose=msg.pose.pose
        rows.append((time.monotonic(),stamp,pose.position.x,pose.position.y,pose.position.z,
                     pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w))
    node.create_subscription(Odometry,'/validation/odom',callback,2048)
    def launch(command,name):
        (a.output/(name+'_command.json')).write_text(json.dumps(command,indent=2)+'\n')
        log=open(a.output/(name+'.log'),'w');logs.append(log)
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(proc);return proc
    bridge=None
    if a.hardware_capture:
        from pelvis_odometry import G1Body,PelvisBridge
        from replay_motion_state import RecordedStateReader
        state=RecordedStateReader(a.hardware_capture/'sensors.npz')
        # Subscription order ensures the replay time base exists before odometry.
        node.create_subscription(Imu,'/livox/imu',lambda msg:state.observe_imu(
            PelvisBridge.stamp_s(msg.header),time.monotonic()),qos_profile_sensor_data)
        mount=None if a.mount_calibration is None else json.loads(a.mount_calibration.read_text())['rotation_lidar_to_torso']
        bridge=PelvisBridge(node,G1Body(load_body_boxes=False),state,a.odometry_port,None,
            record=a.output/'measured_pose.jsonl',mount_rotation=mount,
            odometry_topic='/validation/odom',publish_diagnostics=True)
    if a.sim_state:
        from pelvis_odometry import G1Body,NpzStateReader,PelvisBridge
        bridge=PelvisBridge(node,G1Body(load_body_boxes=False),NpzStateReader(a.sim_state,a.sim_env,1.7e9),
            None,None,3.,record=a.output/'measured_pose.jsonl',odometry_topic='/validation/odom')
    started=time.monotonic()
    try:
        estimator=launch(['ros2','run','direct_lidar_inertial_odometry','dlio_odom_node',
            '--ros-args','--params-file',str(config_copy.resolve()),'-p','use_sim_time:=true',
            '-p','imu/intrinsics/accel/sm:=[1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]',
            '-r','pointcloud:=/validation/lidar','-r','imu:=/validation/imu_si','-r','odom:=/validation/odom'],'dlio')
        while time.monotonic()-started<2:rclpy.spin_once(node,timeout_sec=.05)
        player=launch(['ros2','bag','play',str(a.bag.resolve()),'--clock','100','--rate',str(a.rate)],'player')
        ended=None;last_print=0
        while True:
            rclpy.spin_once(node,timeout_sec=.02)
            now=time.monotonic()
            if estimator.poll() is not None:raise RuntimeError('DLIO exited; see dlio.log')
            if player.poll() is not None:
                if player.returncode!=0:raise RuntimeError('Bag playback failed; see player.log')
                if ended is None:ended=now
                if now-ended>2:break
            if now-started>a.timeout_s:raise TimeoutError('Replay exceeded timeout')
            if now-last_print>5:
                print(f'Offline replay: {len(rows)} odometry samples',flush=True);last_print=now
    except (Exception,KeyboardInterrupt) as exc:error=f'{type(exc).__name__}: {exc}'
    finally:
        for proc in children:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGINT)
        for proc in children:
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        for log in logs:log.close()
        if bridge is not None and bridge.record:bridge.record.close()
        node.destroy_node();rclpy.shutdown()
    values=np.array(rows,dtype=float).reshape(-1,9)
    np.savez_compressed(a.output/'odometry.npz',receipt_monotonic_s=values[:,0],stamp_s=values[:,1],
                        position=values[:,2:5],quat_xyzw=values[:,5:9])
    registered='Registration' in (a.output/'dlio.log').read_text(errors='replace')
    if not registered and error is None:error='DLIO produced no scan-registration report; inspect topic/QoS wiring'
    report=dict(bag=str(a.bag),config=str(a.config),rate=a.rate,samples=len(rows),error=error,raw_imu_unit=a.imu_unit,
                scan_registration_observed=registered,
                scan_packets=a.scan_packets,
                sensor_driver_started=False,motor_publisher=False,elapsed_s=time.monotonic()-started)
    if a.hardware_capture:report['recording_to_replay_monotonic_shift_s']=state.shift
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if error or not rows:raise SystemExit(1)


if __name__=='__main__':main()
