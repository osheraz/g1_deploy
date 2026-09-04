# 29-DoF Mid-360 fall recovery

This path uses three independent Mid-360 scans, not an elevation map. Each scan
is transformed into a pelvis-centred gravity/heading frame using the physical
torso IMU and current waist FK. The scans are voxelized independently and fused
by the checkpoint's temporal encoder. No world pose or scan registration is
used.

## Train the matching planner

The generic architecture is
`motion_diffusion/config/algo/fallrec/unified/goal_steer_lidar.yaml`. Train with
the audited dataset-pinned leaf
`goal_steer_lidar_lafan_i9000_20260902.yaml`: it selects the realized 29-DoF
train split, its split manifest and receipt, and the matching fork-terrain
snapshot. Do not substitute the teacher source-reference corpus or a 23-DoF
pack.

```bash
cd /home/osher/hv2
PYTHONPATH="$PWD" /home/osher/miniforge3/envs/hvsim/bin/python \
  motion_diffusion/train_motion_diffusion.py \
  training.cfg_path=motion_diffusion/config/algo/fallrec/unified/goal_steer_lidar_lafan_i9000_20260902.yaml \
  training.out_dir=/absolute/path/to/a/new/output/root
```

The saved `model.pt` embeds the full sensor and encoder contract. Deployment
rejects a point-count or contract mismatch.

## Required robot-side streams

Source the ROS 2 installation and the workspace containing
`livox_ros_driver2` before starting deployment. The driver must publish custom
messages (`xfer_format=1`) and IMU:

```bash
ros2 topic type /livox/lidar
ros2 topic type /livox/imu
ros2 topic hz /livox/lidar
ros2 topic hz /livox/imu
python -c 'from rclpy.node import Node; from livox_ros_driver2.msg import CustomMsg'
```

Set the driver publish frequency to 30 Hz. Deployment aggregates the timestamped
packets into the checkpoint's rolling 100-ms firing window and rejects a
sustained rate below 20 Hz.

The expected LiDAR type is `livox_ros_driver2/msg/CustomMsg`; the expected IMU
type is `sensor_msgs/msg/Imu`. The G1 SDK side also requires the physical torso
IMU on `rt/secondary_imu`. The pelvis IMU remains the tracker input.

## Configure and validate without commands

Copy `vigor2/fallrec_lidar_29dof.yaml` to a run-specific file and set:

- `planner_model` to the LiDAR-conditioned `model.pt` produced above.

The committed file already pins the native 29-DoF tracker paired with the
realized training dataset. Do not replace it with the old 23-DoF checkpoint.

Then run the complete sensor/planner/tracker path without publishing LowCmd:

```bash
cd /home/osher/hv2/g1_deploy
conda activate g1_deploy
python deploy_vigor2.py \
  --config g1_29dof_sonic.yaml \
  --cfg vigor2/fallrec_lidar_29dof.yaml \
  --no-publish \
  --log LIDAR_NO_PUBLISH.npz
```

The runtime stops instead of degrading if the torso IMU is absent/stale, the
LiDAR or its IMU is absent/stale, scan deskew lacks synchronized IMU samples,
the nearest LowState/torso sample is too far from scan receipt, or the saved
sensor contract does not match deployment.
