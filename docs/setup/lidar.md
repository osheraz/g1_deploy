# LiDAR and DLIO

Build the pinned ROS 2 Humble stack:

```bash
bash g1_deploy/steerable-recovery/setup/build_lidar_workspace.sh
```

Default workspace: `~/.local/share/hv2/lidar/ros2_ws`.
`LIDAR_STACK_ROOT` selects the build root; `LIDAR_WS` selects it at runtime.
The build script pins source revisions and applies the committed patches.
It does not start sensors or install system dependencies.

Configuration lives in
[`steerable-recovery/config/lidar`](../../steerable-recovery/config/lidar).
Adjust `MID360_bluefin.json` for another host. The recording worker needs the
Unitree SDK on its system-Python path.

Raw Livox IMU acceleration is saved unchanged and converted from g to SI for
DLIO. Three 30 Hz cloud packets form one estimator window, preserving point
timestamps for deskew. The live recovery path records its selected mount
calibration. Independent dynamic XYZ accuracy remains unverified.

`livox_system_spdlog.patch` resolves the SDK/ROS logging-library conflict at
shutdown. `dlio_initialized_pose.patch` prevents odometry publication before IMU
initialization. Neither changes the planner's depth input.
