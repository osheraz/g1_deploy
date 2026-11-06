# LiDAR and DLIO

Build the pinned ROS 2 Humble stack:

```bash
bash g1_deploy/rise/setup/build_lidar_workspace.sh
```

Default workspace: `~/.local/share/hv2/lidar/ros2_ws`.
`LIDAR_STACK_ROOT` selects the build root; `LIDAR_WS` selects it at runtime.
The build script pins source revisions and applies the committed patches.
It does not start sensors or install system dependencies.

Configuration lives in
[`rise/config/lidar`](../../rise/config/lidar).
Adjust `MID360_bluefin.json` for another host. The recording worker needs the
Unitree SDK on its system-Python path.

The recorder saves raw Livox acceleration; DLIO receives it converted from g to SI.
Each estimator window contains three 30 Hz cloud packets, with point timestamps
for motion correction. Recording also saves the mount calibration.
Dynamic XYZ accuracy has not been independently verified.

`livox_system_spdlog.patch` resolves the SDK/ROS logging-library conflict at
shutdown. `dlio_initialized_pose.patch` prevents odometry publication before IMU
initialization. Neither changes the planner's depth input.
