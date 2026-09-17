# Deployment setup

**Already on the configured bluefin workstation?** Go directly to
[the recovery operator guide](../steerable-recovery/docs/deploy.md). Setup and viewers are not extra steps
required before every deployment.

For a new machine or robot:

1. [Network and SSH](01_network.md): configure the robot interface and `g1_29`.
2. [Environment](02_install.md): hv2, its pinned deployment submodule and dependencies.
3. [Model/hardware config](03_config.md): bundle, exports, PD binding and camera.
4. [LiDAR workspace](../steerable-recovery/config/lidar/README.md): required for raw sensor recording
   and live odometry; permanent default is `~/.local/share/hv2/lidar/ros2_ws`.
5. [Operator guide](../steerable-recovery/docs/deploy.md): the complete two-terminal run.

[Sensor viewers](04_cameras.md) are optional diagnostics.
[Deployment status](../steerable-recovery/docs/status.md) lists measured results and remaining work.
SONIC has [separate operator workflows](../sonic/workflows/README.md).
