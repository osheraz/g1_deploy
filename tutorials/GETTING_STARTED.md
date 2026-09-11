# Getting started

For the current 29-joint diffusion/flow pipeline:

1. [Connect to the robot](01_network.md). Preserve the workstation's Internet route.
2. [Use the prepared environment](02_install.md).
3. [View robot state and raw depth](04_cameras.md). This publishes no motor commands.
4. [Configure the paired model bundle and hardware](03_config.md).
5. Follow [diffusion deployment](deploy_diffusion.md): validate models, check
   real sensors without motor output, then commission actuation separately.

The [status page](deployment_status.md) separates completed checks from remaining
hardware work. A working viewer is not permission or evidence to actuate.

| Component | Where it runs |
| --- | --- |
| `deploy_diffusion.py`: planner worker + action decoder + watchdog | Inference workstation |
| LowState / optional LowCmd | Robot controller over Unitree DDS |
| `native_depth_server.py` | Jetson connected to the RealSense |
| `sonic/check_streams.py`: read-only Viser | Workstation |

The raw depth publisher does not publish RGB. Its timestamped PNG packets are
not the old filtered RGB-D stream.

Other workflows: [SONIC](../sonic/workflows/README.md),
[optional tactile](06_tactile.md), [robot migration record](08_robot_migration.md).
Old VIGOR2 instructions are isolated under [deprecated](../deprecated/README.md).
