# Code and data flow

| Owner | Responsibility |
| --- | --- |
| `steerable_recovery/cli` | Hardware and MuJoCo argument handling |
| `steerable_recovery/paths.py`, `settings.py` | Workspace roots and model settings |
| `steerable_recovery/sensing`, `recording` | Sensor processes, triggered capture and cleanup |
| `steerable_recovery/review` | Offline preparation and model replay |
| `steerable_recovery/tools` | Diagnostics, benchmarks and figure exports |
| `g1_deploy/core` | Shared robot I/O, episode formats and Viser |
| `steerable_recovery/hardware` | Unitree/ROS transports, device lifecycles, hardware sessions |
| `motion_diffusion/inference` | Portable tracker/planner composition, asynchronous execution, histories and release rules |
| `motion_diffusion/planner/runtime.py` | Frozen sampling and code scheduling shared with simulation |
| `humanoidverse2/simulator/mujoco` | Recovery plant, raw simulated sensors, terrain and compiled scene cache |

The robot-side depth server captures at 30 Hz, publishes the latest pending frame
on a worker and serves clock replies separately. The workstation receiver pairs
depth with robot state and prepares model observations; a recording worker saves
full-resolution pixels. A visual tracker consumes delivered depth together with
measured history; a blind tracker does not receive images. This follows the
selected model's contract.

The planner runs in a separate process. Tracking consumes the latest accepted
plan with measured proprioception. LiDAR/DLIO and raw/camera recording run as
separate CPU processes. This avoids blocking control on their work, but they
still share CPU, network and disk resources. [Timing checks](performance.md).

Tracker startup repeats the measured state and clears requests; planner startup
marks the latest measured state valid and clears latent history. Depth preserves
delivered pre-start exposures. Startup shaping feeds the issued action back into
history and applies only for 0.5 s. These semantics were not changed by the move.

`config/models.yaml` selects immutable bundle pairs and export paths. Verified
training PD, action scale and ordering come from the bundle. Do not exchange a
planner's paired tracker. The primary durable bundle is identified by the
[release guide](../../../docs/guides/deployment.md).

The standalone camera deployment consists of **two loose files**:
`sensing/native_depth_server.py` and `core/common/depth_wire.py`.
ROS worker scripts bootstrap the package from their source checkout because
system Python does not use the policy virtualenv.

[Sensing contracts](../../../docs/guides/sensing.md) cover depth and measured-pose inputs.
Historical guides remain available in Git history.
