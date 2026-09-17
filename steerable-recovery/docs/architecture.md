# Code and data flow

| Owner | Responsibility |
| --- | --- |
| `steerable_recovery/cli` | Hardware and MuJoCo argument handling |
| `steerable_recovery/presets` | Saved starts/routes; calls the same runtimes |
| `steerable_recovery/paths.py`, `settings.py` | Workspace roots and model settings |
| `steerable_recovery/sensing`, `recording` | Sensor processes, triggered capture and cleanup |
| `steerable_recovery/review` | Offline preparation and model replay |
| `steerable_recovery/tools` | Diagnostics, experiments, benchmarks and figure exports |
| `g1_deploy/core` | Shared robot I/O, episode formats and Viser |
| `motion_diffusion/behavioral/hardware` | Asynchronous inference, histories, action limits, measured feedback |
| `motion_diffusion/behavioral/runtime.py` | Learned planner/tracker contract shared with simulation |

The robot-side depth server captures at 30 Hz, publishes the latest pending frame
on a worker and serves clock replies separately. The workstation receiver pairs
depth with robot state and prepares model observations; a recording worker saves
full-resolution pixels. The 50 Hz tracker does not encode images.

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
planner's paired tracker. Existing bundles/engines remain under
`logs/native_recovery/`; this cleanup does not regenerate them.

The standalone camera deployment consists of **two loose files**:
`sensing/native_depth_server.py` and `core/common/depth_wire.py`.
ROS worker scripts bootstrap the package from their source checkout because
system Python does not use the policy virtualenv.

Steering measurement contract:
[Stanley integration note](../../../notes/stanley_steering_measurement_contract.md).
Historical guides are under `docs/archive/`; they are not current launch instructions.
