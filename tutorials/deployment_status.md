# Deployment status

Updated 2026-09-13. Commands: [operator guide](deploy_diffusion.md).
Steering semantics: [Stanley's measurement contract](../../notes/stanley_steering_measurement_contract.md).

## Implemented and checked

- Diffusion/full horizon and flow/replanning use the paired action decoder.
  TensorRT FP32 is the default. The planner runs separately from the 50 Hz tracker.
- Depth remains the planner input. `--live-odometry` adds live DLIO/FK pelvis XYZ
  in monitoring mode; A starts the policy and raw recording together.
- Shared fall-aware heading and full relative orientation are logged. The head
  mount is fixed; measured waist joints convert LiDAR pose to pelvis pose.
- Stationary hoist compute tests with inference, depth, odometry and recording:
  diffusion control work **6.3–6.7 ms p95**, flow **7.0 ms p95**. These were
  sensor-only tests, not dynamic accuracy or recovery-success measurements.
  [Compute evidence and architecture](../../notes/deployment_compute.md).
- Sensor workspace rebuilt at `~/.local/share/hv2/lidar/ros2_ws`. The Livox
  shutdown logging-library conflict is patched: successful 20-second recording,
  two clean driver restarts, 19 focused tests passed.
  [Installation evidence](../../notes/validation/lidar_workspace_20260913.json).
- A ground policy recording exists. Its heading disagreement reached **10.3°**;
  subsequent causal replay also showed tilt dropouts near **10°**. This is not
  proof of translation accuracy. [Ground evidence](../../notes/validation/steering_measurements_20260912.json),
  [live/replay evidence](../../notes/validation/live_odometry_20260913.json).

## Still open

| Requirement | What is missing |
| --- | --- |
| Clock qualification | Independent LiDAR/robot-to-host offset bounds. Arrival jitter and gyro correlation do not supply them. |
| Dynamic XYZ accuracy | Independent reference during recovery. Fixed video/floor marks are the next gross horizontal-motion check; quantitative full XYZ needs calibrated geometry or another 3D reference. |
| Tilt dropouts | Validate the mount candidate on held-out moving data; separate timing, mount rotation, deflection and estimator error. Keep failed estimates visible. |
| Contact-dependent steering | Hardware support-contact estimator and validation. |
| ETH elevation map | Built/imported only; live map chain and prone performance remain unvalidated. Deferred while depth stays in the planner. |

XYZ remains **monitoring only**. Do not enable `--odometry-mode required` or
position-dependent steering without qualified measurements. Stanley owns steering
training/integration; simulator truth must stay labelled as oracle feedback.

Next operator action: the [ground trial in the operator guide](deploy_diffusion.md#2-recovery--live-xyz-recording--terminal-2),
with external video and a new session. Current-pose startup can still produce a
rejected saturated decoder output for some inputs; numerical backend parity does
not establish that every starting pose is supported. See [runtime reference](deployment_reference.md#control-and-startup-behavior).

Superseded commissioning narratives are retained in Git history, not as current
instructions. Dated evidence files retain their original results and limitations.
