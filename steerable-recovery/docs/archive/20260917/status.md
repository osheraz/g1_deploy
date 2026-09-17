# Deployment status

Code/documentation reviewed 2026-09-15; hardware evidence below retains its trial dates. Commands: [operator guide](deploy_diffusion.md).
Steering semantics: [Stanley's measurement contract](../../notes/stanley_steering_measurement_contract.md).

## Implemented and checked

- **Three-route steering integration:** Stanley's shared pose costs, PyTorch
  planner / TensorRT tracker, recorded goals and measured-feedback fallback.
  Three MuJoCo trials are not a successful demo (one reached upright; none
  finished standing). Hardware XYZ is still unqualified. [Procedure](steering.md),
  [checks](../../notes/validation/steering_20260915.md).

- **2026-09-15 XYZ update:** DLIO publishes a mutex-protected pose/time snapshot;
  the live bridge applies a 20 ms causal position filter and retains raw XYZ.
  Filter delay is included in steering freshness checks. Replay displays finite
  tilt-rejected estimates in orange, without qualifying them for steering.
  On recorded inputs, p95 per-tick XYZ changes fell from 3.91 to 2.44 cm;
  p95 added delay was 15.4 ms. This measures jitter reduction, not accuracy.
  [Investigation and limits](../../notes/validation/xyz_tracking_20260915.md).
- **Current operator commands select R8_S25 diffusion40k + R8+6k depth tracker**,
  TensorRT, fresh noise per plan, K20 (2.5 Hz) and a 0.5 s horizon. Preset
  positioning, random seeds, startup target limits, live XYZ and recording/replay
  options are listed in the operator guide. No full-horizon hardware mode was retained.
- **P11 diffusion20k + N17k H3 depth tracker remains the CLI default**, using
  selected PyTorch TF32/DDIM8/fresh noise/K20/blend0. The planner runs separately
  from the 50 Hz tracker. Legacy diffusion/flow retain their older models.
- P11 offline checks: pinned checkpoint hashes and training PD match; full CPU
  planner + depth tracker outputs match Stanley's `43170d4` exactly for three
  missing/partial/full-depth cases. TensorRT FP32 matches FP32, but differs from
  selected TF32; it remains an explicit precision comparison. [Details](deployment_reference.md).
  The actual deployment warmup also passes with both models and the planner
  process, leaving live histories empty and issuing no physical commands.
  These checks do not qualify P11 on physical sensors or motors; subsequent
  operator trials need their own session/measurement review.
- MuJoCo depth geometry is corrected: the legacy MJCF head previously blocked
  every camera ray. Both our earlier code and Stanley's `43170d4` reproduce this.
  The ray-only scene now uses training URDF visuals and matches the training
  renderer at checked poses (identical masks, up to 1 mm depth difference).
  Earlier all-zero-depth simulation comparisons are superseded. P05 and P11
  each passed four corrected smoke runs at matched 2.5 Hz; this is not a model
  ranking. [Audit and reproduction](../../notes/validation/mujoco_depth_20260913.md).
- Depth feeds both P11 planner and tracker. `--live-odometry` adds live DLIO/FK pelvis XYZ
  in monitoring mode; A starts the policy and raw recording together.
- Shared fall-aware heading and full relative orientation are logged. The head
  mount is fixed; measured waist joints convert LiDAR pose to pelvis pose.
- Earlier-model stationary hoist tests with inference, depth, odometry and recording:
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

For a new or changed setup: the [current sensor-only hoist check](deploy_diffusion.md#optional-checks-and-live-xyz),
then the ground trial with external video and a new session. Current-pose startup can still produce a
rejected saturated decoder output for some inputs; numerical backend parity does
not establish that every starting pose is supported. See [runtime reference](deployment_reference.md#control-and-startup-behavior).

Superseded commissioning narratives are retained in Git history, not as current
instructions. Dated evidence files retain their original results and limitations.
