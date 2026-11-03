# Manual standing mode — Y

**Purpose: press Y after get-up to stop further recovery motion and settle into standing.**
It does not wait for an automatic standing detector on the robot. It works with
steered or unguided recovery. Enable it with `--standing-hold` on your existing
robot command; the normal recovery is unchanged until you press Y.

- **A:** start recovery and recording. START remains optional joint preparation.
- **Y after get-up:** stop planner requests, discard queued plans, and blend the
  last executed latent into the paired tracker's original encoded standing
  reference over 0.5 seconds. The frozen tracker keeps balancing with live
  joints, IMU, depth and actual issued-action history.
- **A again / SELECT / Ctrl+C:** existing stop behavior. Y does not stop motors.
- Y before A is ignored. Repeated Y does not restart the transition. Standing
  mode remains latched until the run ends; a later fall does not resume planning.

The reference requests the checkpoint's default standing joints, zero joint
velocities, upright pelvis and the heading at Y. It uses the original encoder
and frozen normalizer. PD gains, action scaling and the startup-only limiter are
unchanged. There is no fixed motor-target hold or additional post-Y slew limit.
The encoder is loaded and warmed before startup; Y does no loading/compilation.

Standing does not require DLIO: the reference uses local coordinates. It does
**not** lock a world position. Steering before Y retains its existing feedback
requirements. Sensor/model/external-camera recording remains unchanged; the Y
activation, original encoder inputs, native codes and tracker outputs are recorded
for model replay. New robot sessions remain under `logs/real_world`.

This optional control is separate from the published autonomous recovery
evaluation and requires qualification for the selected pair.
