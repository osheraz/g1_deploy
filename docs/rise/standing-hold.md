# Manual standing mode — Y

Add `--standing-hold` to a steered or unguided robot command. **Press Y after get-up
to request standing.** It takes effect immediately, without waiting for a standing
detector. Recovery runs normally until Y.

- **A:** start recovery and recording. START is optional joint preparation.
- **Y:** stop planning, discard queued plans and blend to the tracker's encoded
  standing reference over 0.5 seconds. The tracker still uses live joints, IMU,
  depth and issued-action history to balance.
- **A again / SELECT / Ctrl+C:** stop. Y itself does not stop motors.

Y before A is ignored. Repeated Y does not restart the blend. Standing mode lasts
until the run ends; planning does not resume after a fall.

The reference requests default standing joints, zero joint velocities, an upright
pelvis and the heading measured at Y. It uses the original encoder and normalizer,
loaded and warmed before startup.

PD gains, action scaling and the startup-only limiter stay unchanged. The tracker
continues predicting actions; Y adds neither fixed motor targets nor a slew limit.

Standing uses local coordinates and needs no DLIO. It does **not** hold a world
position. Steering before Y still requires its usual feedback.

Recording continues under `logs/real_world`, including Y activation, encoder inputs,
latents and tracker outputs for model replay.

This manual mode is outside the published autonomous evaluation. Validate it
with the selected model pair.
