# Model and hardware configuration

Current deployment uses [`configs/diffusion.yaml`](../configs/diffusion.yaml).
It selects the model bundle, network, camera and optional neural exports.
It inherits the existing 29-joint mapping/limits, but **not** SONIC's policy
gains or action scale: those come from the selected decoder's verified bundle.
Clock alignment is performed at startup. Ordinary depth-based policy execution needs no separate hardware JSON. Live XYZ
monitoring uses the explicit mount candidate; qualified XYZ feedback additionally
requires an independent clock calibration that is not yet available.

`--backend tensorrt` selects compiled inference; `--actuate` enables the motor
path: startup is idle; A starts policy from the current pose, A again stops in
damping and exits. Before starting only, START optionally moves to default over
two seconds, then holds for A. SELECT cancels/stops; Ctrl+C also stops.
`--default-pose --actuate` waits for START, moves and holds; SELECT or A stops.
If the built-in controller is active, add `--release-controller` with the robot
already supported. It requests the SDK handover on A/START, after the startup
pose check and before motor output; otherwise an active controller blocks takeover.
`--config` is optional and selects another YAML when needed.
The current config explicitly permits testing the older bundle with the new
lens, logs the mismatch, and does not rewrite saved model calibration.

The planner predicts action codes; the paired decoder consumes those codes and
measured robot history. Neither the PPO teacher nor a target motion encoder
runs at inference. Simulation and hardware share that implementation.

The `configs/g1_29dof_sonic.yaml` argument used by the **read-only viewer** supplies
stream settings and motor order only. It does not select the diffusion model,
and `sonic/check_streams.py` does not load a SONIC policy.

See [the runtime reference](deployment_reference.md#model-bundle) for bundle validation and
[the camera guide](04_cameras.md) for the current robot connection.
Legacy VIGOR YAML instructions are [archived separately](../deprecated/tutorials/config_vigor.md).
