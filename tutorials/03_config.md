# Model and hardware configuration

Current deployment uses [`configs/diffusion.yaml`](../configs/diffusion.yaml).
It selects the model bundle, network, camera and optional neural exports.
It inherits the existing 29-joint mapping/limits, but **not** SONIC's policy
gains or action scale: those come from the selected decoder's verified bundle.
Clock alignment is performed at startup. There is no separate hardware JSON
or manual calibration-receipt workflow in the normal launcher.

`--backend tensorrt` selects compiled inference; `--actuate` enables the motor
path: START moves slowly to default, then A starts the policy. SELECT stops.
`--default-pose --actuate` only moves to default and holds; A is ignored.
If the built-in controller is active, add `--release-controller` with the robot
already supported. It requests the SDK handover before startup; otherwise the
ownership guard rejects the launch.
`--config` is optional and selects another YAML when needed.
The current config explicitly permits testing the older bundle with the new
lens, logs the mismatch, and does not rewrite saved model calibration.

The planner predicts action codes; the paired decoder consumes those codes and
measured robot history. Neither the PPO teacher nor a target motion encoder
runs at inference. Simulation and hardware share that implementation.

The `configs/g1_29dof_sonic.yaml` argument used by the **read-only viewer** supplies
stream settings and motor order only. It does not select the diffusion model,
and `sonic/check_streams.py` does not load a SONIC policy.

See [deployment commands](deploy_diffusion.md) for bundle validation and
[the camera guide](04_cameras.md) for the current robot connection.
Legacy VIGOR YAML instructions are [archived separately](../deprecated/tutorials/config_vigor.md).
