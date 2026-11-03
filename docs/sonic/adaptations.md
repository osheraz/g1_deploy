# SONIC adaptations

| Setting | Effect |
| --- | --- |
| `official-parity` | Native encoder/reference/history behavior, with physical motor projection |
| `forward-stable-23` | Explicit 23-DoF reference/history masks and direct motor output |
| `--slow-gait` | Encoder-facing sagittal scale 0.28, knee offset +8°, walking/in-place yaw scales 0.15/0.35 |

`forward-stable-23` uses
[`g1_23dof_forward_stable.yaml`](../../configs/adaptations/g1_23dof_forward_stable.yaml).
It is only valid for `g1_23dof`; it does not alter the planner's recurrent 29-joint
trajectory. `--slow-gait` is exclusive with explicit sagittal/hip-pitch scales.
No adaptation is a substitute for validation of the selected robot/model pair.
