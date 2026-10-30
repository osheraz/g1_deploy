# SONIC Python policy modules

This directory owns model-contract and embodiment logic. Operator launch
selection belongs in `g1_deploy/sonic/workflows/`; experiment orchestration belongs
in `g1_deploy/sonic/experiments/`.

| Module | Responsibility |
| --- | --- |
| `model_profile.py` | Atomic encoder/decoder/observation contract |
| `robot_profile.py` | 29-to-23 sensor scatter and physical motor gather |
| `obs.py` | Released flat observation layouts |
| `planner_onnx.py` | V0/V1/V2 planner contract, recurrence, resampling, and splice |
| `inference_backend.py` | Direct TensorRT or ONNX Runtime execution |
| `reference_adaptation.py` | Nonmutating encoder-facing gait-reference scaling; `LandingHoldAdapter` is a withdrawn attempt |
| `gait_templates/` | Legacy mocap-cycle npz for the withdrawn hybrid-gait attempt |
| `reduced_adaptation.py` | Explicit missing-axis proprioception research strategies |
| `action_adaptation.py` | Opt-in physical action shaping used by historical studies |
| `history.py` | Tracker temporal history |

The planner-owned 29-DoF trajectory must not be mutated by an embodiment
adapter. Official-parity zeros only unavailable measured q/dq and gathers only
the 23 physical outputs. `forward-stable-23` is the current 23-DoF
observation contract: all six absent reference/history axes are masked, and
outputs stay direct. `robust-small-gait-23` is a **withdrawn legacy
attempt** (backward-selected wrist-mask / hip-only scale). It is blocked
from live launch. MuJoCo reproduction needs `--legacy-reproduction`.

`sonic/gait_reference.py` and `gait_templates/` are a **legacy attempt**
to replace planner lower-body with mocap cycles (`--gait-templates`).
`LandingHoldAdapter` / `--landing-hold{,-dvel}` is a second **legacy
attempt** to suppress sagittal swing after touchdown. Neither is the
current slow walk. Keep official Slow Walk 0.4 and use `--slow-gait` /
`--sagittal-scale`.

For launch commands, see the [joystick workflows](../workflows/joystick/README.md).
