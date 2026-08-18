# SONIC Python policy modules

This directory owns model-contract and embodiment logic. Operator launch
selection belongs in `g1_deploy/workflows/`; experiment orchestration belongs
in `g1_deploy/experiments/`.

| Module | Responsibility |
| --- | --- |
| `model_profile.py` | Atomic encoder/decoder/observation contract |
| `robot_profile.py` | 29-to-23 sensor scatter and physical motor gather |
| `obs.py` | Released flat observation layouts |
| `planner_onnx.py` | V0/V1/V2 planner contract, recurrence, resampling, and splice |
| `inference_backend.py` | Direct TensorRT or ONNX Runtime execution |
| `reference_adaptation.py` | Nonmutating encoder-facing gait-reference scaling |
| `reduced_adaptation.py` | Explicit missing-axis proprioception research strategies |
| `action_adaptation.py` | Opt-in physical action shaping used by historical studies |
| `history.py` | Tracker temporal history |

The planner-owned 29-DoF trajectory must not be mutated by an embodiment
adapter. Official-parity zeros only unavailable measured q/dq and gathers only
the 23 physical outputs. `forward-stable-23` is the current 23-DoF
observation contract: all six absent reference/history axes are masked, and
outputs stay direct. `robust-small-gait-23` is the historical model-tuned
wrist-mask exception; its checkpoint-specific config is resolved by
`config/sonic_deployment_profiles.json`, not chosen piecemeal.

See the canonical explanation in
[`../../../docs/learning/sonic_deployment.md`](../../../docs/learning/sonic_deployment.md).
