# SONIC model code

| File | Purpose |
| --- | --- |
| `model/model_profile.py`, `deployment_profile.py` | Released model identities and tensor dimensions |
| `model/robot_profile.py` | 23/29-joint mapping and missing-joint treatment |
| `model/obs.py`, `history.py` | Encoder/decoder inputs and history |
| `model/planner_onnx.py` | Planner context, resampling and crossfade |
| `model/reference_adaptation.py` | Encoder-facing gait adaptation; planner context remains unchanged |
| `model/action_adaptation.py`, `reduced_adaptation.py` | Configured motor/proprioception adaptations |
| `model/inference_backend.py` | TensorRT and ONNX Runtime |
| `model/hardware_contract.py` | Required hardware inference settings |
| `sonic_runner.py` | Shared tracking loop used by simulation and hardware |

The model always has 29 joint slots. A 23-DoF robot omits missing motors at the
output boundary. Its measured missing-joint values follow the selected profile.
Encoder/decoder binaries and observation YAMLs must come from the same release.
