# Native planner/tracker inference

The default deployment and MuJoCo backend is TensorRT FP32 with CUDA graph replay.
The tracker stays FP32. Fixed-shape input/output buffers are reused inside each
engine, and a private CUDA stream is ordered with the caller. Returned predictions
own their storage: the second Heun forward cannot overwrite the first prediction.
A separate planner process keeps sampling off the 50 Hz tracker thread.

```bash
# Offline: no robot, camera, DDS or tunnel needed.
.venv/bin/python g1_deploy/benchmark_diffusion.py
# Matched closed-loop precision comparison, six postures and three seeds each.
.venv/bin/python g1_deploy/evaluate_recovery.py
```

The benchmark compares PyTorch, original TensorRT execution, reusable buffers,
CUDA graphs, and an experimental TF32 planner with an FP32 tracker. It saves
GPU/runtime identities, same-input/noise numerical errors, warmed p50/p95/p99/max
latency, concurrent planner IPC latency, and tracker deadline misses. Each profile
uses the saved DDIM8/full-horizon or Heun16/replanning recipe. The concurrent test
runs fixed synthetic input fixtures at 50 Hz, including H2D/D2H and worker IPC.
It excludes camera capture, history construction, robot networking and DDS.
Planner replies are polled at 50 Hz, so observed reply latency includes that delay.
It does not simulate policy feedback or validate recovery.

`evaluate_recovery.py` runs matched postures, seeds and joint perturbations under
PyTorch, TRT FP32 and experimental TRT TF32. Its simulator pauses physics while
inference runs. It saves every trajectory and a comparison report; add `--video`
for MP4s. A one-second upright hold is a diagnostic, not the training success
criterion. Small seeded comparisons cannot establish equivalence for all starts.

```bash
# Explicit baseline / ablation switches, shared by sim_diffusion and deploy_diffusion:
.venv/bin/python g1_deploy/sim_diffusion.py --backend tensorrt --engine-mode standard
.venv/bin/python g1_deploy/sim_diffusion.py --backend tensorrt --engine-mode buffered
.venv/bin/python g1_deploy/sim_diffusion.py --backend tensorrt --engine-mode cuda_graph
# Experimental simulation only; reports differences from the FP32 export.
.venv/bin/python g1_deploy/sim_diffusion.py --planner-precision tf32 --start prone --video
```

TF32 uses a distinct engine cache identity. Simulation and the benchmark measure
its numerical differences even when the original export tolerance is exceeded.
Hardware deployment retains the original strict numerical check, so an engine
that fails that check is not promoted simply because it ran in MuJoCo. FP32 is
the recommended default. No sampler-step or replanning changes are made for speed.
`--backend onnx` remains the explicit ONNX Runtime CPU reference; it is not an
ONNX Runtime CUDA benchmark. CUDA graph execution requires the existing supported
TensorRT 10 environment; there is no silent runtime/precision fallback.

When the robot is available, sensor-only deployment records aligned robot-state
capture-to-proposal and arrival-to-proposal latency with both networks running:

```bash
.venv/bin/python g1_deploy/deploy_diffusion.py --model flow --duration-s 60
```

An explicitly actuated run additionally records capture-to-DDS-publish-return.
This endpoint excludes robot receipt and motor response. The JSONL control rows
include capture, arrival, proposal-ready and publish timestamps and clock
uncertainty. COMPLETE.json contains p50/p95/p99/max distributions. A shut-down
robot cannot provide those live measurements.

Implementation follows [NVIDIA CUDA graph guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html).
Precision changes need numerical and behavioral comparison; see
[NVIDIA accuracy guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html).
