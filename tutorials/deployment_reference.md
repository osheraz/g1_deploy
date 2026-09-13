# Deployment reference

For the commands to run now, use [the operator guide](deploy_diffusion.md).
This page is for model/export maintenance and runtime details.
Run commands from `/home/osher/hv2` with `source .venv/bin/activate`.

## Model bundle

Default simulation bundle:

```text
/nfs/turbo/coe-stellayu-extra/clhsieh/software/hv2_diffusion/experiments/fallrec_latent_planner_20260902/deployments/native_recovery_20260910b
```

B20k diffusion and E20k flow are paired with the **C10k action decoder**:
a deterministic 64-dimensional code plus ten frames of measured robot history
produces 29 joint commands. C10k is not a PPO teacher.

Validate both model pairs without opening a simulator, camera or robot channel:

```bash
python -m motion_diffusion.behavioral.hardware.bundle check --sample \
  --bundle /nfs/turbo/coe-stellayu-extra/clhsieh/software/hv2_diffusion/experiments/fallrec_latent_planner_20260902/deployments/native_recovery_20260910b
```

For a local copy, preserve the manifest and materialize symlink targets.
Startup checks hashes, dimensions, joint order and the paired PD contract.
Do not substitute another decoder merely because its latent dimension matches.

## Control and startup behavior

The paired decoder supplies PD gains, default pose and action scale. Requested
joint targets use `default + action_scale * action`; they are not increments from
the current pose. No added target-angle clipping or target-jump rejection is used.

Startup reports large joint deviations and blocks actuation only for extreme
measured limit excursions (more than 20 degrees outside configured limits), or
existing sensor/output faults. The warning threshold is 90 degrees from default
or within 5 degrees of a configured limit. Deeply folded starts remain allowed.
The optional START move ramps to default over two seconds and holds; it has no
separate tracking-error or settling gate. It is a joint hold, not a balance policy.

Finite/saturated output checks, the 35 rad/s measured-speed check, sensor
freshness, transport faults and the independent command watchdog remain active.
A rejected startup preview retries on new measurements and leaves START/SELECT
available. Recorded C10k startup inputs also saturated in the original training
actor; bypassing output rejection does not repair that failure. Local evidence:
`logs/native_recovery/implementation_audit_20260912/`.

The planner runs in a separate process. The control loop maintains measured
history and consumes valid planner replies; an independent writer publishes PD
commands. See [compute architecture and measured costs](../../notes/deployment_compute.md).
Clock-age checks use the current estimated mappings. They do not establish an
independent LiDAR/robot-to-host offset bound for steering.

## Shared depth processing

The Jetson publishes raw 640×480 depth at 30 Hz, capture metadata and lossless
PNG. No temporal filtering, hole filling, or color alignment. USB/network frames
use the same calibrated crop/ray sampling, metric range/mask and causal history
as the optimized Warp camera in simulation.

The current bundle requires **far/invalid = 0 plus a mask**. There is no
deployment-only `far=max` override. A changed encoding requires its matching
checkpoint contract and validation in both sim and hardware.
The read-only raw viewer is not this processed model input.

## ONNX and TensorRT

Only neural forwards change backend; history, preprocessing, sampling, code
queue, clipping, resets and safety stay in the shared runtime.

- `pytorch`: reference; required for gradient-based planner guidance.
- `onnx`: ONNX Runtime CPU numerical reference.
- `tensorrt`: default FP32 CUDA decoder and unguided planner, with reusable
  buffers and CUDA graph replay.
  No silent precision change or fallback. A TensorRT decoder can accompany
  a guided PyTorch planner.

See [inference latency](inference_latency.md) for benchmark commands, mode
ablations, experimental TF32 simulation and live latency telemetry.

Existing Bluefin exports live at
`logs/native_recovery/exports_20260911/{controller_checked,diffusion,flow}/`:

A byte-identical shared copy is available for another deployment checkout.
Copy it locally to match the default YAML paths (the large files are not in Git):

```bash
mkdir -p logs/native_recovery/exports_20260911
rsync -a /nfs/turbo/coe-stellayu-extra/osher/projects/hv2/deployment_exports/native_recovery_20260911/ \
  logs/native_recovery/exports_20260911/
```

The existing engine caches are for Bluefin's GPU/runtime. Another target uses
the ONNX graphs and validation inputs to build and check its own cache.

Simulation preview with these exports:

```bash
python scripts/eval_native_recovery.py --model diffusion --viser \
  --controller-backend tensorrt \
  --controller-export logs/native_recovery/exports_20260911/controller_checked \
  --planner-backend tensorrt \
  --planner-export logs/native_recovery/exports_20260911/diffusion
```

For hardware use `--backend tensorrt`; export directories come from
`configs/diffusion.yaml`. Simulation retains its explicit per-component flags.

For a new bundle or target GPU:

```bash
python -m motion_diffusion.behavioral.hardware.export \
  --bundle /path/to/bundle --component controller \
  --output /path/to/new/controller --tensorrt

python -m motion_diffusion.behavioral.hardware.export \
  --bundle /path/to/bundle --model diffusion --component planner \
  --output /path/to/new/diffusion --tensorrt
```

Add `--trace /path/to/trajectory.npz` for recorded controller inputs.
`--verify-policy` compares the full command/sampling path without a robot.
Omit `--tensorrt` for ONNX-only export.

Keep `.onnx`, `.validation.npz` and `export.json` together. Engine caches bind
the graph, TensorRT, CUDA, GPU and precision; another target rebuilds and
validates them before sensor connection. Numerical agreement does not qualify
full-loop timing or physical behavior. See [recorded checks](deployment_status.md).
