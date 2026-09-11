# Diffusion deployment: simulation and G1

The current pipeline is **diffusion/flow planner → paired action decoder → joint
commands**. Neither the privileged PPO teacher nor a reference-motion encoder
runs at inference.

`g1_deploy/deploy_diffusion.py` and the simulation preview use one shared core
in `motion_diffusion/behavioral/hardware/`. The planner has its own process;
measured history, timestamped sensing, action-code queue, stale-reply rejection
and command watchdog are preserved. Archived VIGOR2 is not imported.

## Start here

- Robot/camera preview only: [camera and Viser commands](04_cameras.md).
- Installation: [environment](02_install.md).
- Current validation and pending work: [status](deployment_status.md).
- Model selection and physical settings: [configuration](03_config.md).

Commands below run from `/home/osher/hv2` with `source .venv/bin/activate`.

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

## Simulation preview

```bash
python scripts/eval_native_recovery.py --model diffusion --viser
```

Use `--model flow` for flow. `--motion-id 25` chooses a different reset situation;
`--seconds 15` sets preview duration. The current teacher preset supplies terrain
and reset references. **References supply initialization only, not policy inputs.**
The bundle predates the current terrain release, so this is not its original
validation panel.

The viewer's **Camera Viewer → Planner depth (Warp)** shows the newest arrived
64×48 planner frame, not a second camera render. The planner also uses older
captures. **Reset THIS env** clears physical/policy/depth history together and
discards old planner replies. Normal physical randomization stays active;
encoder/IMU observations are ideal simulated measurements in this preview.

Viser repeats until Ctrl+C. Headless runs default to one episode; use
`--episodes 2` for a bounded repeat. The default simulation clock permits slower
physics; `--realtime` instead enforces deployment deadlines and stops when missed.
Neither a preview nor simulation-clock timing qualifies robot latency.

Diffusion's `--yaw-deg 5` enables experimental small-angle guidance (scale 0.05,
gradient cap 0.1). Guided flow is blocked by the bundle's timing gate.
Persistent xyz/height goals require an explicitly connected odometry source.

Outputs under `logs/native_recovery/` include resolved config, summaries and
per-episode `trajectory.npz` / `plans.npz`. States, requested/delayed actions,
codes and contacts are captured before autoreset; plans retain origin tick/frame.
Full matched-panel quality scoring and predicted-plan overlays remain pending.

## Real robot

One config: [`configs/diffusion.yaml`](../configs/diffusion.yaml). No hand-written
hardware JSON or clock/calibration receipts. It inherits the existing 29-joint
mapping and limits; gains, default pose and action scale come from the paired
decoder, not the inherited SONIC policy. Start the [camera stream/tunnel](04_cameras.md)
first. Both depth port **15556** and clock port **15558** must be forwarded.

Run these separately. Start with the no-action checks:

```bash
# PyTorch, real sensors, no actions (30 seconds).
python g1_deploy/deploy_diffusion.py

# TensorRT, real sensors, no actions.
python g1_deploy/deploy_diffusion.py --backend tensorrt

```

Next, with the robot hoisted, limbs clear and the built-in controller in
low-level/debug mode, test **only the default pose**:

```bash
python g1_deploy/deploy_diffusion.py --default-pose --actuate --release-controller
```

`--release-controller` explicitly releases the built-in motion service at
startup, before any motor publisher exists. **The hoist must support the robot
before running it.** L2+R2 may leave the service reporting `ai`; the launcher
uses the existing SDK `ReleaseMode` helper and verifies that ownership is free.
Without this flag, an active service still blocks motor startup. This follows
[Unitree's low-level handover](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/g1/low_level/g1_low_level_example.py).

This moves motors, but never starts the planner. Press **START** to move from
the measured joint positions to the decoder's default pose, with a smooth
target-speed limit of 0.15 rad/s. The duration depends on the starting pose and
is printed. It then holds; **A does nothing**, and **SELECT/Ctrl+C stops in
damping**. `--duration-s` does not end this pose-only hold. This is not a balance
controller: keep the hoist supporting the robot. Do not run another motor
publisher alongside it.

After the pose and command/timing checks pass, the policy-trial command is:

```bash
python g1_deploy/deploy_diffusion.py --backend tensorrt --actuate
```

Here **START moves to default**, then a fresh **A** press starts the policy.
Stopping the pose-only process and starting this command repeats the default
transition; the two processes must not run together.

Use `--duration-s 60` for longer sessions or `--model flow` for flow. Session
logs go under `logs/native_recovery/robot_*`. `--config` selects another normal
YAML. Models and exports are configured there; no long path arguments are needed.

Startup warms models, aligns clocks, requires three paired depth captures, and
reports readiness. Camera host time is aligned by network round trips. Robot
ticks use the fastest observed DDS arrival over one second: this estimates
alignment but does **not** measure constant sensor/internal transport latency.
Clock jumps, stale data and restarted streams stop the session.

For a motor trial, prepare the robot in supported low-level/debug mode, keep
people clear, and have the remote ready. Vendor-controller release requires the
explicit `--release-controller` flag; it is never performed by a no-action run.
**SELECT stops in damping.** Target limits, initial target jump,
subsequent target rate, measured joint speed and the watchdog remain enforced.
No-action runs report command-limit violations; inspect these before actuation.

The replacement camera intrinsics are explicit in the config. The older bundle
is knowingly tested with `allow_intrinsics_mismatch: true`; the difference is
logged and checkpoint bytes stay unchanged. New training uses the new lens.

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

- `pytorch`: default; required for gradient-based planner guidance.
- `onnx`: ONNX Runtime CPU numerical reference.
- `tensorrt`: explicit FP32 CUDA decoder and unguided planner graphs.
  No silent precision change or fallback. A TensorRT decoder can accompany
  a guided PyTorch planner.

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
