# Diffusion deployment: simulation and G1

For the flat-ground MuJoCo comparison with the supported full-horizon diffusion
and replanning flow recipes, see [MuJoCo runner](sim_diffusion.md).

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

Run these separately on Bluefin, with the camera terminal left running:

```bash
cd /home/osher/hv2
source .venv/bin/activate
```

Optional sensor-only startup check; no button press is needed:

```bash
python g1_deploy/deploy_diffusion.py --backend tensorrt --preflight --duration-s 60
```

This repeatedly checks the first plan on current measurements and never creates
a motor publisher. Delayed replies are drained before fresh reset history is
created. Without `--preflight`, the older continuous sensor-only diagnostic
advances recovery while the robot does not execute it; later outputs from that
diagnostic are not a closed-loop behavior test. Motor mode runs live startup
checks automatically, so a separate preflight invocation is optional.

Test TensorRT policy control with the robot supported by the hoist and limbs
clear. Have the remote ready, and run **only one motor process**:

```bash
python g1_deploy/deploy_diffusion.py --backend tensorrt --actuate --release-controller --duration-s 60
```

`--release-controller` releases the built-in motion service only after an
operator A/START request, before any motor publisher exists. Loading and idle
waiting leave ownership intact. **The hoist must support the robot before
takeover.** L2+R2 may leave the service reporting `ai`; the launcher
uses the existing SDK `ReleaseMode` helper and verifies that ownership is free.
Without this flag, an active service still blocks motor startup. This follows
[Unitree's low-level handover](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/g1/low_level/g1_low_level_example.py).

After `Ready`, our controller is idle: **no automatic default-pose movement**.
Startup prints joints more than 90 degrees from the decoder default or within
5 degrees of a configured joint limit, including measured angle, deviation,
motor index and limits. These are warnings: deeply folded ground starts remain
allowed. Motor startup is blocked only for a measured joint more than 20 degrees
beyond its configured limit (or existing invalid-sensor faults). The pose is
checked on sensor connection and again before/after controller release. This
also applies to the optional default-pose mode; it adds no running-policy
position gate and does not change gains or targets. Sensor-only runs report
extreme poses without blocking on this pose check.
Wait for `Live inference ready` before A. After takeover, current-pose PD commands
cover the first planner reply; this is a joint hold, not a balance controller.
START's default-pose hold also rechecks live inference before A starts policy.
A rejected preview now leaves the launcher open and START/SELECT usable. It
reports the reason once while retrying fresh measurements; A remains unavailable
until the preview is valid. A noisy initial robot-clock window is retried
automatically within the existing startup timeout instead of immediately exiting.

| State | Buttons | Result |
| --- | --- | --- |
| Initial idle | A | Start policy directly from the current pose |
| Initial idle | START | Optional two-second move to default, then hold |
| Moving to default | SELECT or A | Cancel, damp and exit |
| Holding default | A | Start policy |
| Policy running | A again | Stop, damp and exit |
| Any stage | SELECT | Stop and exit; damp if commands have begun |
| Policy running | START | Ignored; cannot move to default |

Release buttons between presses. Holding A does not immediately stop the policy.
Buttons held during startup are ignored until released and pressed again. The
default-pose option is available only before the first policy start. Ctrl+C
also stops. Once motor commands have begun, stopping sends damping, not a
standing hold. Restart the command for another trial; it starts idle again.

The 60-second limit begins when the policy starts, not while waiting for buttons.
First verify a short A-start/A-stop trial while supported. Subsequent trials can
start from other supported poses without moving to default. Policy commands use
the SONIC mapping: `target = default + action_scale * action`, with no added
target-angle clipping or target-jump rejection. This is a position-PD setpoint,
not a velocity or an increment from the current joint angle.

Optional pose-only check (no policy or camera required):

```bash
python g1_deploy/deploy_diffusion.py --default-pose --actuate --release-controller
```

This also starts idle. START requests the move, SELECT or A cancels/stops, and A
never starts a policy in this mode. `--duration-s` does not end the hold. This is
a joint-position hold, not a balance controller.

Use `--duration-s 60` for longer sessions or `--model flow` for flow. Session
logs go under `logs/native_recovery/robot_*`. `--config` selects another normal
YAML. Models and exports are configured there; no long path arguments are needed.
Console messages are short; complete structured records stay in
`telemetry.jsonl.gz`, `resolved.json` and `session.json` within each run directory.

Startup warms models, aligns clocks, requires three paired depth captures, and
reports readiness. Camera host time is aligned by network round trips. Robot
ticks use the fastest observed DDS arrival over one second: this estimates
alignment but does **not** measure constant sensor/internal transport latency.
Clock jumps, stale data and restarted streams stop the session.

For a motor trial, prepare the robot in supported low-level/debug mode, keep
people clear, and have the remote ready. Vendor-controller release requires the
explicit `--release-controller` flag; it is never performed by a no-action run.
**A or SELECT during policy execution, or Ctrl+C, stops in damping.** Direct
targets and measured-speed monitoring follow
[SONIC's command path](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp).
Measured URDF-position excursions are logged without stopping policy. The 35 rad/s
speed check applies in both directions. Nonfinite/saturated outputs, stale sensor
data, failed transport, and the 60 ms command watchdog still stop execution.
These checks do not guarantee physical joint limits cannot be crossed.

An occasional control completion beyond 20 ms is counted, not an immediate stop.
The independent writer retains the previous target. The next control step must
start less than one period late, the proposed command's observation must remain
within its 40 ms freshness bound, and publication must occur within 40 ms of its
scheduled tick. No history/code ticks are skipped or fabricated. This accepts
bounded timing jitter; it does not establish hard real-time execution.
The optional two-second default-pose move holds its target after the ramp,
without separate stillness, 2 rad/s speed, tracking-error, or settling gates.
The normal 35 rad/s measured-speed check and operator stops still apply.
The saturation block is an additional check beyond SONIC: recorded startup
inputs reproduce raw actions above 10,000 in the original training actor.
See the September 12 implementation audit in the status page; bypassing that
failure would issue clipped +/-100 requests, not fix the decoder.
Current evidence is in [status](deployment_status.md).

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
