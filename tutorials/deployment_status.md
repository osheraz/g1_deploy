# Diffusion deployment — current status

Goal: use the paired planner/action decoder through one inference core in
our simulator and `g1_deploy`. Teacher training and curation remain separate.
Operator commands: [deployment guide](deploy_diffusion.md).

## Implemented, 11 September

- One normal [`configs/diffusion.yaml`](../configs/diffusion.yaml) selects the
  bundle, backend exports, network and new D435I intrinsics. No separate hardware
  JSON or hand-written calibration receipt is required. Robot limits come from
  the existing G1-29 config; PD gains/default pose/action scale come from the
  paired decoder bundle, not SONIC. The old bundle/new lens mismatch is explicit.
- Startup aligns robot ticks from DDS arrivals and camera host time by clock
  probes, then requires paired captures. These are software clock estimates,
  not an independent measurement of constant sensor/transport latency.
- The live sensor path avoids the SDK's extra FIFO (which drops new packets
  while retaining stale ones), unnecessary whole-message copies and Torch
  dispatch for packet gravity. CRC checks and timestamp/command limits remain.
- `--actuate` waits for START, moves slowly from measured joints to default,
  checks settling, then waits for a fresh A press to start the policy. SELECT
  stops in damping. `--default-pose --actuate` only moves and holds, with A
  disabled. This new transition has synthetic tests, not a physical validation.
- The first pose-only launch correctly stopped before creating a motor
  publisher because `CheckMode` still reported `ai`. L2+R2 did not establish
  ownership release. `--release-controller` now explicitly invokes the existing
  SDK handover helper; without it the guard remains. Release failure stops
  startup. This new flag has only been tested with a simulated SDK, not invoked
  on the physical robot by the agent.
- New planner-depth rendering and the environment training default now use the
  same new-camera intrinsics. Existing saved checkpoints retain their calibration;
  previously rendered training depth must be re-rendered, not relabelled.
- Entrypoints are `deploy_diffusion.py` (current) and `deploy_vigor.py` (legacy).
  VIGOR2, its workers/configs and old guides are archived in `deprecated/`.
  Deployment docs live here; the redundant parent shell launcher was removed.
  This reorganization did not change current inference, threading or safety.
  Verification: 47 focused tests passed, one historical test module skipped;
  all three launchers' `--help` paths and local documentation links passed.
- Selected inference code from `origin/diffusion_policy@70e798a9`, not a full
  training-branch merge. The training checkout and shared bundle are unchanged.
- `scripts/eval_native_recovery.py` and `g1_deploy/deploy_diffusion.py` use the same
  `AsyncNativeRecovery`: measured history, heading, one planner process, code
  queue, requested-action feedback, clipping and reset/stale-reply handling.
  There is no second VIGOR2 tracker, old z_g interface or nominal-standing override.
- Simulation uses master's terrain, physical randomization and optimized Warp
  depth. Reference motion supplies the reset situation, **not policy inputs**.
  The paired PD contract is checked before preview.
- Viser reset clears policy/depth history, starts a new episode and discards old
  planner replies. The Camera Viewer displays actual cached planner depth without
  extra camera rendering. Browser interaction itself was not automated.
- Hardware uses the native Unitree HG adapter's explicit no-publisher dry run,
  RC stop and command watchdog. The old RobotInterface constructs a publisher at
  startup, so it is deliberately not wrapped around this runtime.
- USB and timestamped network depth use the same crop/ray sampling, metric units,
  masks and causal buffer. `g1_deploy/core/common/native_depth_server.py` supplies
  raw 30-Hz frames, capture timestamps and calibration metadata. It does not use
  the old temporal filtering, hole filling or color alignment.
- PyTorch, ONNX Runtime CPU and TensorRT 10 forwards are selectable in both
  launchers. Only the neural graphs are exported: decoder MLP, planner denoiser,
  planner depth CNN/GRU. Sampling/history/safety are not reimplemented.
- TensorRT is explicitly FP32 without TF32/FP16; checks compare it with the
  original PyTorch policy (including its saved TF32 policy where applicable).
  Hashes bind exports to checkpoint/config; static input shapes are checked.
  Per-GPU/runtime engine caches revalidate against saved inputs/outputs at startup.
- Gradient-based steering retains the PyTorch planner. It may use a TensorRT
  decoder. Guided flow remains blocked by the bundle's existing timing profile.

Both PyTorch and TensorRT completed 30-second live-sensor planner/decoder sessions
on the new G1-29. **Actuation is not qualified:** proposals fail command limits
from the current pose. The read-only viewer receives 640×480 depth at 30 Hz and
maps all 29 joints. See [camera commands](04_cameras.md) and
[migration record](08_robot_migration.md).
Generated exports and test logs remain local, outside source control.

## Models and depth contract

Read-only bundle:
`/nfs/turbo/coe-stellayu-extra/clhsieh/software/hv2_diffusion/experiments/fallrec_latent_planner_20260902/deployments/native_recovery_20260910b`.

B20k diffusion and E20k flow share **C10k**, the deterministic 64-dimensional
action decoder. It takes ten measured history frames (930 values) plus a code;
there is no PPO teacher or motion encoder at inference. Planner history is ten
state frames/five code frames, future 25 ticks at 50 Hz. Planner depth is three
64×48 captures; C10k itself does not consume depth.

**Pending upstream:** this bundle still specifies far/invalid = 0 with a mask.
Do not silently substitute far=max. The updated checkpoint/input contract must
be integrated into the shared builder, then verified for sim and hardware.
The integrated snapshot does not contain that change.

## Evidence

Real-sensor diffusion runs on Bluefin (30 seconds each):

| Backend | Control ticks / proposals | Rejected planner replies | Control work median / p95 |
|---|---|---|---|
| PyTorch | 1,500 / 1,498 | 0 | 3.96 / 12.77 ms |
| TensorRT | 1,500 / 1,499 | 0 | 3.71 / 11.70 ms |

These are inference results with the robot uncommanded, not recovery tests.
Every proposal failed the configured initial-jump or joint-limit check. For
TensorRT: 1,395 initial-jump rejections and 104 joint-limit rejections. Right hip
yaw targets differ from measured position by about 2.55 rad median; some
ankle-roll/waist targets exceed configured limits. Verify the physical starting
posture/joint readings and inspect proposals before a motor trial; do not loosen
limits just to obtain a passing dry run.

TensorRT camera/robot pairing error stayed below 0.50 ms under the estimated
clock maps; median capture-to-packet latency was 57.6 ms. Neither independently
establishes absolute exposure-to-IMU synchronization. Control work had occasional
>20 ms outliers (TensorRT maximum 25.1 ms), so these short no-action sessions are
not a hard real-time motor-loop qualification. New sessions also report
`publication_late_ticks` explicitly.

Logs under `logs/native_recovery/`:

- `robot_diffusion_pytorch_1789165662858266798/`
- `robot_diffusion_tensorrt_1789165600080104974/`

Each has resolved settings, session identity, telemetry and `COMPLETE.json`.
Earlier failed runs are retained as diagnostics, not counted as passes.

A subsequent TensorRT repeat (`robot_diffusion_tensorrt_1789165892419738394/`)
stopped at tick 875 on the control-clock deadline, despite zero rejected plans.
It reported two late publication opportunities before stopping. The two complete
sessions above therefore do **not** establish repeatable timing readiness.

The operator's latest TensorRT check
(`robot_diffusion_tensorrt_1789166272384355094/`) ran about 13.8 seconds before
Ctrl+C: 688 proposals, zero rejected planner replies and one late publication
opportunity. It was an operator stop, not a completed long-duration check.
New launcher sessions record Ctrl+C as `STOPPED.json`, not `FAILED.json`.

Current startup/hardware/bundle/backend/camera and handover suite: **84 passed**;
`logs/native_recovery/deploy_startup_tests.log`. This includes duplicate ticks,
clock alignment, packet gravity, remote START edges/SELECT stop, and stale
camera-pair rejection, plus smooth default-pose targets, blocked-move rejection,
and the separate A/pose-only gates. Tests use synthetic motor transport, not
physical commands.

Pre-push motion-diffusion suite plus camera and handover tests: **334 passed,
one skipped** (`logs/native_recovery/deploy_push_tests.log`). The handover tests
also verify explicit opt-in, release-before-robot ordering, and stopping on a
failed release. None calls the physical robot's release API.

Exports: `logs/native_recovery/exports_20260911/`:
`controller_checked/`, `diffusion/`, `flow/`. Each contains ONNX, validation
inputs/outputs, hashes, numerical receipts and this GPU's TensorRT engine.

| TensorRT check | Result |
|---|---|
| Full C10k command, 27 cases including reconstructed recorded preview inputs | max requested-action difference 9.54e-6 |
| Full diffusion sampler, identical history/depth/noise | max S67 difference 1.67e-5; code difference 5.72e-6 |
| Full flow sampler, identical history/depth/noise | max S67 difference 1.20e-5; code difference 2.27e-6 |
| Diffusion simulator/reset integration | 300 physics steps, 298 commands, zero rejected replies |
| Flow simulator integration | 150 physics steps, 148 commands, zero rejected replies |
| PyTorch yaw steering + TensorRT decoder | 100 physics steps, 96 commands, zero rejected replies |

The numerical cases are not a robot behavior evaluation. Raw decoder errors on
out-of-distribution synthetic inputs can be larger; the full-policy check above
compares the **clipped requested commands**, not only relative raw-output error.
Full-sampler checks cover reset/missing history and populated history.

Warm neural-forward medians on Bluefin's 4090: decoder **0.214 ms**, one denoiser
call **0.78 ms**, depth encoder **0.095 ms**. These exclude sensing, history,
multiple denoising calls, IPC and command publication. Full sampler checks also
save timings, but include first-use kernels; they are not a latency qualification.

Receipts: `export.json`, `policy_parity_tensorrt.json` in each export directory;
sim traces: `logs/native_recovery/trt_{diffusion,flow}_sim_20260911/`.
Focused combined suite: **358 passed, one skipped**; log:
`logs/native_recovery/integration_suite_20260911.log`.
Local runtime: Torch 2.7.0+cu128, TensorRT 10.10.0.31, IsaacSim 5, 4090.
Optional bindings/export packages were added without upgrading Torch or Isaac.

Earlier PyTorch previews completed 750 physics steps each for diffusion/flow.
A strict wall-clock diffusion preview failed its 50-Hz deadline while sharing
the GPU with simulation. The new simulation-clock checks do not overturn that
result or certify hardware latency. Engine validation on another target does
not certify the full deployment loop there.
Compiled preview step medians were 34–36 ms for diffusion and 59 ms for flow;
these include the simulator and sensor work, not just inference. They are not
50-Hz wall-clock passes. The guided mixed-backend trace is saved at
`logs/native_recovery/trt_guided_sim_20260911/`.

## Remaining before a robot trial

1. Resolve the proposed-target/initial-posture violations above. Longer no-action
   checks must also establish timing headroom. Command limits remain enforced.
2. Qualify controlled actuation separately with the supported robot and remote.
   No physical recovery/steering/safety result is claimed.
3. Broader simulation evaluation still needs the recovery/drift/smoothness/contact
   and steering callbacks wired to a fixed panel; no success result is claimed.

## Shared runtime access

Six optional packages in the producer's read-only GL overlay had owner-only files.
Byte-identical readable copies (701 files) are available to `clhsieh` at
`/nfs/turbo/coe-stellayu-extra/osher/projects/hv2/runtime_access/isaac6_optional_20260911/`.
Its README gives the container import setup; `ACCESS_RECEIPT.json` records the
comparisons. Read/traversal ACLs only; IsaacSim/IsaacLab/Torch/overlays unchanged.
