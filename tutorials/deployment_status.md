# Diffusion deployment — current status

Goal: use the paired planner/action decoder through one inference core in
our simulator and `g1_deploy`. Teacher training and curation remain separate.
Operator commands: [deployment guide](deploy_diffusion.md).

## Original-training implementation audit, 12 September

The latest default move stopped on an added 0.35 rad tracking-error gate, before
any policy command. Removed that gate, the initial 0.2 rad/s stillness gate, the
move-specific 2 rad/s speed gate, the 0.12 rad/two-second settling gate, and the
separate startup-hold tracking gate. The existing two-second ramp now holds its
target and waits for A without requiring measured joints to match spring targets.
Operator stops and the normal measured-speed check remain.

Replayed actual startup inputs with C10k weights through the actor implementation
at the recorded training commit `b035d36bedbf5971c476351ffa6abd9f3ddc4c11`.
The training actor and deployment actor agree exactly, including raw maxima of
37,990.83 and 11,464.05 in the failed sessions. All 29 outputs reach the action
clip. Changing only the failed input's joint-position history to the successful
run's history reduces its largest action to 2.90. This identifies a learned
pose/code failure, not an actor input-order or TensorRT discrepancy. Default
pose is not a universal cure: the latest recorded code still gives 430.25 with
default joint positions and zero velocities/gyro. These are counterfactual
decoder probes, not physical deployment tests.

The saturation block remains because these recorded failures would command
approximately 55 rad offsets after clipping/scaling. No new robot trial is needed
to reproduce them. Full evidence, crossed-input experiments, original-source
replay script and recorded fixtures are in
`logs/native_recovery/implementation_audit_20260912/` in the parent repository.
This supersedes the earlier inference that bad native codes alone explain all
saturation: the new failures depend strongly on the measured joint pose.

## Startup repair, 12 September

Follow-up after runs `1789231138051950622`, `1789231169466261666`, and
`1789231194146162447`: the first two recorded all 29 requested actions at +/-100
on the very first preview in both TensorRT and PyTorch. This is a real rejected
proposal, distinct from the earlier delayed-preview bug. Preflight now remains
idle and retries instead of crashing, allowing START to request default pose
and SELECT to stop. Actual saturated motor commands remain rejected. Clock
alignment now retries a rolling one-second window after excessive startup
delivery jitter; it does not accept an uncertain mapping or require an immediate
process restart. Persistent failure still ends at the existing startup timeout.

The normal TensorRT launch now checks live first plans before A, releases the
built-in controller only on A/START, and holds the current joints while the
first policy plan arrives. START remains optional and its default-pose hold
rechecks inference before A. The release itself still requires physical support;
a current-joint PD hold is not a balance controller.

Preflight previously reset history while an old planner request was in flight.
The rejected reply could then leave it waiting for the 22-tick replan interval,
advancing a recovery against a stationary robot. It now drains that reply and
restarts from initial measured history. Actual bootstrap retries immediately
when the worker becomes free. Preview sampling preserves the live RNG state.

Offline replay of checks 1616/1617 from
`robot_diffusion_tensorrt_1789228951186081786` isolates the saturated output to
the supplied code: good history/good code gives raw action max 2.8302; good
history/bad code gives 4356.9063; bad history/good code gives 2.8307; both bad
gives 4288.9004. The outlier has ten varying measured rows with zero issued
requests, versus two varying rows in the preceding normal startup checks.
Continuing an unexecuted recovery is not a valid closed-loop decoder test.

The command path follows
[SONIC's direct PD mapping and separate writer](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp).
There is no target-angle/jump rejection or measured URDF-position shutdown.
Positions remain logged. The 35 rad/s speed check applies in both directions;
finite/saturation checks, operator stops, fresh sensor data, transport faults,
and the 60 ms command watchdog remain. An occasional >20 ms completion is
counted rather than immediately stopped: inputs must still be under 40 ms old,
publication no later than 40 ms after its scheduled tick, and the next step
less than one whole period late. Histories and code cursors never skip ticks.

Evidence from this repair:

- `resumed_standing_20260912/`: two eight-second simulation episodes, 800 physics
  steps, zero rejected plans; maximum requested action 15.6563 and measured
  speed 16.6319 rad/s, with no saturation. These are integration traces, not a
  standing/recovery success score. URDF-position excursions reached 0.0679 rad;
  the removed measured-position rule would have interrupted these traces.
- `resumed_preflight_connected_20260912/`: 60 seconds with live robot and camera,
  2,386 first-plan proposals, max requested action 2.6886, no saturation, no motor
  output. Three timing misses were restarted; median/p95 control work was
  2.24/6.94 ms and maximum 33.01 ms. This motivated bounded overrun handling.
- `final_preflight_20260912/`: second 60-second live check, 2,398 first-plan
  proposals, max requested action 2.6773, no saturation, no motor output. One
  strict preview timing miss; median/p95/max work 2.42/7.07/22.30 ms. Preflight
  still counts >20 ms as a miss and restarts; actual command publication accepts
  the bounded overrun described above.
- `resumed_full_suite_20260912.log`: 347 passed, one skipped. Includes synthetic
  operator handover, current-pose hold through the first plan, failed release
  with zero publication, late preview generations, recoverable 25 ms inference
  overrun, and rejected 45 ms stale inference. No physical actuation was used.

Physical policy behavior still needs the operator's supported test. No robot
recovery or hard real-time guarantee is claimed by these checks.

## Earlier implementation and validation

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
- `--actuate` starts idle without motor commands. A starts the policy from the
  current pose; A again stops in damping and exits. Before the first start only,
  START optionally moves to default using VIGOR's two-second
  linear blend, checks settling and holds for A. During the move A cancels;
  during policy execution START is ignored. SELECT cancels/stops in any stage.
  Pose-only mode also waits for START and never starts a policy. Stops are latched by the DDS
  callback and honored by the motor watchdog even if inference is busy.
  The operator reported that the two-second move worked on the new robot;
  this does not validate the full policy or other starting poses.
- Console messages show readable startup stages and one-second health summaries.
  Complete structured telemetry remains in the per-session journal.
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

Both PyTorch and TensorRT completed live-sensor planner/decoder sessions on the
new G1-29. **Full-policy actuation is not qualified:** the 12 September motor
trial stopped on its first proposal with zero commands issued, despite an
upright, near-default robot. The read-only viewer receives 640×480 depth at 30 Hz and
maps all 29 joints. See [camera commands](04_cameras.md) and
[migration record](08_robot_migration.md).
Generated exports and test logs remain local, outside source control.

### Earlier command-limit diagnosis, before the startup repair

- Failed session: `logs/native_recovery/robot_diffusion_tensorrt_1789226546958594952`.
  The original rejected proposal was not recorded because validation preceded
  control journaling. The path now records rejected requests/targets and measured
  joints, and names the offending joints in the error before any publication.
- Fresh five-second sensor-only check:
  `logs/native_recovery/robot_diffusion_tensorrt_1789226746607410827`.
  First proposal: right ankle pitch target +0.643 rad, allowed upper +0.5236,
  measured -0.417. This is evidence from the new check, not a reconstruction of
  the missing original command. These limits agree with the
  [official G1 URDF](https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/robots/g1_description/g1_29dof.urdf).
- Native joint order, default positions and action scales agree with the bound
  decoder contract. Offline replay of all 249 recorded input histories/codes
  through PyTorch and TensorRT reproduces the same problem. First command
  maximum normalized-action difference: 1.91e-6. Some later sensor-only outputs
  reach the checkpoint's +/-100 action clip in both backends; this is not a
  small joint-limit rounding issue or a TensorRT-only conversion problem.
- At the operator's request, hardware now follows
  [SONIC's direct position-target path](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp):
  `target = default + action_scale * action`. The added target-angle and
  first/subsequent target-jump rejection have been removed, not replaced with
  clipping. Gains, action scale, model and requested-action feedback are unchanged.
  Simulation clips PD torque; matching SONIC's target mapping does not establish
  identical hardware torque limiting.
- At this stage, measured-position and velocity violations latched a fault on LowState receipt;
  the writer switches to damping independently of inference. Finite/action-bound
  checks, watchdog and operator stops remain. These checks detect violations;
  they do not guarantee prevention of overshoot. Default-pose tracking/speed
  checks remain separate. No motor trial was run for this change.
- Work remaining at that stage was to explain excessive outputs on these measured
  histories/codes and validate the resulting command path in simulation before
  another motor trial. Removing target rejection does not solve that issue.
  Sensor-only trials have no executed-action feedback and are not closed-loop
  behavior tests.

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

These are historical inference results with the robot uncommanded, not recovery
tests; they predate the direct-target change above. Every proposal failed the
then-configured initial-jump or joint-limit check. For
TensorRT: 1,395 initial-jump rejections and 104 joint-limit rejections. Right hip
yaw targets differ from measured position by about 2.55 rad median; some
ankle-roll/waist targets exceed configured limits. Verify the physical starting
posture/joint readings and inspect proposals before a motor trial. The removal
of these target guards does not retrospectively qualify the outputs.

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

Current motion-diffusion suite plus camera and handover tests: **359 passed,
one skipped** (`logs/native_recovery/sonic_direct_targets_tests.log`). This
includes duplicate ticks, clocks, gravity, stale camera-pair rejection, the
two-second default-pose path, idle startup without commands, optional
START movement, SELECT cancel/stop, blocked-move rejection, A start/stop, held-button and
double-press handling, readable console with complete journals, and watchdog
damping without polling the policy loop. It also checks unchanged SONIC targets
(no target clipping or jump limit), measured-joint faults, and damping on sensor
receipt without another policy tick. Operator stops
write `STOPPED.json`, not `FAILED.json`. Handover tests verify opt-in,
release-before-robot ordering and stopping on failed release. All motor transport
in these tests is synthetic; no physical commands or release API calls occur.

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

## Remaining validation

1. Qualify controlled actuation separately with the supported robot and remote.
   No physical recovery/steering/safety result is claimed.
2. Broader simulation evaluation still needs the recovery/drift/smoothness/contact
   and steering callbacks wired to a fixed panel; no success result is claimed.

## Shared runtime access

Six optional packages in the producer's read-only GL overlay had owner-only files.
Byte-identical readable copies (701 files) are available to `clhsieh` at
`/nfs/turbo/coe-stellayu-extra/osher/projects/hv2/runtime_access/isaac6_optional_20260911/`.
Its README gives the container import setup; `ACCESS_RECEIPT.json` records the
comparisons. Read/traversal ACLs only; IsaacSim/IsaacLab/Torch/overlays unchanged.
