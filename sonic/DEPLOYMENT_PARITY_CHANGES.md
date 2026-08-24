# SONIC deployment refactor and 23-DoF parity ledger

> Historical implementation ledger. For the current three-model × two-input ×
> two-embodiment architecture, direct-target behavior, accumulated fixes,
> validation status, and physical conclusions, use the canonical
> [`SONIC deployment learning guide`](../docs/learning/sonic_deployment.md).
> The latest captured-IDLE physical analysis, 199-recording training-free MuJoCo
> adaptation campaign, and model-training roadmap are in
> [`../docs/analysis/sonic_23dof_slow_walk_adaptation.md`](../docs/analysis/sonic_23dof_slow_walk_adaptation.md).
> In particular, the current official-parity C++ path does not clamp joint
> position, command rate, or finite tracking error; tracking excursions are
> diagnostics-only. Invalid/stale state, excessive measured velocity,
> temperature/fault, and pelvis tilt still latch damping on hardware.

Date: 2026-08-12

This is the canonical change record for `g1_deploy`. It describes the Python
runtime, its relationship to NVIDIA's released 29-DoF SONIC implementation,
and the remaining gates for a 23-DoF G1. It is not a safety certification.
First robot runs still require physical support, a tether, a reachable
independent damping/deadman path, and conservative current/temperature limits.

## Reviewed baseline

- Official reference: `NVlabs/GR00T-WholeBodyControl` at
  `1983e88888217f6c69283cf3a9d1af01e87f07af`.
- Local branch before this refactor:
  `agent/sonic-23dof-parity-planner` at `1ab9aa6`.
- The official Python sources, production C++ deploy, observation YAML files,
  planner wrapper, model card, PICO manager, packed ZMQ protocol, and Unitree
  CRC code were reviewed together. C++ is the production reference where it
  differs from unused or training-only Python paths.

The released planner, encoder, and decoder remain 29-DoF. The physical
23-DoF adapter does not change their tensor dimensions or pretend that the
missing joints exist.

## Runtime structure

The refactor separates three independent choices:

1. `sonic/model/model_profile.py` defines an atomic released model family:
   `default`, `low_latency`, or `sonic_v1_1`.
2. `sonic/model/robot_profile.py` defines the physical embodiment:
   `g1_23dof` or `g1_29dof`.
3. `sonic/model/inference_backend.py` selects direct TensorRT or ONNX Runtime.

`sonic/model/robot_g1.py` now contains only the official 29-joint order. All
29-to-23 projection, absent-joint masking, and physical motor selection live in
`robot_profile.py`. This keeps the released model math reviewable and makes the
29/23 switch explicit in YAML:

```yaml
sonic_model_variant: low_latency
sonic_robot_profile: g1_23dof   # or g1_29dof
sonic_inference_backend: tensorrt
sonic_planner_backend: tensorrt
sonic_tensorrt_fp16: false
sonic_tensorrt_tf32: false
```

`configs/g1_29dof_sonic.yaml` inherits the common SONIC configuration and only
overrides embodiment-dependent arrays. `SonicRunner` validates motor order,
fixed-joint order, dimensions, defaults, limits, finite values, and all ONNX
input/output shapes at startup.

## Selected checkpoint: official Low-Latency SONIC

The default hardware configuration now uses the official Low-Latency row from
the SONIC model card as an atomic set:

| file | contract | SHA-256 |
| --- | --- | --- |
| `sonic/low_latency/model_encoder.onnx` | `(1,1247) -> (1,64)` | `60be43157f57d812f38bdbb740a5de5d5d070e8840d9edc16f02a91a6d06255b` |
| `sonic/low_latency/model_decoder.onnx` | `(1,994) -> (1,29)` | `c4ac2e74045e7cbfb568f15e6bf47ea7ce023df7a94322af50be223e0a628bab` |
| `sonic/low_latency/observation_config.yaml` | G1/teleop step 1; four SMPL frames | `582b9a273a3d69fbf49ae59b39295a3be2b4a295e195ef4cf674b5e2571c90ab` |

The ONNX files are Git LFS objects. Never swap only one file or reuse an
observation layout from another model family. Startup shape checks catch many,
but not all, mixed-checkpoint errors.

Low-Latency shortens the command horizon. It does not remove Ethernet, DDS,
host scheduling, Python, or planner latency. Those still need live measurement.

## Encoder and recurrent-state parity

`sonic/model/obs.py` implements all three released flat layouts:

- Default: 1762 encoder values.
- Low-Latency: 1247 encoder values.
- SONIC v1.1: 1751 encoder values with heading-normalized orientation.

The encoder buffer is zero-initialized and only terms required by the selected
encoder are filled, matching the official C++ gatherer. A missing required
term fails closed; it no longer falls through to another encoder.

For the 23-DoF profile:

- physical positions and velocities fill their 23 correct IsaacLab slots;
- nonexistent waist roll/pitch and wrist pitch/yaw proprioception are zero;
- native 29-D task references remain unchanged;
- native 29-D raw decoder output remains unchanged in `last_action`;
- only physical command application gathers the 23 present actions.

Native 29-DoF execution still keeps all 29 outputs exactly as the official
controller does. NVIDIA defines `last_action` as raw decoder output, not
executed motion. The physical q/dq observation independently shows that absent
joints did not move. Earlier combined A/B runs that favored action-history
masking were confounded; an isolated straight-walk comparison favored native
reference plus raw history and the implementation now follows official parity.

## Supported input modes

The Python path now subscribes to the official `command`, `planner`, and `pose`
topics on port 5556 and decodes their 1280-byte self-describing headers.

| CLI mode | Reference source | Encoder behavior |
| --- | --- | --- |
| `motion` | finite 50 Hz `.npz` clip | G1 |
| `planner` | kinematic planner; upper and lower body from planner | forced G1 |
| `pico` in PLANNER | planner topic without VR3 fields | G1 |
| `pico` in VR3PT | planner topic with complete VR3 targets | teleop |
| `pico` in STREAMED_MOTION | fresh complete pose window | SMPL |
| `smpl` | official STREAMED_MOTION command plus fresh pose required | forced fail-closed SMPL |

Important mode behavior:

- Teleop is selected only when a fresh, valid three-point position is present.
  Orientation is optional and uses the official comfort-quaternion fallback.
- `--mode planner` remains G1 even if stray VR fields arrive.
- `--mode smpl` refuses to run unless `command.planner == false` and a complete,
  fresh SMPL window is available.
- An incomplete streamed pose never silently becomes G1.
- A planner message older than one second transitions to IDLE, zero movement,
  and G1 while preserving the last facing direction.

### Navigation frame correction

The official PICO manager accumulates right-stick yaw and publishes both
movement and facing in one persistent planner-session world frame. The local
runtime used to rotate those vectors by live robot/planner yaw a second time.
Real PICO vectors now pass through unchanged. Only explicitly local scripted
inputs are rotated once.

### VR3 calibration feedback

`sonic/common/robot_feedback.py` publishes the official `g1_debug` msgpack feedback
on port 5558. `body_q_measured` is the measured physical posture expanded into
29 MuJoCo motor slots, with the six absent axes set to zero. The official PICO
manager uses this value to recalibrate wrist FK when entering VR3PT.

Start feedback before entering VR3PT. If VR3PT was already active, leave and
re-enter it. The PICO process must report recalibration from measured posture;
its zero-posture fallback is not acceptable for a hardware trial.

## Planner wrapper and jitter mitigation

`sonic/model/planner_onnx.py` mirrors the official V0/V1/V2 ONNX contract,
four-frame context, two-control-frame lookahead, production V2 token mask,
30-to-50 Hz resampling, quaternion SLERP, and eight-frame replan crossfade.

### Constant straight-command refresh defect

The released C++ scheduler refreshes RUN at 0.1 seconds and CRAWLING at 0.2
seconds, but uses 1.0 second for ordinary locomotion. This can be hidden by
change-triggered replans: a turning stick continuously changes facing and
therefore replans at the 10 Hz planner-loop rate, while a perfectly constant
forward stick uses only the one-second timer.

Three operator recordings exhibited exactly that split. A deterministic
healthy-plant A/B then changed only the ordinary period: at 1.0 seconds a
0.4 m/s SLOW_WALK command produced 0.043--0.075 m/s median speed and
24--47% near-stop time; at 0.1 seconds all three released model profiles
produced 0.388--0.417 m/s and zero near-stop time. Left-ankle force RMS also
fell 17--47%. The default is consequently 0.1 seconds in
`PicoReferenceWindow`, the Python YAML, and the physical C++ launcher. Both
simulators expose an explicit period override for reproduction and logging.

This is a deliberate evidence-based deviation from the release timer, not a
change to planner tensors, command axes, gains, decoder output, or 23-DoF
projection. Full data and limitations are in
[`../docs/experiments/mujoco/2026-08-13_unitree_remote_forward_replan/`](../../docs/experiments/mujoco/2026-08-13_unitree_remote_forward_replan/README.md).

One concrete jitter source was fixed: the old blend independently mixed joint
position and joint velocity. When blend weight changes, that velocity omits
the derivative of the crossfade itself, so the encoder sees mutually
inconsistent q/dq references. The runtime now blends q and recomputes dq from
the final 50 Hz trajectory. The C++ wrapper received the same fix.

Correction after the later three-model trials: the reduced embodiment must not
project the planner-owned trajectory. Those waist/wrist values are coherent
29-DoF recurrent model context, not measured proprioception. The old +0.20 m/s
A/B was confounded by simultaneous action-history and scheduling changes. A
controlled isolated comparison found better progress and lower tilt with
native planner context, and removing projection eliminated the SLOW_WALK
right-shoulder artifact. Missing axes are neutral/zero only in measured
proprioception and omitted at the physical motor-output boundary.

This does not guarantee a smooth kinematic plan. Other likely jitter sources
remain:

- noisy stick, VR, or facing values triggering frequent replans;
- 10 Hz planner completion jitter and a result arriving too near its horizon;
- stochastic planner output, even with a fixed seed when context changes;
- discontinuity caused by an out-of-distribution 23-DoF measured posture;
- GPU/host scheduling stalls;
- real joint tracking lag converting a smooth reference into noisy feedback.

Keep official replan gating enabled. First compare planner `q`, derived `dq`,
replan times, and raw operator commands offline. Do not add a low-pass filter to
decoder actions blindly: it changes recurrent policy dynamics and adds phase
lag. If smoothing is needed after measurement, filter operator commands before
replanning or constrain replan rate/deadband, then validate in simulation.

Stable-stand IDLE is now the default. Releasing the stick follows the planner's
gait-ending IDLE transition, then smoothly centers over 1.5 seconds to the
measured startup stance before holding. This avoids freezing a mid-stride pose
and avoids indefinitely tracking the planner's heel-raised terminal stance.
`--no-idle-hold` is diagnostic only.

## Motion mode and real-world initialization

The simulator can set the full robot state directly to clip frame 0. Hardware
cannot. Motion mode therefore:

1. moves to the configured official default posture;
2. refuses a frame-0 gap above 20 degrees;
3. ramps to frame 0 over three seconds;
4. holds for one second and requires at most 5 degrees joint error and
   0.5 rad/s speed in the tail;
5. aligns the clip's initial yaw to measured pelvis yaw;
6. resets SONIC history and uses a one-second raised-cosine handover;
7. plays once by default and returns to default before damping.

Start with `motion_neutral_default.npz`, a six-second exact neutral hold with
zero velocity and zero pelvis tilt. It verifies ordering, gains, defaults,
logging, timing, and handover without asking for locomotion.

`motion_walk11_neutral_start.npz` is only a later simulation candidate. Its
first frame is neutral and source pelvis tilt stays below about 7.5 degrees,
but the source later asks for substantial nonexistent waist pitch. It is not a
safe zero-shot 23-DoF walking recommendation. `motion_walk_straight.npz` also
starts neutral but reaches about 12 degrees source pelvis tilt and is not the
first hardware clip.

Motion mode is SONIC reference tracking. It does not directly replay joint
positions open-loop. Direct walking trajectory replay was intentionally not
added as a hardware controller.

## Python TensorRT and ONNX Runtime

`sonic/model/inference_backend.py` provides:

- direct TensorRT 10 engine construction and execution using CUDA PyTorch
  buffers and `execute_async_v3`;
- per-model serialized engine caches keyed by model metadata, TensorRT/CUDA,
  GPU name/compute capability, precision, explicit TF32 math mode, cache
  schema, and input shape;
- a dedicated non-default CUDA stream per session with serialized context use;
- incompatible-cache rebuild;
- ONNX Runtime CUDA/CPU execution for debugging and golden traces;
- opt-in `auto` fallback with an explicit warning for offline tools only.

TensorRT is an optional environment because its wheel, CUDA, driver, and
PyTorch versions must match the deployment PC. Install with the project extra
only after selecting a compatible stack:

```bash
pip install -e '.[tensorrt]'
```

The reviewed hardware profiles already require direct TensorRT. The
robot-facing launcher rejects `auto`, ONNX Runtime, FP16, and TF32, verifies
the actual direct provider and engine file for every constructed model before
starting auxiliary streams or DDS, and records the complete contract. Offline
tools may select ONNX Runtime explicitly. FP16 and TF32 remain disabled until
a separate numerical trace proves acceptable parity.

## Logging and safety changes

`deploy_eskin.py --auto-record` records one coherent 50 Hz LowState snapshot
per row, including:

- exact post-limit command and measured physical q/dq;
- full 29-slot command/measured motor arrays;
- tracking error, saturation, raw 29-D decoder action, and missing-axis demand;
- encoder input, token, decoder input, and encoder/decoder/planner timing;
- pelvis quaternion and explicit roll/pitch/yaw;
- LowState age/tick/CRC rejection count and loop/compute timing;
- both motor temperatures, fault state, estimated torque;
- PICO mode, age, stale state, VR targets, and planner timing;
- command writer and 500 Hz rewriter counts.

Run `sonic/scripts/analyze_sonic_episode.py EPISODE` after every staged trial.

Safety checks now fail to damping on stale LowState, command-writer timeout,
absolute joint speed, motor temperature, motor fault, pelvis tilt, tracking
error, excessive command step/rate, NaN/invalid quaternion, or planner failure.
The watchdog's damping publication bypasses its own normal-command latch, so a
latched watchdog cannot block emergency damping. Joint targets are clamped to
the configured physical limits.

The thresholds are conservative starting values, not certified limits:

| check | configured threshold |
| --- | --- |
| LowState age | 100 ms |
| command age | 100 ms |
| absolute joint speed | 35 rad/s |
| motor temperature | 75 C |
| pelvis roll/pitch norm | 0.65 rad |
| joint tracking error | 0.45 rad |
| target rate | 12 rad/s |

The actual deployment SDK's CRC, motor temperature shape, and fault fields
must be verified during a read-only/neutral test before locomotion. Do not
disable these checks merely to get past an unexplained abort.

## Validation completed on the RTX 4090

- `pyzmq 27.1.0` and `msgpack 1.2.1` are installed and import successfully.
- All Python files compile.
- Twenty parity/unit tests pass via `python sonic/tests/test_sonic_parity.py`,
  including explicit TensorRT flags, cache identity, and hardware fail-closed
  regressions.
- The 23- and 29-DoF configurations instantiate with 23/6 and 29/0
  physical/fixed joints respectively.
- The official low-latency encoder, decoder, and V2 planner pass ONNX checking.
- The full supplied direct-TensorRT verifier passed against CPU ONNX Runtime:
  raw 29-D action max error `4.47035e-7`, physical 23-D action max error
  `4.47035e-7`, and planner joint-q max error `4.92692e-5`.
- Warm model-only policy p99 was `6.941 ms` and planner p99 was `9.694 ms`.
- Planner installed velocity exactly matched finite-difference installed q
  (`max_abs=0.0`).
- Shell launchers pass `bash -n`.

The cross-repository remediation record is
`../docs/analysis/sonic_4090_remediation.md`. PICO hardware, DDS on the robot network,
and physical robot motion remain unvalidated.

## 2026-08-12 C++ simulation-to-robot promotion gate

The production physical path uses the same C++ binary, low-latency
encoder/decoder, V2 planner, observation YAML, `g1_23dof` embodiment, and FP32
math contract as the completed headless MuJoCo matrices. The C++ fixes are
therefore not reimplemented in a second hardware controller: native recurrent
planner context, raw native `last_action`, missing-motor output holds,
null-state handling, asymmetric arm torque limits, and persistent fault
damping are in the shared binary.

`sonic_cpp/teleop/run_synthetic_23dof_validation.sh --comprehensive` now
promotes a candidate only after both exhaustive matrices and extra acceptance
checks pass. It records SHA-256 hashes of the tested binary, models, config,
reference motion, physical launch scripts, PICO adapter, and evidence in:

```text
/data/clhsieh/cache/e_skin/validation/robot_candidate_23dof.json
```

The no-DDS hardware preflight verifies the receipt. `sonic_cpp/deploy.sh`
requires it for every real `g1_23dof` launch and verifies it again after the
mandatory build, immediately before physical confirmation. The PICO hardware
terminal verifies it a third time before publishing manager traffic. A changed
binary/model/config/script, retargeted symlink, mutated evidence file, or Git
revision rejects deployment and requires a new comprehensive run.

The candidate gate checks identity and the known 23-DoF regressions; it does
not convert simulation into hardware certification. Simulation-only parity and
fall relaxations remain forbidden on a real interface. Hardware keeps CRC,
activation bridging, watchdogs, command-rate/tracking/fall guards, and
irreversible damping. The mechanical, firmware, thermal, cable, support,
independent-stop, and trained-observer gates remain mandatory in
`../docs/smpl_teleop.md`.

## Required gates before 23-DoF locomotion

1. Install the deployment PC's Unitree SDK, CUDA/TensorRT stack, and Git LFS
   files; build engines before connecting motor commands.
2. Verify incoming HG LowState CRC, temperatures, motor fault codes, joint
   order, IMU WXYZ convention, and 50/500 Hz rates read-only.
3. Run record-only default hold, then neutral SONIC motion with support and no
   camera/tactile load. Inspect every safety and timing field.
4. Run the same input/state trace through Python ONNX Runtime, Python TensorRT,
   and official-derived C++ and compare encoder input, token, decoder input,
   raw 29 action, and physical target.
5. Run 29-DoF and 23-DoF MuJoCo A/B with identical planner commands. Quantify
   pelvis tilt, tracking, saturation, missing waist demand, and planner jitter.
6. Validate PLANNER/G1 with `--idle-hold` and zero movement before slow walking.
7. Validate VR feedback and wrist recalibration before moving arms.
8. Inspect ankle temperatures continuously. Official issue #174 reports severe
   ankle-roll heating even in idle; this robot's own prior asymmetric ankle
   heating makes that a separate stop condition.

## Fundamental unresolved risk

The strongest remaining failure hypothesis is embodiment mismatch, not Python
alone. A 23-DoF pelvis IMU supplies floating-base roll/pitch through SONIC's
gravity/orientation observations, but it cannot measure nonexistent waist
roll/pitch joints or create their stabilizing torque. Synthesizing waist joint
angles from pelvis orientation would give the model false proprioception.

Official issue #168 reports that zeroing the same six joints did not yield
stable 23-DoF transfer and that locking waist pitch damaged the learned balance
prior. No demonstrated zero-shot fix is available. Stable locomotion may
require 23-DoF simulation fine-tuning, distillation, residual control, or a
policy trained with locked/removed waist axes. C++ and TensorRT can reduce
latency and jitter; they cannot restore missing mechanical authority.
