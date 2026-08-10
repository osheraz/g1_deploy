# SONIC 23-DoF deployment parity changes

Date: 2026-08-10

This is the change and risk ledger for adapting the released 29-DoF SONIC
planner/encoder/decoder to the physical 23-DoF G1. It is not a safety
certification. All first executions must be supported/tethered with an
independent damping/deadman path.

## Upstream provenance

- NVIDIA reference: `NVlabs/GR00T-WholeBodyControl` commit
  `1983e88888217f6c69283cf3a9d1af01e87f07af`.
- Local deployment upstream: `osheraz/g1_deploy` commit
  `6d3d36c0ff3200cba03135cf96df887d79295567` before these edits.
- Both local commit IDs exactly matched GitHub `main` on 2026-08-10.
- The local encoder, decoder, and observation-config files are byte-for-byte
  identical to the official `policy/release` files; the actual ONNX inputs are
  `obs_dict[1,1762]` and `obs_dict[1,994]`.
- GitHub contains no `sonic.planner.kinematic.SonicKinematicPlanner`. The old
  local import therefore depended on an unreleased/ambient research checkout.
  `policy/sonic/planner_onnx.py` now provides a pinned, auditable wrapper based
  on the official C++ planner and its documented ONNX contract.
- The official dormant ONNX C++ backend enables token counts 6--11, while the
  executable actually constructs `LocalMotionPlannerTensorRT`, whose active
  mask enables only 9--11. The local wrapper deliberately follows that
  production TensorRT mask even when ONNX Runtime is its execution engine.

## Applied changes

### Planner parity

`policy/sonic/planner_onnx.py` now implements:

- V0/V1/V2 named ONNX input and output validation.
- Four frames of 30 Hz MuJoCo-qpos context.
- The official two-control-frame planning lookahead.
- The production V2 allowed-token mask from
  `localmotion_kplanner_tensorrt.hpp`.
- 30 Hz to 50 Hz interpolation using linear position/joint interpolation and
  shortest-path quaternion SLERP.
- Joint velocity recomputation from resampled positions.
- The official eight-frame replan blend, including inference-latency-aware
  placement of the generated trajectory.
- The official end-of-IDLE lower-body readaptation state machine (0.10/0.05/
  0.045 rad thresholds and a 2% per-tick measured/nominal blend).
- Initial normalized planner heading plus encoder-side alignment to the
  measured robot heading.
- Planner inference and total latency diagnostics.

ONNX Runtime provider order is TensorRT, CUDA, then CPU. The release executable
uses the direct TensorRT backend; a CPU-only provider is for parity/debugging,
not presumed adequate for live locomotion. Encoder and decoder sessions now use
the same provider preference instead of being hard-pinned to CPU, and all three
sessions print the providers actually selected.

`policy/pico_reference.py` now evaluates replanning at 10 Hz, with the official
mode-dependent periodic intervals and change rules. A planner failure is
surfaced to `deploy_eskin.py`, whose cleanup path sends damping; a failed
planner is not silently allowed to continue walking a stale trajectory.

### PICO encoder mode and stale input

Encoder mode is selected dynamically:

- Fresh valid `vr_position`: teleop encoder mode 1; a missing orientation uses
  the official comfort-quaternion fallback.
- Fresh navigation without VR3: G1 encoder mode 0.
- No fresh complete planner message for one second: IDLE command, zero movement,
  last facing preserved, G1 encoder mode 0.

VR fields are sampled once per control tick and that same sample determines
both encoder mode and VR targets, preventing a transition tick from selecting
the teleop encoder without corresponding targets.

If every fresh message always contains valid `vr_position` three-point data,
fixing the active encoder to teleop would produce the same mode during normal
streaming.
It was still unsafe at startup/disconnect and invalidated navigation-only tests,
so mode selection is now dynamic.

`common/pico_client.py` timestamps only fully decoded messages and rejects
short/trailing/missing-field payloads. `common/scripted_teleop.py` now omits VR3
fields for `nav` and `idle`, making those genuine mode-0 tests.

### Navigation coordinate frame

The official PICO manager integrates the right stick into a persistent yaw:

```text
facing_world = [cos(accumulated_yaw), sin(accumulated_yaw)]
movement_world = R(facing_world) @ movement_local
```

Both published vectors are already in planner-world coordinates. The former
local implementation rotated them by the planner root yaw again. At a 90°
heading, for example, an intended world `+Y` forward command became world `-X`.

`PicoClient.navigation_frame` is now `world`, so real PICO vectors pass through
unchanged. The scripted and Unitree-remote stand-ins explicitly declare
`local`, and only those are rotated once.

### Recurrent decoder history

The raw 29-D decoder output is now the next step's `last_action`, matching the
official C++ code and SONIC training/evaluation runner. The six absent joint
outputs are removed only from physical application, not recurrent history.

This is the correct way to remain closer to the learned SONIC distribution.
It does **not** make the 23-DoF mechanics equivalent: large waist-roll/pitch
outputs still indicate that the policy wants stabilization authority the robot
does not possess. Those six raw outputs are now logged for this reason.

Startup proprio/action history is zero-padded, matching the official
`StateLogger`, rather than repeating the first live sample ten times.

### Mode-specific encoder buffer

`policy/sonic/obs.py` retains the fixed 1762-D model layout but zeroes every
term outside the selected official encoder branch:

- G1: mode, full 29-joint position/velocity future, future anchor orientation.
- Teleop: mode, lower-body future, VR3 position/orientation, current anchor
  orientation.
- SMPL: mode, SMPL joints/orientations, wrist future.

### VR3 measured-posture calibration

`common/robot_feedback.py` publishes official-compatible
`[g1_debug][msgpack]` feedback at the 50 Hz control rate. `body_q_measured` is
the actual 23-joint LowState posture expanded to the 29-joint MuJoCo layout;
waist roll/pitch and four wrist pitch/yaw joints are zero.

The official PICO manager already recalibrates wrist FK from this field when
entering `PLANNER_VR_3PT`. Because local tactile occupies port 5557, feedback
uses port 5558. The PICO process must be launched with:

```bash
--zmq_feedback_host 127.0.0.1 --zmq_feedback_port 5558
```

This avoids applying a second calibration to already-calibrated VR targets in
the controller. A custom PICO publisher that does not implement the official
recalibration routine will not gain calibration merely by receiving this
feedback.

### Experiment logging

Press X or pass `--auto-record`. Each 50 Hz row in `state.npz` now contains:

- `command_q_23`: exact post-limit joint positions put into LowCmd.
- `command_q_23_unclipped` and `command_saturated_23`.
- `measured_q_23`, `measured_dq_23`, and `tracking_error_q_23`.
- Raw motor-order command/measured q and dq for all 29 message slots.
- Both motor temperature channels, motor-state/error code, and estimated torque
  when provided by the SDK.
- Raw 29-D and selected 23-D SONIC action, including the six unavailable
  actions separately. `sonic_missing_action_raw` is ordered as waist roll,
  waist pitch, left wrist pitch, right wrist pitch, left wrist yaw, right wrist
  yaw; the IL indices and names are also stored in `meta.json`.
- Full 1762-D encoder input, 64-D token, full 994-D decoder input, and
  encoder/decoder/policy/planner inference time.
- LowState age/tick, PICO age/staleness/mode, loop period and compute time.
- Main command-send and 500 Hz repeat counters.
- Existing IMU, tactile, camera, and command data.

Joint, IMU, raw-motor, and tick fields for a row are read from one locked
LowState message snapshot, rather than mixing callback updates within a row.

The 500 Hz writer repeats the same snapshot and is therefore counted rather
than storing 25 duplicate rows per policy step.
Full tensor logging is intentionally diagnostic-heavy (about 0.55 MB/s before
other fields); keep first validation episodes short.

The DDS `--net` option is now passed to `ChannelFactoryInitialize`; it was
previously parsed and ignored. All non-keyboard exceptions now enter a common
cleanup path that stops reference threads and sends repeated damping before the
500 Hz writer exits. LowState older than 100 ms or an absolute measured joint
velocity above the official 35 rad/s threshold also raises into that damping
path instead of letting an invalid state drive new commands. The same checks
now remain active during the initial default-pose move, the wait-for-start
hold, motion-frame alignment, and the normal return ramp.

Example conservative logging invocation:

```bash
python deploy_eskin.py --mode motion --auto-record --no-camera --no-tactile \
  --clip policy/sonic/motion_neutral_default.npz --no-loop \
  --net enp132s0 --duration 6
```

### Motion preparation and initialization

`scripts/export_sonic_motion.py` now uses the official sampling grid,
quaternion SLERP, and recomputes 50 Hz velocity rather than independently
interpolating stored velocity.

`scripts/prepare_sonic_motion.py` generated:

1. `policy/sonic/motion_neutral_default.npz`: six seconds, exact configured
   default q, zero dq, zero pelvis tilt. This is the selected first test.
2. `policy/sonic/motion_walk11_neutral_start.npz`: 15.7 seconds, a three-second
   quintic reference transition from the exact default pose into `walk11`.
   Among the available walking clips under 10° pelvis tilt, `walk11` has the
   smallest unprepared frame-0 present-joint gap (31.4°); maximum source pelvis
   tilt is 7.4°. This is a later test only.

Motion mode now:

1. Moves to configured default.
2. Compares measured q to clip frame 0.
3. Refuses gaps above 20°.
4. Ramps to frame 0 before initializing policy/history.
5. Holds frame 0 for one second and refuses activation unless the last quarter
   of the hold remains within 5° maximum joint error and 0.5 rad/s maximum
   absolute joint velocity.
6. Aligns clip yaw to measured robot yaw.
7. Plays once by default and returns to default on normal completion.

The selected clips start at the configured default, so MuJoCo's clip-frame-0
initialization and the real robot handoff represent the same joint state. A
one-second continuous raised-cosine policy handoff is enabled by default: its
first command is exactly the measured pose and its 50th command is exactly the
policy target. The official C++ controller switches abruptly after its default
pose hold; use `--no-blend` only for off-robot parity/debugging. The applied
`policy_handover_alpha` is included in every experiment row.
`scripts/sim2sim_sonic.py` now uses the identical handoff and default, replacing
its older fixed 50% mixture.

`--mode motion` remains SONIC reference tracking; it does not bypass the
encoder/decoder and directly replay q. An open-loop direct walking replay was
not added because it is not the official SONIC pipeline and is unsafe as an
unsupported locomotion controller.

### Official checkpoint selection

The configured encoder/decoder checksums exactly match the official **Default
SONIC** deployment pair. This is intentional for motion tracking and the
PLANNER/G1 use case. The official model card calls Default SONIC the
general-purpose planning-compatible model.

The newer **Low-latency teleoperation** model is not a faster implementation of
the same tensors. It shortens reference lookahead and changes the encoder from
the default 1762-D `step5` layout to a 1247-D `step1` layout. It is intended for
responsive streamed SMPL/VR/VLA control and requires its encoder, decoder, and
observation configuration together. The local runner now validates the Default
SONIC ONNX shapes and `sonic_model_variant: default`, so swapping only a file
fails at startup instead of silently building the wrong observations.

For current PLANNER/G1 debugging, keep Default SONIC. Low-latency is a separate
future integration for VR3 responsiveness; it does not address missing 23-DoF
waist authority or Ethernet/DDS latency.

Official references:

- [SONIC model card](https://github.com/NVlabs/GR00T-WholeBodyControl#model-card)
- [Low-latency observation configuration](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic_deploy/policy/release/observation_config_low_latency.yaml)

### Explicit PLANNER / G1 mode

`deploy_eskin.py --mode planner` is now the local counterpart of the official
PICO mode table's PLANNER state:

- kinematic planner supplies both lower and upper-body reference motion;
- encoder mode is forced to G1 even if incoming messages accidentally contain
  VR fields;
- PICO movement/facing vectors remain in their already-global planner-session
  frame;
- the planner starts from measured 23-joint LowState posture expanded to 29
  slots and aligns normalized planner heading to measured pelvis-IMU heading;
- the official one-second stale-input transition still produces IDLE;
- the Unitree-remote stand-in emits no VR fields and mirrors the official PICO
  persistent-yaw, deadzone, and walk/run speed math.

Use `--mode pico` for intentional dynamic PLANNER/VR3 switching. Robot-feedback
wrist recalibration applies only when entering VR3, not to PLANNER/G1.

### Pelvis proprioception and the 23-DoF limit

The 23-DoF G1 still senses the floating pelvis orientation through its pelvis
IMU. SONIC already consumes roll/pitch information as body-frame gravity in
the 10-frame decoder history, and the encoder consumes measured-base versus
reference-pelvis orientation. The logger now exposes explicit
`pelvis_roll_rad`, `pelvis_pitch_rad`, and `pelvis_yaw_rad` in addition to the
quaternion and gravity vector.

This is not the same as sensing the missing `waist_roll`/`waist_pitch` joints.
Those mechanical joints do not exist on the consumer robot. Synthesizing their
angles from pelvis IMU and putting them in the 29-joint proprio slots would tell
SONIC that nonexistent actuators moved and would not restore their torque.
Adding new proprio dimensions is also impossible without changing and
retraining the fixed-shape model.

Official issue [#168](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/168)
reports the same 23-DoF transfer: zeroing all six unavailable outputs did not
produce stable sim2sim, and the reporter found waist-pitch locking particularly
damaging to the 29-DoF balance prior. As of this audit, it has no maintainer
resolution or demonstrated zero-shot 23-DoF success. This is the strongest
evidence that embodiment mismatch remains the primary suspect.

### Offboard latency and C++ decision gate

An RTX4090 only accelerates inference; it does not fix Ethernet/DDS delay,
LowState age, Linux scheduling jitter, packet loss, or the 500 Hz command
publisher. A full C++ rewrite should therefore follow measurement rather than
precede it. The official direct TensorRT/CUDA-graph stack is the lower-jitter
production target, but Python/ONNX Runtime is not automatically the cause when
encoder+decoder compute fits comfortably within the 20 ms control period and
the planner remains asynchronous at 10 Hz.

`scripts/analyze_sonic_episode.py` reports p50/p95/p99/max for loop period,
control compute, state age, policy/encoder/decoder/planner time and PICO age,
then correlates these with joint tracking, pelvis tilt, saturation and missing
waist/wrist action demand. It deliberately does not claim true one-way network
latency: robot and PC clocks are not synchronized in the current packet path.

On the actual RTX4090 deployment PC, first confirm printed and logged execution
providers, then run record-only and neutral-policy episodes before locomotion.
If inference is fast but loop/LowState timing is late, prioritize CPU isolation,
real-time scheduling, wired DDS configuration and synchronized timestamping.
If those remain unacceptable, port the real-time controller to the official
C++ stack. If all timing is healthy but tilt/tracking diverges with large
unavailable waist demand, C++ will not fix the 23-versus-29-DoF mechanics.

## Validation performed without hardware

- Both upstream repositories were verified against current GitHub `main`.
- All Python sources pass AST parsing and bytecode compilation.
- Mode-mask tests confirmed zeros in every non-selected 1762-D region.
- Fifteen parity unit tests pass with the deployment transport dependencies,
  including the official planner wire decoder, missing-orientation fallback,
  one-second idle transition, navigation frames, forced PLANNER/G1 behavior,
  planner blending, episode analysis, and exact 23-to-29 feedback joint
  placement.
- Planner resampling/context/blending math was exercised on synthetic finite
  trajectories.
- A disposable ONNX Runtime 1.24.3 CPU smoke test executed the actual official
  files: encoder `(1,1762)->(1,64)` and decoder `(1,994)->(1,29)` were finite;
  100-run median/p99 latency with the deployment's one-thread settings was
  0.758/0.873 ms and 0.573/0.655 ms respectively on this host.
- The actual V2 planner initialized and replanned a SLOW_WALK command into 73
  finite 50 Hz frames. CPU inference measured 47.1 ms initialization and 46.6
  ms replan on this host. These are compatibility smoke-test numbers, not the
  deployment computer's latency qualification.
- The explicit PLANNER/G1 profile was instantiated with the actual planner,
  encoder, and decoder ONNX files. One finite control action was produced with
  mode `g1` and exact `(1762,)`/`(994,)` logged observation shapes. This host
  exposed CPUExecutionProvider only; it is an integration smoke test, not the
  required RTX4090 provider/timing qualification.
- A first-control-tick CPU probe with the exact finalized observation mask,
  zero-padded official history, and neutral reference produced a maximum raw
  present-joint action of 0.73. After per-joint scaling, the largest unblended
  target displacement was 18.4 degrees. This is why the safer handoff is the
  default even though it intentionally differs from official C++.
- An offline reference-state oracle trace (measured state forced to each
  reference frame; not a closed-loop stability result) ranked the prepared
  `walk11` clip better than the other locomotion candidates. Its raw present
  action p95/max was 1.74/2.73 and 11.8% of reference frames requested at least
  one target outside the hardware limits. `walk20` requested saturation in
  42.6% of frames, `walk_fwd2` in 27.0%, and `m19` in 99.6%. `walk11` is still
  a high-risk later-stage test, not a hardware approval.
- Generated clip audit:
  - neutral: frame-0 present-joint gap below `2e-6` degrees, tilt 0°, dq 0;
  - prepared walk: same frame-0 gap, maximum tilt 7.424°, finite dq.

MuJoCo, Unitree SDK/DDS, PICO XR, and live hardware execution were not available
on this host. Run the deployment's managed environment and staged hardware
gates below before locomotion.

## Remaining risks and required gates

1. **23-DoF dynamics are out of distribution.** Raw action history fixes a
   semantic error but cannot restore missing waist roll/pitch authority or the
   29-DoF torso/wrist inertia. Fine-tuning/distillation with the actual 23-DoF
   model may still be required.
2. **`phantom_hold_ref_dof` is unresolved.** Holding missing reference joints
   at zero removes trained waist motion; retaining it presents an impossible
   target. Compare both in 23-DoF simulation and inspect missing-joint actions.
3. **Runtime latency must be measured.** The official stack uses TensorRT/CUDA
   graphs. Confirm the wrapper printed an accelerated provider. Abort hardware
   testing if p99 policy compute approaches 20 ms or planner generation
   outruns its usable horizon; do not assume the CPU fallback is adequate.
4. **No live C++/Python numerical parity trace yet.** Replay identical context,
   commands, state history and references through both implementations and
   compare planner trajectory, 1762-D encoder input, token, 994-D decoder input,
   and raw 29-D action.
5. **Hardware watchdog parity is incomplete.** A 100 ms LowState timeout and
   the official 35 rad/s measured-dq limit now transition to damping, but the
   Python layer still needs validated CRC rejection, temperature hysteresis,
   motor error handling, absolute q/command-rate limits, IMU tilt limits, and
   an independent process/hardware deadman.
6. **VR feedback depends on the official manager.** Confirm the PICO console
   prints measured-q wrist recalibration, not its zero-q fallback, every time
   VR3PT is entered.
7. **Prepared walking is not yet approved.** First pass neutral hold in 23-DoF
   MuJoCo from deployment default, then supported hardware. Only then test the
   prepared walk, once, at reduced workspace/energy with no looping.
8. **No real experiment log was available for this audit.** Use the new log to
   correlate instability with tracking error, missing-waist action magnitude,
   saturation, state age, timing overruns, pelvis tilt, and temperature.

## Recommended gate order

1. Offline C++/Python tensor parity.
2. 23-DoF MuJoCo neutral clip, initialized from configured default.
3. Supported hardware `--mode record --auto-record` default-pose hold.
4. Supported hardware neutral SONIC clip, finite six seconds.
5. PICO idle/navigation-only with no VR fields; verify encoder mode 0.
6. Enter VR3PT; verify measured-q recalibration and encoder mode 1 before arm
   motion.
7. Supported, finite prepared walk only after all logged timing/tracking gates
   pass.
