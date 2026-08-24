# Suspended joint calibration

This package measures and models the difference between the ideal G1 23-DoF
joint coordinates used by SONIC/MuJoCo and a particular physical robot. It is
separate from SONIC: the physical measurement executable loads no policy,
planner, encoder, or decoder and cannot generate locomotion.

## Coordinate boundary

The intended final boundary is bidirectional:

```text
hardware encoder q  -> hardware-to-model calibration -> SONIC proprioception
SONIC model target  -> model-to-hardware calibration -> joint PD target
```

The inverse target map is therefore applied immediately before joint PD, not
to planner references, decoder output history, or SONIC action history. A
target-only map is useful for an initial experiment, but a true encoder-zero
or linkage-coordinate correction must be applied consistently in both
directions or the tracker observes a different coordinate system than the one
it commands.

## Components

- `gated_protocol.py` is the current 23-DoF measurement design. It generates
  fixed 50 Hz pilot/screening blocks, each starting and ending at a captured
  neutral pose. The script, not the operator, chooses joint order, range,
  scale, static/loop content, combined leg probes, and the suspended
  walking-like signature. Full screening also contains isolated 0.35--2.2 Hz
  leg sweeps, a coupled multi-frequency sweep, and asymmetric direction
  reversals so identification is not limited to ordinary walking data.
- `scripts/run_gated_calibration_mujoco.sh` is the current visual rehearsal.
  It uses two compliant shoulder attachment points and a dynamic floating
  root, so a tilted hanging baseline, sway, and gravity preload are measured
  rather than overwritten with a level pelvis.
- `scripts/run_gated_calibration_robot.sh` shares the exact protocol with the
  C++ physical runner. It defaults to build/parse dry-run with no DDS. In an
  explicitly armed foreground run, it first moves smoothly from measured
  launch angles to the parity-checked NVIDIA/SONIC initial pose. After
  `INITIAL POSE READY`, F1+F3 captures a 1.5-second joint/IMU baseline, A
  approves each displayed block, B skips/cancels, and Select aborts to damping.
- `interactive.py` is the device-independent operator state machine. It gates
  motion on an explicit neutral capture, selects individual or multi-joint
  motions, switches static/loop operation, and adjusts range, amplitude, and
  tempo without knowing whether input came from a viewer or remote.
- `scripts/run_interactive_calibration_mujoco.sh` is the one-command visual
  rehearsal. The ideal hoist fixes only the floating base; all 23 physical
  joints retain gravity, inertia, effort limits, and official SONIC PD gains.
- `protocol.py` creates a deterministic 50 Hz, one-joint-at-a-time plan. Every
  target is an offset from the pose captured at run time. Screening is the
  physical default; full range is separately gated.
- `joint_mapping.py` implements a bounded monotone inverse with a local
  increasing/decreasing residual for hysteresis. Outside measured knots it
  keeps only the bounded endpoint correction; it never clamps an absolute
  SONIC target back to a screening endpoint.
- `scripts/run_suspended_calibration_mujoco.py` runs the plan against the real
  23-DoF MJCF, official SONIC Kp/Kd, gravity, inertia, and force limits while
  an ideal pelvis harness holds the root.
- `sonic_cpp/.../g1_suspended_calibration.cpp` is the guarded real-robot
  measurement runner. It accepts both the gated and legacy deterministic
  plans, sends 50 Hz targets through a 500 Hz LowCmd writer, and buffers all
  useful HG LowState data in memory: timing/tick/modes/remote, captured and
  live IMU attitude/rates/acceleration/temperature, and per-physical-motor
  mode, q/dq/ddq, estimated torque, two temperatures, voltage, raw sensor
  words, and fault code.
- `scripts/merge_external_calibration_truth.py` synchronizes external
  model-coordinate joint angles with the motor log.
- `scripts/fit_suspended_calibration.py` fits the provisional inverse map.

## What each measurement can identify

| Measurement | Gain/lag | Hysteresis/stiction | Encoder zero/link geometry | Load compliance/contact |
| --- | --- | --- | --- | --- |
| Suspended encoder sweep | yes | yes | no | no |
| Suspended external pose sweep | yes | yes | yes | no |
| Harness-supported contact sweep | yes | yes | yes | yes, locally |

An encoder cannot validate its own zero. For that reason, an `encoder_only`
profile is always written with `approved_for_deployment=false`, and the profile
class rejects attempts to mark it approved. `external_truth` fitting refuses a
plain robot log: `q_external_0` through `q_external_28` must first be merged.

The robot being measured is physically 23-DoF. The CSV/protocol keeps the
official sparse 29-slot indexing convention only for compatibility; absent
slots 13, 14, 20, 21, 27, and 28 are never commanded. HG LowState does not
provide foot force, external foot/link pose, or shoulder-strap tension, so
those signals require separate synchronized instrumentation.

## Safety contract

- The real runner requires both `--arm-real-robot` and a foreground terminal.
- It verifies the plan without DDS before any armed invocation.
- Its first active command holds the measured pose, followed by a minimum-jerk
  move to the exact configured default pose at no more than 0.15 rad/s target
  rate. Select aborts the startup move; the pose settles for two seconds before
  capture is enabled.
- Every gated block waits for explicit operator approval, starts/ends at the
  captured pose, and allows B to cancel through a smooth neutral return.
- Single-joint probes excite one motor; named combined blocks are bounded and
  shown before approval. Absent 23-DoF slots remain zero-offset.
- Static gated transitions use minimum jerk at no more than 0.35 rad/s. The
  fast pilot uses at most half of the small screening envelope. Full screening
  expands isolated legs to 0.18–0.60 rad and waist yaw to 0.18 rad, while arms remain
  at the collision-free 0.08–0.10 rad envelope. Explicit dynamic blocks taper
  amplitude over 0.35--2.2 Hz and never exceed 2.0 rad/s generated target rate.
- The captured pose and every target must remain at least 0.04 rad from the
  modeled joint limits.
- Start requires a 300 ms F1+F3 chord followed by a 1.5-second steady capture:
  `max |dq| <= 0.20 rad/s`, IMU gyro norm at most 0.08 rad/s, and no more than
  two degrees of attitude spread.
- Select or Ctrl-C enters damping. Stale LowState, a motor fault, 70 C motor
  temperature, or `|dq| > 5 rad/s` aborts into damping.
- The physical launcher defaults to dry-run. Full range additionally requires
  `--allow-full-range` and should never precede review of screening data.

The older free-form interactive physical state machine remains intentionally
unconnected to DDS. The review-gated scripted runner is the physical path;
freshness, temperature, fault, and damping checks remain an additional
hardware layer around the shared plan.

The code does not turn a suspended identification result directly into a
walking correction. Promotion requires external geometry, repeat-run
cross-validation, a harness-supported load test, and an explicit reviewed
deployment step.
