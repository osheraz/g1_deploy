# PICO teleop

Live VR teleop drives the SONIC tracker. Run with `deploy_eskin.py --mode pico`
for official message-selected PLANNER/VR3 behavior, or `--mode planner` for an
explicit PLANNER-only profile that always uses the G1 encoder.

## Pieces

| | |
| --- | --- |
| PICO publisher | separate process owning the XR SDK; publishes the `planner` ZMQ topic |
| [`common/pico_client.py`](../common/pico_client.py) | subscribes and decodes that topic |
| [`policy/pico_reference.py`](../policy/pico_reference.py) | official-style planner scheduling, mode selection, encoder window |
| [`policy/sonic/planner_onnx.py`](../policy/sonic/planner_onnx.py) | Python ONNX mirror of the official C++ planner pipeline |
| [`common/robot_feedback.py`](../common/robot_feedback.py) | publishes measured 23-DoF posture as official-compatible `g1_debug` feedback |
| [`policy/sonic_runner.py`](../policy/sonic_runner.py) | encoder -> token -> decoder -> 23-dof action |

## Wire format

`[b"planner"][1280-byte JSON header][little-endian payload]`, from
NVlabs/GR00T-WholeBodyControl `gear_sonic/utils/teleop/zmq/zmq_planner_sender.py`.

Always present:

| field | |
| --- | --- |
| `mode` i32 | `LocomotionMode`, includes object carrying |
| `movement` f32[3], `facing` f32[3] | base navigation |
| `speed`, `height` f32 | -1 = planner default |

Optional: `upper_body_position/velocity`, `left_hand_joints`, `right_hand_joints`
(Dex3 hand control), `vr_position` f32[9] (3 endpoints xyz), `vr_orientation`
f32[12] (3 endpoints wxyz), `vr_compliance`.

Check the stream on its own:

```bash
python -m common.pico_client --server <publisher-ip>
```

## Command mapping

The official PICO manager's `YawAccumulator` integrates right-stick yaw into a
world-frame facing vector. It rotates left-stick motion from the operator's
local frame into that same planner-world frame before publishing. Therefore
`PicoClient` passes real PICO `movement` and `facing` directly to the planner;
it does not rotate them again by robot yaw. Scripted/Unitree-remote stand-ins
declare local framing and are converted once by `pico_reference.py`.

Here “world” means the planner session frame: `YawAccumulator.reset()` makes
`+X` forward whenever planner mode is freshly entered. It is persistent while
that planner session runs, but it is not an external map/odometry frame.

`pico_reference.py` feeds the planner:

| topic field | `SonicKinematicPlanner.update()` |
| --- | --- |
| `mode` | `mode` |
| `speed` | `target_vel` |
| `height` | `height` |
| `movement` | `movement_direction` |
| `facing` | `facing_direction` |

`gather(10, 5)` then returns the encoder's future window, and the VR endpoints
pass through in the pelvis frame (`vr_targets_local=True`).

Encoder mode follows official message presence:

- A fresh message containing valid `vr_position` selects teleop mode (1);
  `vr_orientation` is optional and falls back to the official comfort quats.
- Navigation without VR3 fields selects G1 mode (0).
- After one second without a fresh complete planner message, the command
  transitions to IDLE and G1 mode while preserving the last facing direction.

If every fresh message always contains valid VR3 fields, teleop mode is the
correct active mode. Dynamic selection still matters for startup, disconnects,
malformed messages, and navigation-only testing.

### Explicit PLANNER / G1 profile

The official mode table describes PLANNER as: locomotion planner active,
upper body supplied by the planner, encoder G1, and joysticks controlling
direction and heading in walking/running modes. The local equivalent is:

```bash
python deploy_eskin.py --config g1_23dof_sonic.yaml --mode planner \
  --net enp132s0 --auto-record --no-camera --no-tactile
```

`--mode planner` subscribes to the same PICO `planner` topic but hard-selects
G1. If a stale PICO process accidentally appends VR3 fields, the fields are
logged as ignored and do not switch the encoder or upper-body reference. Use
`--mode pico` when intentional PLANNER-to-VR3 switching is wanted.

At startup, the reference is calibrated from one measured LowState snapshot:
the 23 physical joints are installed into their 29 SONIC slots, the six absent
joints are zero, and the normalized planner +X heading is aligned to the
measured pelvis-IMU yaw. Freshly enter PLANNER in the PICO manager after the
deploy is ready so its `YawAccumulator.reset()` also defines +X forward for
the same session. The manager then publishes persistent world/session-frame
movement and facing vectors; the deploy does not rotate them again.

For a headset-free bench/simulation check, `--mode planner --teleop joystick`
uses the Unitree remote and emits no VR fields. The MuJoCo `--unitree-remote`
path uses `GamepadManagerTeleop`, which mirrors the physical C++
`gamepad_manager`: `Start` activates control, `Select` stops, the D-pad selects
one of four motion sets, `X`/`Y` cycle next/previous within non-boxing sets,
`B` resets the set, and the sticks control movement/facing. Its set tables,
fixed speed/height values, edge detection, 45-degree movement bins, smoothing,
deadzone, crawling transitions, and boxing auto-revert follow the native
controller. The simpler `JoystickTeleop` remains available for keyboard and
scripted tests.

## Setup

The planner ONNX is ~774 MB, so it is not vendored. Point the config at a
checkout that has it. The wrapper itself is vendored locally:

```yaml
sonic_planner_onnx: "{REMOTE_ROOT}/../others/GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"
sonic_planner_version: 2
pico_server_ip: "127.0.0.1"
pico_server_port: 5556
pico_feedback_port: 5558
```

The reviewed hardware configuration uses `sonic_planner_backend: tensorrt`,
FP32, and TF32 disabled. The robot-facing launcher fails before auxiliary
streams or DDS if the planner is not actually direct TensorRT or its serialized
engine is missing. `auto` and explicit ONNX Runtime remain offline-analysis
options only; confirm the printed provider, engine path, and logged planner
latency before any later hardware gate.

Port 5557 is already used by this project's local tactile server. Start the
official PICO manager with its feedback subscriber pointed at the local
deployment publisher instead:

```bash
# If PICO and deployment run on the same host:
--zmq_feedback_host 127.0.0.1 --zmq_feedback_port 5558
# Otherwise replace 127.0.0.1 with the deployment computer's reachable IP.
```

When entering `PLANNER_VR_3PT`, the official manager polls `g1_debug`, receives
the robot's measured 23-joint posture expanded to the 29-joint MuJoCo layout
(the six unavailable joints are zero), and recalibrates wrist FK before sending
new VR targets. Do not use `--no-feedback` for VR3PT.
Start deployment and wait for feedback publication before clicking into
VR3PT. If VR3PT was already active, leave it and re-enter it; then verify the
PICO console reports measured-pose wrist recalibration rather than zero-q
fallback before moving either arm.

## Recorded per step

Run with `--auto-record` or press X. Each 50 Hz row includes commanded and
measured 23/29-joint positions, tracking error, unclipped/clipped target,
saturation flags, raw 29-D SONIC action, unavailable-joint actions, token,
full encoder/decoder inputs, encoder/decoder/planner timing, LowState age, PICO
age/mode, motor temperature, tactile, colour, and depth. Full tensor rows add
roughly 0.55 MB/s, so keep initial diagnostic episodes short.

Analyze one on the actual deployment PC with:

```bash
python scripts/analyze_sonic_episode.py episodes/episode_001
```

The report separates encoder/decoder/planner inference from loop-period and
LowState-age jitter, and also reports tracking error, pelvis roll/pitch,
command limiting, and the unavailable waist/wrist action demand. It cannot
derive true one-way Ethernet latency because the robot and PC packets do not
yet carry a synchronized timestamp.

## Status

The implementation is statically checked against NVIDIA commit
`1983e88888217f6c69283cf3a9d1af01e87f07af`. It still requires a supported,
tethered live-hardware validation before locomotion; static parity is not a
safety certification.
