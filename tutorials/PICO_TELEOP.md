# PICO teleop

Live VR teleop drives the SONIC tracker. Run with `deploy_eskin.py --mode pico`.

## Pieces

| | |
| --- | --- |
| PICO publisher | separate process owning the XR SDK; publishes the `planner` ZMQ topic |
| [`common/pico_client.py`](../common/pico_client.py) | subscribes and decodes that topic |
| [`policy/pico_reference.py`](../policy/pico_reference.py) | drives the SONIC kinematic planner, builds the encoder window |
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

`pico_reference.py` feeds the planner directly:

| topic field | `SonicKinematicPlanner.update()` |
| --- | --- |
| `mode` | `mode` |
| `speed` | `target_vel` |
| `height` | `height` |
| `movement` | `movement_direction` |
| `facing` | `facing_direction` |

`gather(10, 5)` then returns the encoder's future window, and the VR endpoints
pass through in the pelvis frame (`vr_targets_local=True`).

## Setup

The planner ONNX is ~774 MB, so it is not vendored. Point the config at a
checkout that has it:

```yaml
sonic_repo: "~/tactile"
pico_server_ip: "127.0.0.1"
pico_server_port: 5560
```

## Recorded per step

`teleop_mode`, `teleop_movement`, `teleop_facing`, `teleop_speed`,
`teleop_height`, `teleop_left_hand`, `teleop_right_hand`, `teleop_vr_pos`,
`teleop_vr_quat_wxyz` — alongside robot state, tactile, colour and depth.

## Status

Untested against a live publisher. The wire decoder is verified by round-trip
against NVIDIA's own encoder, and the planner call signatures are checked
against `sonic/planner/kinematic.py`, but no PICO headset has driven it yet.
