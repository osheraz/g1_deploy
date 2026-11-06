# PICO teleoperation

`--mode planner` uses navigation commands and the G1 encoder. `--mode pico`
also accepts VR3 upper-body targets and selects the encoder from each message.

```bash
python -m sonic.inputs.pico_client --server <publisher-ip>
python g1_deploy/sonic/deploy.py --config g1_23dof_sonic.yaml \
  --mode planner --net enp132s0 --auto-record --no-camera --no-tactile
```

The deployment command moves the robot during startup. Use `--preflight-only`
to check the model engines without opening DDS or sensors.

## Configuration

```yaml
pico_server_ip: "127.0.0.1"
pico_server_port: 5556
pico_feedback_port: 5558
sonic_planner_version: 2
sonic_planner_backend: tensorrt
```

Set `sonic_planner_onnx` to the installed planner. Build its TensorRT engine on
the deployment GPU. Hardware requires direct TensorRT in FP32, with TF32 disabled.
See [models](models.md).

Configure the PICO manager with `--zmq_feedback_host <deployment-ip>` and
`--zmq_feedback_port 5558`. Port 5557 is for tactile input.
Start deployment before entering `PLANNER_VR_3PT`. Verify wrist recalibration
from the measured pose, and keep VR3 feedback enabled.

## Frames and scheduling

Enter PLANNER after deployment is ready so both use the same forward direction.
PICO movement and facing already use this frame; do not rotate them by robot yaw
again. Scripted and Unitree-remote inputs use local frames and are converted once
in `pico_reference.py`.

In pico mode, a valid `vr_position` selects the teleop encoder. Navigation-only
messages select G1. Planner mode always uses G1 and ignores VR fields.

After one second without a fresh complete message, commands switch to IDLE and
keep the facing direction. VR endpoints use the pelvis frame.

## Message and code

The `planner` topic carries a 1280-byte JSON header and a little-endian payload:
`mode`, `movement`, `facing`, `speed`, `height`, plus optional upper-body,
hand and VR3 targets. VR quaternions are WXYZ; speed/height -1 selects defaults.

| Operation | Code |
| --- | --- |
| Receive commands | `sonic/inputs/pico_client.py` |
| Select mode and reference window | `sonic/pico_reference.py` |
| Generate motion | `sonic/model/planner_onnx.py` |
| Publish measured posture | `sonic/sensing/robot_feedback.py` |
| Encode and track | `sonic/sonic_runner.py` |

Record with `--auto-record` or X. Analyze timing, tracking error, command limits
and motor telemetry with:

```bash
python g1_deploy/sonic/scripts/analysis/analyze_sonic_episode.py episodes/episode_001
```
