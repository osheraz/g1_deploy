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

Set `sonic_planner_onnx` to the installed SONIC planner. Build its TensorRT
engine on the deployment GPU; the hardware launcher requires direct TensorRT,
FP32 and TF32 disabled. See [models](models.md).

Configure the PICO manager with `--zmq_feedback_host <deployment-ip>
--zmq_feedback_port 5558`. Port 5557 belongs to tactile input. Start deployment
before entering `PLANNER_VR_3PT`, then verify measured-pose wrist recalibration
in the manager. Keep feedback enabled for VR3.

## Frames and scheduling

PICO movement and facing vectors already use the planner session frame; they
must not be rotated by robot yaw again. Enter PLANNER after deployment is ready
so both agree on the session's forward direction. Scripted and Unitree-remote
inputs declare local framing and are converted once in `pico_reference.py`.

A valid `vr_position` selects the teleop encoder in pico mode. Navigation-only
messages select G1. Planner mode always selects G1 and ignores VR fields.
After one second without a fresh complete message, commands transition to IDLE
while preserving facing. VR endpoints are expressed in the pelvis frame.

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
