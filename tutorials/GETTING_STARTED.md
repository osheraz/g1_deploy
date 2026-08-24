# Getting started

In order. Each ends with a check you can run.

| | | |
| --- | --- | --- |
| 1 | [Network](01_network.md) | ethernet to the robot, static IP, robot + internet on one port |
| 2 | [Install](02_install.md) | uv venv, torch, Unitree SDK + CycloneDDS |
| 3 | [Config](03_config.md) | which config, what to change, depth as policy input |
| 4 | [Cameras](04_cameras.md) | viewing the RGB and depth streams |
| 5 | [Run](05_run.md) | stream check, then the policy |

Optional:

| | |
| --- | --- |
| [Tactile](06_tactile.md) | e-skin over ZMQ |
| [PICO teleop](PICO_TELEOP.md) | VR teleop from a PICO headset |
| [WiFi hotspot](wifi_hotspot.md) | robot with no wired internet |

Nothing sends robot commands until [5. Run](05_run.md). `sonic/check_streams.py` is read-only.

## What talks to what

| Piece | Where it runs | Transport |
| --- | --- | --- |
| `deploy_g1.py` (policy) | your machine | Unitree DDS over ethernet |
| robot state / motor commands | G1 onboard | DDS `rt/lowstate`, `rt/lowcmd` |
| RealSense colour + depth | G1 onboard PC `192.168.123.164` | ZMQ port 5556 |
| tactile | G1 onboard, system python | ZMQ port 5557 |
