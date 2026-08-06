# g1_deploy

Remote-side G1 deploy runtime. Runs a trained policy on your machine and drives the robot
over Unitree DDS.

## Start here

**[tutorials/GETTING_STARTED.md](tutorials/GETTING_STARTED.md)** — ethernet to the robot,
environment, camera, stream checks, first run.

## Tutorials

- [Getting started](tutorials/GETTING_STARTED.md) — index
- [1. Network](tutorials/01_network.md) — ethernet to the robot
- [2. Install](tutorials/02_install.md) — uv, torch, Unitree SDK
- [3. Config](tutorials/03_config.md) — configs and what to change
- [4. Cameras](tutorials/04_cameras.md) — viewing RGB and depth
- [5. Run](tutorials/05_run.md) — stream check, then the policy
- [6. Tactile](tutorials/06_tactile.md) — e-skin over ZMQ
- [PICO teleop](tutorials/PICO_TELEOP.md) — VR teleop
- [WiFi hotspot](tutorials/wifi_hotspot.md) — robot with no wired internet

## Run

Read-only stream check — never sends robot commands:

```bash
python check_streams.py --config g1_23dof.yaml
```

Policy — **moves the robot on launch**:

```bash
python deploy_g1.py --net enp132s0 --config g1_23dof.yaml --log TEST.npz
```

## Layout

| Path | What |
| --- | --- |
| [deploy_g1.py](deploy_g1.py) | policy runtime |
| [check_streams.py](check_streams.py) | read-only stream checker |
| [config.py](config.py) | config loader |
| [configs/](configs/) | per-robot / per-policy configs |
| [policy/](policy/) | policy runner, reference providers, ONNX |
| [robot/](robot/) | DDS interface |
| [common/](common/) | camera, depth, tactile, viser, logging |
| [tools/](tools/) | replay, video, MuJoCo sim2sim |
