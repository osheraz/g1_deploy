# g1_deploy

Remote-side G1 deploy runtime. Runs a trained policy on your machine and drives the robot
over Unitree DDS.

## Start here

**[tutorials/GETTING_STARTED.md](tutorials/GETTING_STARTED.md)** — ethernet to the robot,
environment, camera, stream checks, first run.

For SONIC, first read the canonical
[`deployment learning guide`](../docs/learning/sonic_deployment.md), then use
the operator launchers under [`sonic/workflows/`](sonic/workflows/README.md). The generic
legacy policy commands below are not the reviewed robust 23-DoF SONIC preset.

## Tutorials

- [Getting started](tutorials/GETTING_STARTED.md) — index
- [1. Network](tutorials/01_network.md) — ethernet to the robot
- [2. Install](tutorials/02_install.md) — uv, torch, Unitree SDK
- [3. Config](tutorials/03_config.md) — configs and what to change
- [4. Cameras](tutorials/04_cameras.md) — viewing RGB and depth
- [5. Run](tutorials/05_run.md) — stream check, then the policy
- [6. Tactile](tutorials/06_tactile.md) — e-skin over ZMQ
- [7. 29-DoF Mid-360 fall recovery](tutorials/07_lidar_fallrec.md) — LiDAR/IMU
  streams, checkpoint contract, and no-publish validation
- [PICO teleop and PLANNER/G1](tutorials/PICO_TELEOP.md) — VR teleop and
  planner-controlled locomotion
- [SONIC parity changes and risk ledger](DEPLOYMENT_PARITY_CHANGES.md) — historical 23-DoF implementation ledger; current method is the learning guide
- [WiFi hotspot](tutorials/wifi_hotspot.md) — robot with no wired internet

## Run

Read-only stream check — never sends robot commands:

```bash
python sonic/check_streams.py --config g1_23dof.yaml
```

Policy — **moves the robot on launch**:

```bash
python deploy_g1.py --net enp132s0 --config g1_23dof.yaml --log TEST.npz
```

SONIC planner-only G1-encoder profile — **moves the robot** after the normal
start gate; run the official PICO manager in PLANNER mode first:

```bash
python deploy_eskin.py --config g1_23dof_sonic.yaml --mode planner \
  --net enp132s0 --auto-record --no-camera --no-tactile
```

After a diagnostic episode, inspect timing, tracking, pelvis tilt, target
saturation, and unavailable 29-DoF waist demand:

```bash
python sonic/scripts/analyze_sonic_episode.py episodes/episode_001
```

## Layout

| Path | What |
| --- | --- |
| [deploy_g1.py](deploy_g1.py) | VIGOR reactive policy runtime |
| [deploy_vigor2.py](deploy_vigor2.py) | VIGOR2: diffusion planner -> z_g -> tracker (`--cfg vigor2/fallrec.yaml`; `--no-publish` = full flow, no LowCmd) |
| [deploy_eskin.py](deploy_eskin.py) | SONIC teleop + tactile + episode recording |
| [sonic/check_streams.py](sonic/check_streams.py) | read-only stream checker |
| [configs/](configs/) | per-robot yamls (`g1_23dof_sonic.yaml` for both VIGOR2 teachers) |
| [core/](core/) | config loader, DDS interface, camera / depth / viser / logging |
| [vigor/](vigor/) | VIGOR policy runner + student model |
| [vigor2/](vigor2/) | planner (own process), tracker, runner, watchdog robot, `fallrec.yaml` |
| [sonic/model/](sonic/model/) | SONIC tracker + vendored ONNX |
| [sonic/common/](sonic/common/) | tactile, gamepad, SONIC-only clients |
| [sonic/workflows/](sonic/workflows/) | canonical operator launchers grouped by operation |
| [sonic/tools/](sonic/tools/) | offline analysis and visual replay; no robot command publishing |
| [sonic/scripts/](sonic/scripts/README.md) | compatibility entrypoints and specialized utilities |
| [sonic/experiments/](sonic/experiments/README.md) | controlled multi-run studies |
