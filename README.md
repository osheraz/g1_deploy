# G1 deployment

Start with [the diffusion deployment guide](tutorials/deploy_diffusion.md).
It covers simulation preview, real-sensor checks, PyTorch/ONNX/TensorRT, and the
separate actuation gate. [Current status](tutorials/deployment_status.md) records
what has actually been verified.

| Entrypoint | Purpose |
| --- | --- |
| `sim_diffusion.py` | Flat-ground MuJoCo comparison: diffusion full horizon and flow replanning, live viewer or headless videos. [Guide](tutorials/sim_diffusion.md). No hardware transport. |
| `benchmark_diffusion.py` | Offline inference parity, warmed latency and concurrent planner/tracker benchmark. [Guide](tutorials/inference_latency.md). |
| `evaluate_recovery.py` | Matched ground-posture/seed comparison across PyTorch, TRT FP32 and experimental TRT TF32. |
| `compare_diffusion_samples.py` | Identical resting pose and sensing, different planner seeds: sampled Z plots and a simultaneous recovery video grid. |
| `deploy_diffusion.py` | Current diffusion/flow planner + paired action decoder. `--actuate`: initially idle, A starts policy, A again stops. START optionally moves to default before starting; SELECT stops. Without `--actuate`, no motor output. |
| `zero_torque.py` | Release the built-in controller and continuously send zero torque, stiffness and damping. Support the robot and stop other motor publishers first. Ctrl+C stops publication. |
| `sonic/check_streams.py` | Read-only robot and camera Viser viewer. No policy or motor commands. |
| `deploy_eskin.py` | Separate SONIC teleoperation and sensor recording workflow. Can actuate. |
| `deploy_vigor.py` | Legacy VIGOR reactive policy. **Moves during startup.** |
| `deprecated/` | Archived VIGOR2 runtime, configs and historical guides. Not the current diffusion pipeline. |

## Robot setup

- [Getting started](tutorials/GETTING_STARTED.md)
- [Network / new G1-29 SSH](tutorials/01_network.md)
- [Environment](tutorials/02_install.md)
- [Model bundle and hardware configuration](tutorials/03_config.md)
- [Camera and read-only Viser commands](tutorials/04_cameras.md)
- [Onboard migration record](tutorials/08_robot_migration.md)

SONIC is separate: use [its operator workflows](sonic/workflows/README.md).
Optional [tactile](tutorials/06_tactile.md), [PICO teleop](tutorials/PICO_TELEOP.md)
and [WiFi setup](tutorials/wifi_hotspot.md) are not prerequisites for diffusion.

## Code layout

- `deploy_diffusion.py` enters the **same inference core as simulation**:
  `../motion_diffusion/behavioral/hardware/`. It is not a copy or a VIGOR2 wrapper.
- `core/` holds reusable configuration, DDS, sensor, viewer and logging modules.
  Their presence does not mean the current policy imports every one of them.
- `sonic/` holds SONIC-specific tools and models.
- `vigor/` supports the explicitly named legacy `deploy_vigor.py` launcher.
- [`deprecated/README.md`](deprecated/README.md) explains the archived code.

Deployment commands and status live here, not in the parent repository's
teacher-training command directory.
