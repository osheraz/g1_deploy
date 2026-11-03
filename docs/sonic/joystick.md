# SONIC

Select the model (`low-latency`, `original-release`, `sonic-v1.1`) and embodiment
(`g1_23dof`, `g1_29dof`) independently. Model files remain paired with their
observation layout. See [models](models.md) and [adaptations](adaptations.md).

Python entry points:

```bash
python g1_deploy/sonic/sim.py --help
python g1_deploy/sonic/deploy.py --help
```

The simulator uses `SonicRunner`, the same tracker as deployment.
`sonic/deploy.py --preflight-only` checks model engines before opening sensors or
DDS. Without that flag, the launcher can move the robot during startup.

Joystick scripts are under `sonic/scripts/joystick/`:

```bash
bash g1_deploy/sonic/scripts/joystick/mujoco.sh --model-profile original-release --embodiment g1_23dof --adaptation official-parity
bash g1_deploy/sonic/scripts/joystick/robot_cpp.sh --model-profile original-release --embodiment g1_23dof --adaptation official-parity
```

These scripts require the separate `sonic_cpp` checkout beside `g1_deploy`.
The C++ command defaults to preflight; `--arm-real-robot` enables its physical
path. Never run two motor publishers together.

[PICO](pico.md) · [Recording/replay](replay.md) · [Calibration](calibration.md).
