# G1 deployment

Run commands from the parent HV2 checkout. Use its pinned deployment revision
and install once with `python -m pip install --no-deps -e ./g1_deploy`.

| Task | Guide | Entry point |
| --- | --- | --- |
| RISE recovery, steering and recording | [Recovery](docs/recovery/deploy.md) | `steerable-recovery/deploy.py` |
| Recovery in MuJoCo | [Simulation](docs/recovery/sim.md) | `steerable-recovery/sim.py` |
| Replay a recording | [Replay](docs/recovery/recording.md) | `steerable-recovery/replay.py` |
| SONIC tracking and teleoperation | [SONIC](docs/sonic/joystick.md) | `sonic/deploy.py`, `sonic/sim.py` |
| VIGOR reactive policy | [VIGOR](docs/vigor.md) | `vigor/deploy.py` |
| Network, cameras and installation | [Setup](docs/setup/README.md) | — |

Shared robot I/O, sensors, recording and viewers live in `core/`.
[Code ownership](docs/architecture.md) · [All guides](docs/README.md).
Model weights and recordings remain outside Git.
