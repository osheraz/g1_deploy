# G1 deployment

RISE recovery, SONIC tracking and VIGOR visual control. Run commands from the
parent HV2 checkout; install with `python -m pip install --no-deps -e . -e ./g1_deploy`.

| Task | Entry point | Guide |
| --- | --- | --- |
| RISE on the robot | `rise/deploy.py` | [Deploy](docs/rise/deploy.md) |
| RISE in MuJoCo | `rise/sim.py` | [Simulation](docs/rise/sim.md) |
| RISE recordings | `rise/replay.py` | [Replay](docs/rise/recording.md) |
| SONIC | `sonic/deploy.py`, `sonic/sim.py` | [Joystick](docs/sonic/joystick.md) |
| VIGOR | `vigor/deploy.py` | [VIGOR](docs/vigor.md) |

```bash
python g1_deploy/rise/deploy.py --help
python g1_deploy/rise/sim.py --help
python g1_deploy/rise/replay.py --help
```

Start with the [code map](docs/architecture.md) to review or extend the code.
[Setup](docs/setup/README.md) · [Operator guides](docs/README.md).

RISE bundles and recordings are external assets. SONIC's retained models and
motion clips live in `sonic/assets/`; fetch its LFS weights with `git lfs pull`
inside this submodule.
