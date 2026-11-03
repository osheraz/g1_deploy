# Steerable Recovery

| Command | Guide |
| --- | --- |
| `python g1_deploy/steerable-recovery/deploy.py` | [Robot deployment](../docs/recovery/deploy.md) |
| `python g1_deploy/steerable-recovery/sim.py` | [MuJoCo](../docs/recovery/sim.md) |
| `python g1_deploy/steerable-recovery/replay.py RUN_NAME` | [Recording/replay](../docs/recovery/recording.md) |

[Models and starts](../docs/recovery/models.md) ·
[Lab steering](../docs/recovery/lab-steering.md) ·
[Y standing](../docs/recovery/standing-hold.md) ·
[Code ownership](../docs/architecture.md).

Run from the HV2 checkout. Prepare a paired model bundle with the
[release guide](../../docs/guides/deployment.md).
Direct deployment requires `--actuate` for motor output; named robot presets
already enable it. Use `--print-command` to inspect a preset without launching.
