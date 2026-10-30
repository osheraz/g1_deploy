# Steerable Recovery

Run from the hv2 checkout with its pinned `g1_deploy` submodule. The supported
models are `ours`, `joint`, `depth_blind`, and `blind_blind`; the default is `ours`.
The [release guide](../../docs/recovery/deployment.md) identifies the verified
bundle and explains how to package another published pair. Joint uses 8 DDIM steps.

| Task | Guide | Entry point |
| --- | --- | --- |
| Robot setup and model checks | [Deploy](docs/deploy.md) | `deploy.py` |
| MuJoCo and recorded starts | [Simulation](docs/sim.md) | `sim.py` |
| Goal inputs and feedback | [Steering](docs/steering.md) | `--goal-json` |
| Encoded standing reference | [Standing mode](docs/standing-hold.md) | `--standing-hold` |
| Sensor/model replay and video | [Recording](docs/recording.md) | `replay.py` |
| Runtime ownership | [Architecture](docs/architecture.md) | — |
| Timing measurements | [Performance](docs/performance.md) | — |
| Sensor troubleshooting | [Status](docs/status.md) | — |

Install the package from the hv2 root:

```bash
python -m pip install --no-deps --no-build-isolation -e ./g1_deploy
python g1_deploy/steerable-recovery/deploy.py --model ours --check --max-plan-age-ticks 10
```

Configuration lives in `config/models.yaml`. Sensor transport, recording, replay,
and diagnostics live in `steerable_recovery/`; learned inference lives in
`motion_diffusion/behavioral/`. The checkpoint determines inference, gains and
joint ordering. Compiled engines require their own export validation.

[Environment/network setup](../tutorials/GETTING_STARTED.md) covers a new machine.
The current model bundle is not hardware-qualified. No preset implicitly enables
motor output; actuation requires an explicit operator command.
