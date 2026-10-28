# Steerable Recovery

Run commands from the **hv2 root**, with its pinned `g1_deploy` submodule.

| Task | Guide | Entry point |
| --- | --- | --- |
| Robot recovery, depth and recording | [Deploy](docs/deploy.md) | `deploy.py` |
| New S25R40k + P42/22k pair | [S25R](docs/s25r.md) | `--model s25r_diffusion` |
| MuJoCo, terrain and starting poses | [Simulation](docs/sim.md) | `sim.py` |
| Pelvis-trajectory steering | [Steering](docs/steering.md) | `deploy.py --route` / `sim.py --route` |
| Selected measured-lab S50R pairs + Y | [Lab steering](docs/lab-steering.md) | `deploy.py --gallery` |
| Legacy terrain steering catalog | [Gallery commands](docs/terrain-steering.md) | `deploy.py --gallery` / `sim.py --gallery` |
| Replay, video, depth and model checks | [Recording](docs/recording.md) | `replay.py` |
| Compare saved starting poses | [Examples](docs/examples.md) | `--preset prone` / `--preset supine` |
| Depth faults and remaining validation | [Status](docs/status.md) | — |
| Code layout and sensor pipeline | [Architecture](docs/architecture.md) | — |
| Timing and benchmarks | [Performance](docs/performance.md) | — |
| Old commands and collaborator update | [Migration](docs/migration.md) | — |

After updating this checkout, register the relocated Python package once:

```bash
cd /home/osher/hv2
.venv/bin/python -m pip install --no-deps --no-build-isolation -e ./g1_deploy
```

New machine: [environment/network setup](../tutorials/GETTING_STARTED.md).
Configuration: [`config/models.yaml`](config/models.yaml). Always specify the
model in an experiment; the full launcher's historical default is P11.

```text
steerable-recovery/
  deploy.py  sim.py  replay.py       public commands
  config/                           model settings, poses, routes, LiDAR
  docs/                             short task guides
  setup/                            sensor workspace build
  steerable_recovery/                importable Python package
    cli/  presets/                  runtime launchers and saved examples
    sensing/  recording/  review/    streams, capture, replay
    tools/                          diagnostics, experiments, media, benchmarks
    paths.py  settings.py           shared paths and configuration loading
```

The learned runtime stays in `hv2/motion_diffusion/behavioral/`.
Shared robot I/O, recording formats and Viser live in `g1_deploy/core/`.
SONIC, VIGOR and deprecated tools retain their own workflows.
