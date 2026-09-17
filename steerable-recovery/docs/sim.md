# MuJoCo

Run from `/home/osher/hv2` with a graphical display. No robot connection is needed.
These live commands do not record video or impose a ten-second cutoff.

## Flat, fixed starting joints

Choose one; replace `prone` with `supine` for face-up recovery:

```bash
.venv/bin/python g1_deploy/steerable-recovery/sim.py --preset prone --model r8_s25_diffusion --seed random --replan-hz 2.5
.venv/bin/python g1_deploy/steerable-recovery/sim.py --preset prone --model r8_s50r_diffusion --seed random --replan-hz 5
```

These presets use saved resting states and show depth. The robot presets use the
paired saved joint targets. [Which poses they select](examples.md).

## Terrain and free starts

```bash
.venv/bin/python g1_deploy/steerable-recovery/sim.py \
  --model r8_s25_diffusion --backend tensorrt --replan-hz 2.5 \
  --terrain fork3 --start prone --seed random --viewer
```

| Control / flag | Effect |
| --- | --- |
| R | Repeat the episode |
| N | New noise and nearby start; `--noise-only` keeps the start fixed |
| M | Sample another terrain |
| Space | Pause/resume |
| `--terrain flat` | Flat floor |
| `--terrain stairs --terrain-seed 43` | Fixed cached stair tile |
| `--replan-hz 5` | Change planning cadence; tracker remains 50 Hz |
| `--noise-policy fixed_episode` | Reuse episode noise instead of fresh noise every plan |
| `--start-state PATH.json` | Explicit saved simulator start |
| `--robot PATH.urdf` | Override the robot asset |
| `--no-depth-view` | Hide depth window |

Depth: near black, far white, invalid purple. Fork3 uses its corrected terrain
meshes and the shared training Warp depth renderer. The plant is built from the
training URDF. MuJoCo physics and synchronous planner execution still differ from
Isaac and asynchronous hardware deployment.

Starts settle before policy time begins. Default tracker startup repeats the
measured initial state; depth uses `prefill`. The startup limiter is active only
for 0.5 s. No camera delay is injected by default.

Headless, without video:

```bash
.venv/bin/python g1_deploy/steerable-recovery/sim.py \
  --model r8_s25_diffusion --backend tensorrt --terrain flat \
  --start supine --seed 42 --seconds 10 --no-depth-view
```

Add `--video` only when an MP4 is wanted. [Steering commands](steering.md).
