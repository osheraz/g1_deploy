# Terrain steering — September 21 gallery

Current selected S50R commands: **[Lab steering + Y](lab-steering.md)**.
The commands below describe the older September 21 set; append
`--catalog g1_deploy/steerable-recovery/config/steering/terrain_gallery/catalog.json`
to use them. The default gallery now contains the 20 selected lab runs.

Run from `/home/osher/hv2`. Each `--gallery` name selects its original model,
tracker, command, guidance strength and starting scene. No model flag is needed.
The existing unguided `s25r_diffusion` and older `--route` commands remain available.

## Robot: supine 01, supine 03 and prone 03

Supine means **face up**; prone means **face down**. Place the robot manually
before launch. Preserve the same body placement, heading and support for both
directions of a pair. A starts from the measured pose; selecting a gallery name
does not reposition the body or move the joints automatically.

| Pair | Physical setup | Planner | Left / right source guidance |
| --- | --- | --- | --- |
| `supine_01` | Face up on the long **60 × 36 × 9-inch** platform; back/pelvis supported on top, feet near its edge | S50R_INV10 | Full pelvis pose / 200; XYZ + terminal heading / 600 |
| `supine_03` | Face up at the edge of a **36 × 24 × 9-inch** block; upper back supported, legs extend onto the floor | S50R_GEOMETRY | Full pelvis pose / 600; XYZ + terminal heading / 600 |
| `turn_03` | Face down at the short block; upper body supported, legs toward the floor | S50R_CAUSAL | XYZ + terminal heading / 200 on both sides |

Original start images and videos are local under
`logs/native_recovery/terrain_turn_gallery_20260921/campaign/video_gallery/03_steering/`:
open `supine_01_left/preview_0.jpg`, `supine_03_left/preview_0.jpg`, or
`turn_03_left/preview_0.jpg`. Supine video files are in their corresponding
folders, e.g. `supine_01_left/supine_01_left.mp4`.

Keep Terminal 1 depth running as described below. In Terminal 2, run **one at a time**:

```bash
cd /home/osher/hv2
# Face up, long platform:
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery supine_01_left --feedback estimated
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery supine_01_right --feedback estimated

# Face up, short block:
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery supine_03_left --feedback estimated
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery supine_03_right --feedback estimated

# Face down, previous short-block pair:
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery turn_03_left --feedback estimated
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery turn_03_right --feedback estimated
```

All six optional joint presets pass the existing hardware bounds. Append
`--position-start` only if you want **START** to move to the recorded joint angles;
wait for positioning and inference readiness, then press **A**. Without that flag,
wait for **Live inference ready** and press **A** directly. Pose positioning
does not establish the intended body support or contacts.

Defaults are 2.5 Hz replanning, fresh noise per plan, no joint jitter and the
source seed (43021 for supine; 32021 for prone). Append `--seed random` for a
different seed on each launch. The left/right guidance types and gains above
are deliberately preserved from Stanley; these are not direction-only ablations.
Guided planning uses PyTorch gradients with the TensorRT tracker. All three
planners use the same P42 tracker, so changing planners does not resolve the
ankle-feedback issue observed in the last robot run.

The four new supine lanes were imported from the shared manifest and task files;
Git guide commit `3e4cceb4`. Their source files and media are cached locally, with
the import receipt under `supine_update_20260921/` in the gallery cache. Both
directions have identical recorded starts within each pair. Existing prone
commands and runtime behavior are unchanged. No new simulator execution is required.

## MuJoCo first

Start with the **single-block prone pair**, `turn_03`:

```bash
cd /home/osher/hv2
.venv/bin/python g1_deploy/steerable-recovery/sim.py --gallery turn_03_left
.venv/bin/python g1_deploy/steerable-recovery/sim.py --gallery turn_03_right
```

Run one at a time. The live viewer and depth open; no video is recorded and there
is no automatic duration limit. **N** changes planner noise with the same start;
**R** repeats. This imports the recorded joints, base pose, velocities and terrain
tile. Physics uses nominal MuJoCo dynamics, not the original Isaac batch/DR.

Both source-seed runs finished upright in the 10-second checks and turned in
opposite directions. Their XYZ path tracking was imperfect. The staggered
`turn_06_left` / `turn_06_right` runs recovered but missed the requested steering;
keep those as experimental comparisons.

```bash
# All imported individual examples, with model, terrain, posture and gain:
.venv/bin/python g1_deploy/steerable-recovery/sim.py --gallery list

# Explicit variation; the recorded start stays fixed:
.venv/bin/python g1_deploy/steerable-recovery/sim.py --gallery turn_03_left --seed random --replan-hz 2.5
```

Defaults: source seed **32021**, fresh noise each plan, **2.5 Hz**, DDIM8 per
stage, EMA, TF32, no blending. The gallery models have a **1-second forecast**.
`--replan-hz` and `--guidance-scale` are explicit changes to the source recipe.

## Robot

**Terminal 1:** start [the depth server and SSH forwards](deploy.md#terminal-1-depth).
The September 21 checks left no camera server or robot controller running.

**Terminal 2, sensor-only check:** includes live DLIO, external camera, depth,
raw sensors and model traces; sends no motor commands.

```bash
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery turn_03_left --sensor-only --feedback estimated --seconds 20
```

**Terminal 2, recovery:** choose one direction. Inspect the original start image
and place the robot face down with matching block support and heading.
The single-block terrain is 9 inches high. Joint targets alone do not recreate
the base placement or contact support.

```bash
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery turn_03_left --feedback estimated
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery turn_03_right --feedback estimated
```

1. An actuated launch enters zero torque. Wait for **Live inference ready**.
   First-time compilation can add a startup wait; it finishes before activation.
2. **A** starts from the measured pose and starts the episode recording.
   By default there is no saved joint-positioning step.
3. For optional saved joints, append **`--position-start`**. **START** then moves
   to the example's joints; **A** still works directly without START.
4. A again / SELECT / Ctrl+C stops and saves. Wait for the save confirmation.
   Runs last 60 seconds by default; change `--seconds` explicitly.

Of the original 25 poses, 17 satisfy the existing hardware preset bounds;
all four newly imported supine poses also pass, including both requested pairs.
Both `turn_03` poses pass. Optional START is rejected for `turn_01`,
`turn_02`, `turn_07` and `turn_08`; their simulator poses are preserved, not clipped.

Each actuated launch creates `logs/real_world/gallery_NAME_TIMESTAMP` with raw
sensors, full-resolution processed depth, external video, XYZ and model inputs.
Sensor-only checks go under `logs/deployment_validation`.

```bash
.venv/bin/python g1_deploy/steerable-recovery/replay.py RUN_NAME --speed 0.5
```

Open **http://localhost:8081**. Play and loop default to on.

## What is preserved, and what remains unverified

- Guided execution uses a **PyTorch planner for gradients and TensorRT tracker**.
  The tracker, PD binding, histories and 0.5-second startup limiter are unchanged.
- Goals use physical elapsed time and fixed heading axes at A. XYZ is pelvis
  displacement since A; Z is rise from the starting pelvis, not floor clearance.
- `combined` is XYZ trajectory plus final heading at every replan. `turn_06_right`
  uses `waypoint_pose`: XYZ plus full pelvis orientation. These existing shared
  cost functions and all source targets/gains are unchanged. The source schedule
  retains its final target, but **robot deployment releases steering once an
  upright standing posture is confirmed** and stays unguided until the next run.
  Detection uses pelvis IMU upright cosine >= 0.9, both hip-pitch magnitudes
  <= 0.7 rad and both knee magnitudes <= 1.0 rad continuously for 0.2 s.
  It does not depend on DLIO or a floor-height estimate. The tracker keeps running
  while the first unguided plan is computed; late guided replies are discarded.
  The console prints `Upright pose confirmed`; telemetry records the release.
  This is a posture trigger, not a guarantee of stable standing.
- `--feedback estimated` explicitly uses DLIO without independently verified
  clock bounds or dynamic accuracy. Invalid feedback suspends guidance while
  recovery continues. Do not interpret a successful sensor check as a successful
  steered hardware recovery.
- Two 20-second sensor-only checks completed with **zero motor commands**, no
  rejected plans and no recorder drops. Combined guidance had four control ticks
  with stale odometry across three brief events. Its planner latency was 135 ms
  median / 157 ms p95; full-pose guidance was 86 / 90 ms. Two saved tracker and
  two planner calls reproduced within replay tolerance, with unchanged source.
- Compilation speeds up the same combined cost; it does not change its objective.
  Focused tests: 48 passed. Evidence:
  `logs/deployment_validation/gallery_integration_20260921/verification.json`.

## Models and import

The six new model names are `s50r_diffusion`, `s50r_causal_diffusion`,
`s50r_geometry_diffusion`, `s50r_h3_diffusion`, `s50r_inv10_diffusion`, and
`s50r_recovery_diffusion`. All use the verified P42 depth tracker. They are
different checkpoints from the older `r8_s50r_diffusion`.

```bash
# Recreate the original 25 registrations from the initial source cache.
# This does not include the separately imported supine update; do not run it
# over the expanded catalog if you want to retain those four entries.
.venv/bin/python -m steerable_recovery.tools.import_gallery
```

Source cache: `logs/native_recovery/terrain_turn_gallery_20260921/`.
Imported bundle: `logs/native_recovery/bundles/stage3_40k_20260921/`.
Targets and starts: `g1_deploy/steerable-recovery/config/steering/terrain_gallery/`.
Weights and terrain remain outside Git. New unguided TensorRT planner engines
have not been exported; the gallery commands use the guided backend described above.

Exact Isaac reproduction is separate: use [Stanley's guide](https://github.com/osheraz/hv2/blob/7a7ccd2f/commands/motion_diffusion/terrain_turn_gallery_replay.md)
with its complete original batch, validation dataset and runtime. That dataset
and Isaac 6 container are not in this local cache; no new Isaac replay is claimed.
