# Measured lab: selected S50R pairs + manual Y standing

Current `--gallery` includes **robust_set_01 + robust_set_02 + left_right_qualified_01 + progress_recipe_03 + progress_recipe_04 + lab_terrain_7sec_01**: 32 runs in fifteen
same-start pairs plus two singles, all supine (face up), using **S50R 40k + P42 22k**. Imported
from `gallery/terrain-turn-replay` at `255e200f`; timed implementation `b9f2f7e`,
progress implementation `bfa03ef`. Runs 01–20 retain their original commands;
runs 21–24 select recipe 03. Runs 25–28 retain seven-second timed steering.
Runs 29–30 add recipe 04 at the lower-right edge. Singles 31–32 move that joint
pose to distinct upper-platform placements; 32 also rotates the root orientation.
The earlier progress recipe is superseded under the same 21–24 aliases.
The older September 21 catalog is retained separately.

The 32 retained recipes are separate from the published `ours`, `joint`,
`depth_blind`, and `blind_blind` pairs. Inputs retain their original identities;
these historical checks do not qualify the integrated runtime on hardware.

## Setup and controls

1. Keep [Terminal 1: depth](deploy.md#terminal-1-depth) running. The commands below
   use the existing `vigor_legacy` processing; server and client must agree.
   Close any standalone external-camera preview before launch so the recorder
   can use the same USB camera.
2. Match the **body placement and support** in the start image. Keep them the same
   between the two commands of a pair. Selecting a run does not move the body.
3. Launch one command. It starts in zero torque. Wait for live inference, depth,
   and steering feedback readiness, then press **A**. START is optional.
4. Once the robot has recovered to standing, press **Y**. The original P42 encoder
   supplies standing intent with a **0.5-second blend**. Live tracker feedback
   continues. Repeated Y does not restart the blend. This is not a world-position
   lock; it does not freeze the joints or automatically recover after a later fall.
5. **A again / SELECT / Ctrl+C** stops and damps. Wait for recording to finish.

For **pairs 01–10**, guidance uses the complete sparse **XYZ + quaternion** trajectory in the initial
pelvis heading frame, including pitch and roll. Z is displacement from the initial
pelvis, not height above the floor. Predicted waypoints after **3 seconds** are
excluded from the cost; new planning becomes unguided at that cutoff. These
presets disable the older early-upright guidance cutoff. **Y is separate and manual**:
it can interrupt planning before or after 3 seconds. Pairs 11–12 and 15, and singles 31–32, use the progress
rule below instead of this timed cutoff. Pairs 13–14 use **7 seconds**, with
XYZ + orientation guidance; they do not use roll-triggered release.

Depth, external camera, joints/IMU, live DLIO, model inputs/noise/actions and the Y
handover are recorded under a unique `logs/real_world/gallery_NAME_TIMESTAMP/`.
Sensor recording starts at A. `--live-odometry` starts the existing XYZ pipeline;
no additional DLIO terminal is needed. `--feedback estimated` retains the current
measured-feedback setup; it does not certify its clock or position accuracy.

## New: single 32 right — pair 15 pose at pair 05 placement

Match the [run 32 start image](../../../logs/native_recovery/custom_lab_robust_20260922/starts/robust_32_right.jpg):
**face up at the upper-front interface**, using pair 15's joint pose at pair 05's
pelvis XY. The root orientation is rotated **−90° about world Z** from pair 15.
Nominal scene root XYZ is `[0.42, 1.0279, 0.6841994503]` m; these coordinates
identify the source placement, not a live absolute-position command.

```bash
cd /home/osher/hv2
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_32_right --feedback estimated
```

Keep depth running and stop the external-camera preview. Wait for live readiness;
**A starts; Y after get-up requests standing; A again / SELECT stops.** Recording
and live odometry are automatic. The run name selects the model and command;
it does not move the body into the starting placement.

Exact settings: S50R/P42, seed **54004**, gain **3500**, gradient cap **1**, position
scale **0.12 m**, fresh noise, **5 Hz**, no blend. Nine XYZ knots, **2–22 cm** look-ahead,
30 cm progress window, 25 cm corridor, advance bounded by twice measured movement.
Rightward side-roll fade starts at **30°**, releases at **45° for three observations**;
15% floor until release, applied after gradient clipping. No timed or orientation
cost; the next scheduled plan is unguided after release. Y stays manual.

The distinct reset is motion 304 in case 19 of the archived 48-case, 600-tick
simulation plan. All deployment assets are copied locally from the extra Turbo
volume. This is a selected single-seed example with no independent replay or
seed-consistency claim. The producer used a disclosed 45° demonstration criterion,
not its stricter 60° criterion; neither is a hardware qualification.

Single 31 (`robust_31_right`) is also retained from the same manifest: a different
upper-right placement, gain 3500 / seed 54004, fade 20° / release 60°.
Its [start image](../../../logs/native_recovery/custom_lab_robust_20260922/starts/robust_31_right.jpg)
and settings must not be substituted for run 32.

## New: recipe 04, pair 15 — lower-right edge

**Face up at `lab19`, on the lower platform's right edge** (motion 304, pose
template 641). This is a different initial body/joint configuration from pairs
11–12. Match the [new start image](../../../logs/native_recovery/custom_lab_robust_20260922/starts/robust_29_left.jpg),
and restore that same placement between left and right trials.

```bash
cd /home/osher/hv2

# Pair 15 left: seed 54001, no fade, release at 60 degrees
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_29_left --feedback estimated

# Pair 15 right: seed 54002, fade from 0 degrees, release at 45 degrees
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_30_right --feedback estimated
```

Run one at a time. Keep Terminal 1 depth running and stop the external-camera
preview before deploying. **A starts; Y after get-up requests standing intent;
A again / SELECT stops.** Recording and live odometry are automatic.

Same `bfa03ef` progress controller as recipe 03: S50R/P42, gain 2500, gradient
cap 1, fresh noise, 5 Hz replanning, no latent blend, XYZ-only cost, no timed
cutoff. Fade acts after clipping with a 15% floor until the three-observation
roll-release latch. The seven-second source path supplies geometry, not a deadline.

**Seeds differ between directions: 54001 left / 54002 right.** Preserve these
selected seeds for reproduction; this pair is not a same-seed command-only
comparison. Source selected-seed simulation checks are not hardware validation.
Pairs 1–14 are unchanged, including recipe 03 and the seven-second options.

## Next comparison: recipe 03 progress versus seven-second timed steering

Both versions use S50R/P42, seed 54001, fresh noise, **5 Hz replanning** and no
latent blend. Place the robot face up, match the linked image, and reposition
between trials. A starts from measured joints; `--position-start` optionally offers
START joint positioning. It cannot establish the required body support.

| Setup | Progress pair | Seven-second timed pair | Start image |
|---|---|---|---|
| Lower platform / floor | 11: `robust_21_left` / `robust_22_right` | 13: `robust_25_left` / `robust_26_right` | [Lower](../../../logs/native_recovery/custom_lab_robust_20260922/starts/robust_21_left.jpg) |
| Upper block / lower platform | 12: `robust_23_left` / `robust_24_right` | 14: `robust_27_left` / `robust_28_right` | [Upper](../../../logs/native_recovery/custom_lab_robust_20260922/starts/robust_23_left.jpg) |

Keep [Terminal 1: depth](deploy.md#terminal-1-depth) running; close the standalone
external-camera preview. Run **one command at a time**:

```bash
cd /home/osher/hv2

# Lower: recipe 03 progress, left / right
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_21_left --feedback estimated
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_22_right --feedback estimated

# Lower: seven-second timed, left / right
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_25_left --feedback estimated
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_26_right --feedback estimated

# Upper: recipe 03 progress, left / right
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_23_left --feedback estimated
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_24_right --feedback estimated

# Upper: seven-second timed, left / right
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_27_left --feedback estimated
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_28_right --feedback estimated
```

Remote **Y**, recording and live odometry are enabled automatically. Wait for
readiness; **A** starts, **Y after get-up** switches to standing intent, **A again /
SELECT / Ctrl+C** stops. Wait for recording finalization after stopping.

Recipe 03 projects the measured pelvis onto the current or next path segment,
inside a **30 cm forward arc window** and **25 cm corridor**. Progress is monotonic;
each advance is capped at twice the measured pelvis displacement. A stationary
robot cannot advance the path. Look-ahead remains **2–18 cm**. Knot timestamps
label geometry, not deadlines; the cost uses **XYZ only**, without orientation
cost, time cutoff, or a path-completion requirement for release.

Roll is measured about the **initial horizontal spine**, not the gravity-up
heading axis. Release requires three consecutive new control observations.

| Setup | Left: fade / release | Right: fade / release |
|---|---|---|
| Lower | No fade / 60° | Fade from 30° / release at 60° |
| Upper | No fade / 60° | Fade from 0° / release at 45° |

Fade decreases monotonically, retains a **15% floor until release**, and multiplies
the guidance update **after gradient clipping**. Release latches; the next
scheduled planner request is unguided. The queued plan may finish before that
boundary. Release does not activate standing mode: **Y remains manual**.
Missing position freezes progress and retains the existing feedback fallback.
Preview cannot latch release. The initial command frame stays anchored at A.

Seven-second pairs retain full sparse **XYZ + quaternion** targets and physical
time, with steering disabled at **7 seconds**. Pairs 3/4 remain available for the
older three-second comparison. This compares different recipes, not only clocks.
`steering_goal.json`, per-tick `steering_progress` (including fade weight), and
planner context snapshots record the exact settings and measured progression.

All required deployment assets are local. The producer reports exact selected-seed
Isaac replay and original motion-quality acceptance for all four recipe 03 clips;
the earlier relaxed upper-right acceptance no longer applies. This is not
arbitrary-seed, placement, or hardware qualification. Seven-second pairs remain
available as requested, with their own simulation evidence and no hardware claim.

## Choose one pair

The measured scene has a 22.86 cm lower platform and an upper block whose top is
50.038 cm above the floor. Exact geometry and source start images are local:

- `logs/native_recovery/custom_lab_robust_20260922/scene/scene_overview.png`
- `logs/native_recovery/custom_lab_robust_20260922/starts/NAME.jpg`
- `scene/lab_scene.usda` and `scene/lab_terrain.usda` must remain together.

| Pair | Placement / start image | Comparison | Planner schedule |
|---|---|---|---|
| 1 | `lab00`, lower-front platform/floor interface; `robust_01_minimal.jpg` | Minimal / right floor turn | 5 Hz, no blend |
| 2 | `lab26`, upper-front block/platform interface; `robust_03_minimal.jpg` | Minimal / right floor turn | **10 Hz, 2-tick (40 ms) latent blend** |
| 3 | `lab31`, upper-front block/platform interface; `robust_05_left.jpg` | Left / right side roll | 5 Hz, no blend |
| 4 | `lab01`, lower-front platform/floor interface; `robust_07_left.jpg` | Left / right side roll | 5 Hz, no blend |
| 5 | `lab25`, upper-front block/platform interface; `robust_09_minimal.jpg` | Minimal / right floor turn | 5 Hz, no blend |
| 6 | `deck10`, fully on lower deck, front; `robust_11_left.jpg` | Left / minimal side roll | 5 Hz, no blend |
| 7 | `deck26`, fully on lower deck, middle; `robust_13_minimal.jpg` | Minimal / right side roll | 5 Hz, no blend |
| 8 | `deck42`, fully on lower deck, right; `robust_15_left.jpg` | Left / minimal side roll | 5 Hz, no blend |
| 9 | `flat25`, unobstructed flat floor away from platforms; `robust_17_left.jpg` | Left / minimal side roll | 5 Hz, no blend |
| 10 | `flat43`, unobstructed flat floor away from platforms; `robust_19_left.jpg` | Left / right side roll | 5 Hz, no blend |
| 11 | `lab01`, lower-front platform/floor interface; `robust_21_left.jpg` | Progress-based left / right side roll | 5 Hz, no blend |
| 12 | `lab31`, upper-front block/platform interface; `robust_23_left.jpg` | Progress-based left / right side roll | 5 Hz, no blend |
| 13 | `lab01`, lower-front platform/floor interface; `robust_25_left.jpg` | Seven-second timed left / right side roll | 5 Hz, no blend |
| 14 | `lab31`, upper-front block/platform interface; `robust_27_left.jpg` | Seven-second timed left / right side roll | 5 Hz, no blend |
| 15 | `lab19`, lower-right platform/floor edge; `robust_29_left.jpg` | Recipe 04 progress left / right side roll | 5 Hz, no blend; seeds 54001 / 54002 |

Floor turn means rotation about gravity-up. Side roll means rotation about the
initial horizontal spine: chest toward the robot's left or right. **Minimal**
refers to the group's indicated axis: floor turn for pairs 1/2/5, side roll for
pairs 6–9. Normal pitching up is allowed. The new lower-deck starts are fully
supported on the exposed platform; they do not straddle its edge. Pairs have different
joint configurations: use the corresponding image rather than a generic supine pose.

Run **one line at a time**, repositioning between trials:

```bash
cd /home/osher/hv2

# Pair 1: lab00, lower front
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_01_minimal --feedback estimated
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_02_right --feedback estimated

# Pair 2: lab26, upper front
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_03_minimal --feedback estimated
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_04_right --feedback estimated

# Pair 3: lab31, upper front
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_05_left --feedback estimated
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_06_right --feedback estimated

# Pair 4: lab01, lower front
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_07_left --feedback estimated
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_08_right --feedback estimated

# Pair 5: lab25, upper front
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_09_minimal --feedback estimated
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_10_right --feedback estimated

# Pair 6: deck10, lower deck front — left / minimal side roll
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_11_left --feedback estimated
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_12_minimal --feedback estimated

# Pair 7: deck26, lower deck middle — minimal / right side roll
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_13_minimal --feedback estimated
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_14_right --feedback estimated

# Pair 8: deck42, lower deck right — left / minimal side roll
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_15_left --feedback estimated
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_16_minimal --feedback estimated

# Pair 9: flat25, flat floor away from platforms — left / minimal side roll
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_17_left --feedback estimated
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_18_minimal --feedback estimated

# Pair 10: flat43, flat floor away from platforms — left / right side roll
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_19_left --feedback estimated
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_20_right --feedback estimated
```

The presets retain source gains, source seeds, fresh noise per plan and their
individual cadence/blending. Y is enabled automatically. Guided planning uses
PyTorch gradients and the TensorRT tracker. A 60-second maximum duration applies;
stop earlier with the remote. No joint jitter is added.

Optional: `--position-start` offers the recorded joint pose on **START** before A;
it cannot place the pelvis or establish the intended terrain contacts.
`--seed random` changes the episode seed while preserving the command.
`--no-standing-hold` is only for a comparison without Y.

## Check and replay

```bash
# Inspect resolved command; no process is launched
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_04_right --feedback estimated --print-command

# Model/encoder check; no sensors or motors
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_04_right --check

# Live sensors / recording check; no motor takeover
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --gallery robust_04_right --feedback estimated --sensor-only --seconds 20

# After a trial, replace RUN_NAME with its printed folder name
.venv/bin/python g1_deploy/steerable-recovery/replay.py RUN_NAME --speed 0.5 --loop
```

Playback and looping default on; Viser is at `http://localhost:8081`.
Model weights, TensorRT exports/engines, preset commands, start poses, start images
and the measured scene are local. Local weight symlinks resolve to files under
`/home/osher/hv2` on local storage. Provenance retains original Turbo paths as
references; deployment does not load from them. Check again without mounting Turbo:

```bash
PYTHONPATH=g1_deploy/steerable-recovery .venv/bin/python -m steerable_recovery.tools.verify_lab_gallery
```

The file/hash inventory is
`logs/native_recovery/custom_lab_robust_20260922/LOCAL_ASSETS.json`.
This verifies deployment artifacts; the original 48-environment Isaac reproduction
still uses Stanley's cluster runtime and dataset.

## Historical checks (September 2026)

The current 32-run integration passed **44 focused checks**, covering source parity,
progress bounds, post-clip fade, release, timed cutoffs, Y, and local assets.
All 280 inventoried assets are local, including the extra-volume singleton inputs.
All previous 30 goal/start/joint-pose triples and run settings are unchanged.
Run 32 has its own reset, nine-knot geometry, fade/release and seed checks.
The recipe 03 upper-right model/encoder check passed with fade enabled. These
checks verify implementation and loading, not real-robot performance.

The earlier 20-run import passed 70 focused checks, including a local-asset audit
with Turbo/GPFS file reads blocked. It preserves the original 18 goal/start/joint-pose triples
byte-for-byte. All 20 optional START poses satisfy P42 hardware joint/action
bounds. All 20 commands match the source sparse waypoints, cost values and
gradients, gain, seed, cadence, blend and 3-second cutoff. Each pair preserves
identical source body/joint starts. Hardware runtime tests cover manual Y before
and after the cutoff, retaining tracker history and discarding in-flight plans.
The S50R/P42 pair, original standing encoder, normalizer and PD binding pass the
model load/forward check. The 10 Hz, B2 synthetic spawned-planner check measured
73–75 ms on two warm solves with the TensorRT tracker configured; sensor/recording
load was absent, so this is not a sustained live-rate guarantee.

The standing remedy's earlier online tests used this same measured geometry and
S50R/P42, with the earlier `first_six` commands; they are documented in
[standing-hold.md](standing-hold.md). Those results do **not** establish success
for all 32 current commands. No robot process was started during integration.
Stanley's source replay uses 48 Isaac environments and oracle pelvis feedback;
these commands are a single-robot measured-feedback adaptation. The new
flat43 pair was selected for direction, smooth recovery and coarse tracking;
Stanley reports that it fails the stricter whole-path own-target-preference
comparison. This is not a claim of precise path tracking or hardware reliability.
The source qualification report is cached in
`logs/native_recovery/custom_lab_robust_20260922/left_right_qualified_01/README.md`.

To reimport this pinned package while Turbo is mounted:

```bash
git show 1664363:steerable-recovery/steerable_recovery/tools/import_lab_gallery.py
```

Legacy September 21 commands require
`--catalog g1_deploy/steerable-recovery/config/steering/terrain_gallery/catalog.json`.
