# Recovery starts with different spatial paths

For the current simulation-to-robot comparison, use
[the same-start experiment note](../../../../notes/real_world_same_start.md).
It includes the robot-compatible supine simulation command and the latest
six-run check. The original supine example below uses slightly different ankles.

**Limiter update, September 15:** current runs limit targets only for the first
0.5 s. The spatial comparisons below were measured with the earlier, longer
limiter profile; they are historical results, not verified spreads for today's
commands. See the linked note for the focused recheck.

Use **supine** for the largest confirmed spatial separation. R8_S25 diffusion
40k and its frozen depth tracker recover toward opposite sides of the starting
point with seeds 49 and 707. Their pelvis locations are 1.16 m apart at the end
of the low recovery phase, and 1.21 m apart at first sustained upright entry.
Both finish standing quietly. This is translation, with no yaw term in selection.

The **prone** snapshot now lies flat, face down, with its pelvis 0.089 m above
the floor. Seeds 50 and 44 both recover, with 0.30 m endpoint separation at
upright entry. This replaces the rejected tucked pose; its larger 0.77 m
separation does not apply to the flat start.

## Live MuJoCo

Run from `/home/osher/hv2` on the graphical desktop, one window at a time:

```bash
.venv/bin/python g1_deploy/multimodal_recovery.py \
  --catalog g1_deploy/config/multimodal/r8_s25_spatial/catalog.json \
  --start supine
```

```bash
.venv/bin/python g1_deploy/multimodal_recovery.py \
  --catalog g1_deploy/config/multimodal/r8_s25_spatial/catalog.json \
  --start prone
```

**N** switches between the two verified seeds; **R** repeats the current seed;
**Space** pauses. The exact initial full pose, zero velocities, sensor randomness
and initial histories are restored each time. The camera stays at the initial
location so motion across the floor remains visible; depth is displayed. There
is no automatic timeout and no video recording.

Add **`--random`** to either command for a new random seed each launch; then N
advances to new seeds instead of cycling the verified pair. Both modes use fresh
diffusion noise at every replan. Selected seeds reproduce examples; arbitrary
seeds need not produce the same displacement or succeed.

## Compare replanning rates

Add `--replan-hz 5` or `--replan-hz 10` to either live command; explicitly use
`--replan-hz 2.5` for the saved baseline. For example:

```bash
.venv/bin/python g1_deploy/multimodal_recovery.py \
  --catalog g1_deploy/config/multimodal/r8_s25_spatial/catalog.json \
  --start supine --replan-hz 5
```

At 2.5 / 5 / 10 Hz a new plan is requested every 400 / 200 / 100 ms. The tracker
remains at 50 Hz and the model horizon remains 0.5 seconds. Keep the selected
seeds for this comparison so the initial pose, sensor seed and planner seed
remain matched across rates. N still switches between the two seeds.

Higher rates also draw fresh planner noise more frequently; they can change
the chosen route, rather than simply execute the old route faster. The spatial
separation measurements above apply to 2.5 Hz. MuJoCo pauses physics for
inference, so a successful simulation at a higher rate does not establish its
real-world compute budget.

I repeated supine seeds 49 and 707 for 12 seconds at each rate. All six runs
recovered and finished quietly, with identical initial physics, first model
conditions and first noise for each seed across rates. The actual recorded
plan intervals were 20, 10 and 5 tracker ticks, respectively.

| Replanning | Pelvis endpoint separation at upright entry | One-second upright hold reached (seeds 49 / 707) |
| --- | --- | --- |
| 2.5 Hz | 1.21 m | 5.74 / 6.46 s |
| 5 Hz | 0.36 m | 5.78 / 5.34 s |
| 10 Hz | 0.97 m | 6.54 / 7.20 s |

This selected pair shows that rate can materially change the spatial contrast;
it does not establish a generally best rate. Detailed comparison:
`logs/multimodal_recovery/spatial_replan_20260914/comparison.json`.
The launcher and replanning-contract checks passed 43 tests.

## Supine on the robot

Keep the [usual depth stream](../../../tutorials/deploy_diffusion.md#1-depth-stream--terminal-1)
running. Put the robot **face up** in the intended floor position. Set the
planner rate explicitly in the first line below:

```bash
cd /home/osher/hv2
REPLAN_HZ=2.5
SESSION="logs/real_world/r8_supine_${REPLAN_HZ}hz_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python g1_deploy/deploy_diffusion.py \
  --model r8_s25_diffusion --backend tensorrt \
  --start-pose g1_deploy/config/multimodal/r8_s25_spatial/supine_robot.json \
  --replan-hz "$REPLAN_HZ" --seed random \
  --actuate --release-controller --duration-s 60 \
  --live-odometry --record-sensors --record-label supine_spatial_6819 \
  --external-camera --camera-size 1920x1080 --camera-fps 30 \
  --output "$SESSION"
```

Use `REPLAN_HZ=2.5`, `5`, or `10`; the tracker remains 50 Hz. Start with 2.5 Hz
to match the original spatial example. To compare rates with the same planner
seed, replace `--seed random` with `--seed 49` for those comparisons.

Wait for readiness, press **START** to move slowly to the saved supine joint
targets, then **A** once the ramp is complete and the robot is still. A during
the movement cancels. **A again / SELECT / Ctrl+C** stops recovery and saves.
The command also stops at 60 seconds. Wait for **Saved and verified raw sensors**.
LiDAR, live XYZ and recording start automatically; no separate recorder command.

Afterward, view the recorded camera, depth and estimated motion:

```bash
.venv/bin/python g1_deploy/replay_motion.py "$SESSION"
```

The robot preset retains 27 simulation joint angles exactly; left/right ankle
roll targets are brought within configured limits by 2.30/1.58 degrees. These
targets passed the real deployment contract at all three rates. Both seeds
49 and 707 recovered quietly in a 12-second MuJoCo check with those adjusted
angles. The check did not reproduce the real positioning ramp; the physical
robot still starts from measured joints and contacts, not an exact simulation
reset. For this adjusted pose, seeds 49/707 separated by 0.21 m rather than the
original 1.21 m; carrying a seed to a slightly different state does not preserve
its original trajectory. Four additional seeds (46, 50, 44, 606) also recovered
quietly; the largest upright-entry separation among these six trials was 0.43 m
(seeds 50/606). The original 1.21 m contrast is therefore not validated for the
hardware-adjusted pose. Later operator sessions on 2026-09-15 used this exact
unjittered preset, including `r8_supine_fixed_episode_20260915_010102` and
`r8_supine_fixed_episode_20260915_010205`; their existence does not validate
the simulated spatial gap. Earlier simulation evidence:
`logs/multimodal_recovery/supine_robot_preset_20260914/validation.json`.

## Small joint perturbations

Both simulation and robot commands accept `--noise-policy fixed_episode`
(sample once at episode start, reuse for replanning) and `--noise-policy fresh`
(sample every plan; the R8_S25 default). Both still replan from current
observations. `--random` in the demo launcher or `--seed random` in deployment
selects a new sequence per run. Keep pose jitter off when comparing only these
two noise modes from the same saved start.

Add `--start-pose-jitter-deg 3` to the robot command above to sample joint
targets within ±3° of the nominal preset, restricted to joint and action bounds.
The target is sampled once per launch; **START** uses the usual slow positioning
ramp. The pose seed, sampled targets and actual measured start are recorded.
Planner noise stays independent (`--seed random`).

In MuJoCo, for either `--start prone` (flat, face down) or `--start supine`:

```bash
.venv/bin/python g1_deploy/multimodal_recovery.py \
  --catalog g1_deploy/config/multimodal/r8_s25_spatial/catalog.json \
  --start supine --random --replan-hz 2.5 --start-pose-jitter-deg 3
```

MuJoCo positions the joints and settles before starting the policy. **N** samples
another nearby pose and planner noise; **R** repeats both. Add
`--start-pose-seed 101` to repeat the same joint target across separate launches,
independently of the planner seed. Simulation saves the settled state as
`prepared_start.json`; both sensor and policy histories start from that state.
These are nearby-start experiments, not same-start multimodality comparisons.
Omit the jitter flag to keep the original fixed-start behavior. More varied
initial joints do not guarantee more varied successful recoveries.

Checked ±3° with pose seeds 101/102: two prone and two supine TensorRT runs
completed quiet recoveries at 2.5 Hz. Repeating preparation was exact; 100 pose
seeds passed the actual hardware PD bounds. No hardware run was performed.
Validation: `logs/multimodal_recovery/jitter_validation_20260915.json`.

## Validation and scope

Candidates were ranked by measured pelvis XY path separation among the existing
teacher-derived prone/supine sweeps, requiring successful quiet recoveries.
The selected runs were repeated in fresh MuJoCo processes. All four complete
trajectories matched their earlier recordings exactly; each pair had identical
initial physics and model conditions with distinct noise. The same R8_S25
weights, TensorRT FP32, DDIM8, 2.5 Hz replanning, 50 Hz tracker, camera and bounded
command settings were retained. No steering objectives were added.

The path RMS metric resamples each XY path by traveled distance through its
first one-second sustained upright interval. Endpoint gaps additionally measure
translation before later standing motion: upright entry is the start of that
interval (pelvis at least 0.55 m, upright cosine at least 0.75); low-phase end
is the first sample above 0.4 m with upright cosine above 0.55. Each trajectory
can reach a milestone at a different time. No independent rotations or spatial
alignments were applied to the two paths.

Supine artifacts are under `logs/multimodal_recovery/spatial_confirm_20260914/`:
`validation.json` and `supine_comparison.jpg`. That directory also preserves the
older tucked-prone comparison. Current flat-prone validation and the archived
previous catalog are under `logs/multimodal_recovery/flat_prone_confirm_20260914/`.
The snapshots and model hashes are packaged here; the teacher corpus is not
needed to run the examples. The noisy-seed, fixed-state and launcher regression
suite passed 10 tests.

These are selected simulation examples, not an unbiased estimate of mode
frequency or real-world validation. `prone.json` and `supine.json` restore
MuJoCo physics; `supine_robot.json` is the optional hardware joint-target preset.
