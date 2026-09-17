# Flat-ground comparison with Stanley's September 15 models

From `/home/osher/hv2`, open one live MuJoCo window at a time.

**New supine example: pose 1444, R8 S50R, 5 Hz.** Three tested seeds finish quietly,
with 0.29–0.71 m pairwise pelvis separation at sustained-upright entry:

```bash
.venv/bin/python g1_deploy/steerable-recovery/steerable_recovery/tools/experiments/flat_recovery.py view --case supine1444_s50r
```

**Same pose and seeds, faster S25 model, 5 Hz:**

```bash
.venv/bin/python g1_deploy/steerable-recovery/steerable_recovery/tools/experiments/flat_recovery.py view --case supine1444_s25
```

**Earlier examples, with their original 2.5 Hz recipes:**

```bash
.venv/bin/python g1_deploy/steerable-recovery/steerable_recovery/presets/flat.py --start supine
.venv/bin/python g1_deploy/steerable-recovery/steerable_recovery/presets/flat.py --start prone
```

**N** cycles the three fixed seeds; **R** repeats; **Space** pauses. Close the window
to exit. The camera stays fixed and displays the actual planner depth input.
These commands have no time limit and record no video. They save compact
trajectory diagnostics. Initial pose, zero velocity and camera delivery seed
stay fixed; fresh planner noise is sampled at each plan. Joint jitter is zero.

The previous prone R2 flow examples remain the better spatial demonstration:
0.41–0.76 m separation at 2.5 Hz, versus 0.03–0.11 m for S50R at 5 Hz on the same
pose/seeds. The tested S50R supine 151 case has two quiet finishes out of three;
all three get up. Do not replace the previous examples indiscriminately.
These small selected sets do not establish population success rates or three
separate semantic recovery modes.

All new trials used the same TensorRT FP32 backend as the previous examples,
nominal PD, training Warp depth, real delivered depth prefill, repeated tracker
startup history, and the **0.5-second-only** command limiter. The new S50R planner
has a 1 s horizon; S25 has 0.5 s. Both use DDIM8 per stage, EMA and fresh noise.
The tracker runs at 50 Hz. Full-sampler FP32 parity passed for the new export;
this does not imply identical trajectories to Stanley's PyTorch TF32 runs.

## Other comparisons and results

Use `view --case NAME` with `supine151_s50r`, `supine151_s25`,
`earlier_supine`, `earlier_supine_s50r`, `earlier_prone`, or
`earlier_prone_s50r`. All comparison cases default to **5 Hz**. Append
`--replan-hz 2.5` or `--seeds 114 819 862` for a separate exploration; those
overrides were not part of the three-seed comparison.

[Full results and paths](../../../../../logs/multimodal_recovery/stanley_flat_20260915/README.md)
retain all 24 new trials and the six previous 2.5 Hz examples. Within-case first
observations matched exactly, the noise differed, and shared starts matched
across models. The report's upright time includes the full one-second hold;
quiet finish additionally requires final-second joint-speed RMS below 0.5 rad/s.

`supine_151.json` and `supine_1444.json` adapt Stanley's named joints and pelvis
attitude to a **settled resting** MuJoCo state. The exported Isaac first-decision
states had nonzero velocities; they are not copied as moving initial states.
Each snapshot records source hashes and passive-rest checks. Settling changes
joint angles and pelvis height; this is a new rest-start experiment.

`reference_routes/flat_pose_0/1/2.json` are unchanged copies of Stanley's three
supine-151 route goals. They are plotted as references only. **No goal is passed
to the policy in this unguided comparison.** The separate
[steering launcher](../../../docs/steering.md) now passes these exact costs
to the planner. Its first MuJoCo checks did not reproduce three successful
routes; hardware feedback remains unqualified. The hardware option below is unguided.

## Run on the robot

Use the [two-terminal deployment procedure](../../../docs/deploy.md).
It selects `r8_s50r_diffusion`, TensorRT, 5 Hz, fresh noise and
`supine_1444_robot.json` with `--seed random` on each launch.
START moves the joints slowly, then A starts recovery. The launch enters zero
torque first. Depth, LiDAR, robot state, model trace, live estimated XYZ and the
external camera record through the existing runtime.

The robot preset corrects the two ankle-roll targets by 1.80 and 2.35 degrees
and the right-knee target by 0.006 degrees to fit the configured hardware ranges.
The exact corrections and source-snapshot hash are stored in the JSON. Body
position, orientation and contacts still depend on how the operator places the
robot; a saved joint preset does not reset those quantities.

The code/PD/model check opens no sensors. Slow positioning was rehearsed in
MuJoCo from ordinary and selected supine poses; after positioning, all three
tested S50R recovery seeds finished quietly. [Validation receipt](../../../../../logs/deployment_validation/s50r_1444_20260915/validation.json).
This is not a hardware trial. Hardware RNG initialization differs from the
paired-episode simulation RNG; equal seed numbers do not reproduce its draws.

## Reproduce the comparison

The materialized S50R bundle lives at
`logs/native_recovery/bundles/r8_s50r_20260915a`; its exact R8+6k tracker is shared
with S25. The new planner export is under
`logs/native_recovery/exports_five_20260915/r8_s50r_diffusion_planner`.
On another checkout, `prepare` copies the pinned pair from shared Turbo and
regenerates the rest snapshots for that MuJoCo plant/version. Build the local GPU
export before running:

```bash
.venv/bin/python g1_deploy/steerable-recovery/steerable_recovery/tools/experiments/flat_recovery.py prepare
.venv/bin/python -m motion_diffusion.behavioral.hardware.export \
  --bundle logs/native_recovery/bundles/r8_s50r_20260915a \
  --model r8_s50r_diffusion --component planner --tensorrt \
  --output logs/native_recovery/exports_five_20260915/r8_s50r_diffusion_planner
```

The S25 tracker export and earlier model bundles must also be available, as in
the existing deployment setup. Use a new results directory:

```bash
.venv/bin/python g1_deploy/steerable-recovery/steerable_recovery/tools/experiments/flat_recovery.py run --output logs/multimodal_recovery/my_flat_comparison
.venv/bin/python g1_deploy/steerable-recovery/steerable_recovery/tools/experiments/flat_recovery.py report --output logs/multimodal_recovery/my_flat_comparison
```
