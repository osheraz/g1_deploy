# Three flat-ground recovery examples per start

Run from `/home/osher/hv2`. Each posture has **one fixed resting pose** and
three selected noise seeds. No joint jitter is enabled by default.

| Start | Model and frozen tracker | Options 1 / 2 / 3 | Replanning |
| --- | --- | --- | --- |
| Prone | R2_F50R 40k, R2+12k blind tracker | 49 / 101 / 707 | 2.5 Hz |
| Supine | R8_S25 diffusion 40k, R8+6k depth tracker | 50 / 44 / 42 | 2.5 Hz |

Both planners use depth, fresh noise at every plan and TensorRT FP32. The
tracker runs at 50 Hz. The prone planner has a 1-second horizon; supine has
a 0.5-second horizon. The existing startup limiter applies only in the first
0.5 seconds. The launcher changes no policy, history, camera or safety code.

## Live MuJoCo

```bash
.venv/bin/python g1_deploy/steerable-recovery/steerable_recovery/presets/flat.py --start prone
.venv/bin/python g1_deploy/steerable-recovery/steerable_recovery/presets/flat.py --start supine
```

Run one window at a time. **N** cycles the three selected seeds, **R** repeats
the current seed, **Space** pauses. The full starting pose, zero velocities,
camera delivery seed and initial histories are restored on every reset.
The camera stays fixed and depth is displayed. No video recording or time limit.
Add `--option 2` or `--option 3` to begin with that example.

For the rate comparison, append `--replan-hz 5`. This changes the recovery;
it does not preserve the selected 2.5 Hz paths. To explore new draws, append
`--seed random`; N then advances to new noise instead of cycling the three.

Optional nearby starts: append `--start-pose-jitter-deg 5`. N then changes both
joints and noise. `--start-pose-seed 123` fixes the initial pose perturbation
across launches. These are nearby-start comparisons, not identical-start trials.

All six selected seeds also completed 12-second recoveries at 5 Hz. One
5-degree perturbation trial per posture (pose seed 123, option 1 at 2.5 Hz)
also recovered. This is a limited check of the options, not a guarantee for
other perturbations. Simulation pauses for inference and does not certify
the hardware scheduling budget.

## Robot

Keep [terminal 1's depth stream](../../../docs/deploy.md#terminal-1-depth)
running. Place the robot on flat ground in the selected face-down or face-up
orientation, then run **one** command:

```bash
.venv/bin/python g1_deploy/steerable-recovery/steerable_recovery/presets/flat.py --start prone --robot --option 1
.venv/bin/python g1_deploy/steerable-recovery/steerable_recovery/presets/flat.py --start supine --robot --option 1
```

Use `--option 2` and `--option 3` for the other seeds. The command enters the
normal deployment's startup zero torque. Once ready, **START** moves slowly
to the saved joint targets; after positioning/readiness, **A** starts recovery.
**A again / SELECT / Ctrl+C** stops and saves. Default duration is 60 seconds.

The same `--replan-hz 5`, `--seed random` and optional
`--start-pose-jitter-deg 5` flags work on the robot. By default the command
records the external camera, original-resolution depth, raw sensors, model
trace and live estimated XYZ under `logs/real_world/three_POSTURE_optionN_*`.
No separate LiDAR or recorder command is needed. `--no-external-camera` and
`--no-live-odometry` are available if those devices are not being used.

For **diffusion in either posture**, add `--model r8_s25_diffusion` (default
2.5 Hz) or `--model r8_s50r_diffusion` (default 5 Hz), plus `--seed random` for
new draws. This keeps the posture's same joint preset; it does not reproduce
the selected catalog paths. Explicit `--replan-hz` overrides the model default.
These runs include the model name in their output folder.
See [all four robot commands](../../../docs/deploy.md#sensor-session-and-operator-controls).

Wait for saving to finish, then open the recording with:

```bash
.venv/bin/python g1_deploy/steerable-recovery/steerable_recovery/review/session.py RUN_NAME
```

`--print-command` prints the expanded command without loading models or opening
sensors. The simulation and robot preset files contain **exactly the same
29 joint targets**. Hardware positioning cannot reset root pose and contacts
exactly, so a simulation seed does not guarantee the same hardware trajectory.

## What the examples demonstrate

- **Prone:** different knee/leg placement and floor paths within a push-up
  recovery family. These are not three distinct roll strategies. Pairwise
  pelvis separation at upright entry is **0.41–0.76 m**.
- **Supine:** roll through face-down, rise from the side, and curl through
  sitting. Pairwise pelvis separation at upright entry is **0.32–0.62 m**.

The prone start is a symmetrized teacher-derived pose, passively settled on
the floor; supine uses the teacher-derived 6819 pose with the robot's ankle
bounds. Initial pelvis heights are 8.5 cm and 6.2 cm, respectively, with zero
velocity. No initial falling or raised push-up pose is used.

These are selected successful 12-second examples, not a success-rate estimate
or a measurement of the number of modes. The selection compares joint motion,
body tilt and XY paths before standing; it does not rank final heading alone.
Upright entry is the beginning of a one-second interval with pelvis height
at least 0.55 m and upright cosine at least 0.75, a MuJoCo diagnostic rather
than Isaac's training success definition. No physical trial was run for this
selection. Results and confirmation are in
[`logs/multimodal_recovery/three_modes_20260915`](../../../../../logs/multimodal_recovery/three_modes_20260915).
