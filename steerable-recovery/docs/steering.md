# Three pelvis-trajectory steering tests

These are Stanley's three **pelvis XYZ + full-orientation trajectories** from
flat supine151. The original timestamps, goal frame and tolerances are preserved.
Guidance defaults to the source gain300; `--guidance-scale` explicitly overrides
its strength in MuJoCo and on the robot. The source files already contain six-waypoint approximations.
There are no single-axis substitutes or custom guidance cutoffs.

Choose `--model r8_s25_diffusion` for a 0.5 s horizon (default 2.5 Hz), or
`--model r8_s50r_diffusion` for a 1 s horizon (default 5 Hz). Omitting the model
keeps S50R. `--replan-hz 2.5|5|10` changes the planning cadence; tracking stays
at 50 Hz. All use fresh noise each plan and seed42 unless overridden.
The route resolutions below were selected with S50R, not tuned for S25.

**XYZ definition:** pelvis displacement from the first policy observation (A on
the robot), in fixed initial-heading axes with Z upward, in metres. The origin
persists across replans. `Z=0.539` for route3 means a53.9 cm rise, not53.9 cm above
the floor. MuJoCo uses true pelvis position; hardware uses DLIO plus waist FK.
The coordinate formulas match, but real height accuracy is not established.
The route3 held MuJoCo start is7.0 cm high, so its final target is60.8 cm above
the floor. Its early negative delta-Z also puts the target1.2 cm below the floor;
the held start differs from the donor's partly raised start. Do not add a guessed
floor offset to live feedback. [XYZ/height audit](../../notes/validation/steering_xyz_20260915.md).

| Option | Default resolution | Goal duration | MuJoCo XYZ RMSE | Final second standing |
| --- | --- | ---: | ---: | --- |
| 1 | Dense,338 points | 6.74 s | 12.2 cm | Yes |
| 2 | Stanley's6 points | 6.04 s | 16.8 cm | Yes |
| 3 | Stanley's6 points | 5.42 s | 9.7 cm | Yes |

The defaults select one resolution per route from a matched dense/coarse check.
**Inspect route3 first. Routes1/2 lose substantial height after initial get-up,
then recover again; ending standing does not mean stable standing throughout.**
All used the same held supine151 start, seed42, fresh noise, 5 Hz planning,
R8_S50R40k and frozen R8+6k depth tracker. These are individual15 s simulation
trials selected after comparison, not success rates or hardware qualification.
Six points did not give the best standing outcome for route1; dense was not
best for routes2/3. [Detailed comparison](../../notes/validation/pelvis_trajectories_20260915.md).

The planner uses PyTorch for guidance gradients; the50 Hz tracker uses TensorRT.
Depth remains the model input. LiDAR supplies estimated pelvis feedback.

**Lower-strength comparison:** use `--guidance-scale 50` for one sixth of the
source guidance coefficient. This biases the sampled diffusion trajectory less;
it does not divide motor torque or PD gains by six. Full XYZ/orientation targets,
timestamps, fresh-noise sampling and endpoint hold stay the same. The actual and
reference gains are saved in the session goal, and robot folder names include
`_g50_`. Omitting the flag retains300.

Route3/S50R, held supine151, seed42, 5 Hz, one15-second MuJoCo trial per gain:

| Guidance gain | XYZ RMSE | Held upright for1 s | Final second upright |
| ---: | ---: | --- | --- |
| 300 (existing reference) | 9.7 cm | Yes | Yes |
| 100 | 22.5 cm | Yes | No |
| 50 | 21.0 cm | Yes | Yes |

Gain50 is an available weaker comparison, not a demonstrated hardware fix.
Lower gains did not consistently improve stability and gave worse path tracking
in these samples. The two latest hardware trials also lost valid DLIO feedback
repeatedly; see [run diagnosis](../../notes/validation/steering_failures_20260915.md).

```bash
# Live MuJoCo, identical preset, noise seed and route:
.venv/bin/python g1_deploy/steering_options.py --model r8_s50r_diffusion --option 3 --replan-hz 5 --guidance-scale 50

# Robot; keep the terminal1 depth stream below running:
.venv/bin/python g1_deploy/steering_options.py --robot --feedback estimated --model r8_s50r_diffusion --option 3 --replan-hz 5 --guidance-scale 50
```

**S25 check, September15:** route3, the same supine151 rest state and seed42,
15 seconds each at 2.5, 5 and 10 Hz: none held upright for one second or ended
standing. No action saturation or physics warnings occurred. These checks show
that S25 accepts the steering interface; they do not reproduce S50R's success.
Watch these in MuJoCo before choosing a hardware trial. The 10 Hz trial also
had substantially worse path error. [Recorded checks](../../logs/deployment_validation/s25_steering_20260915/comparison.json).

S25's original bundle remains unchanged and describes unguided use. The explicit
experimental `guided_timing` entry in `configs/diffusion.yaml` supplies a five-tick
(100 ms) reply budget for trajectory steering. Session records retain the original
bundle profile and the override. Two warm synthetic guided samples took68–70 ms;
this excludes concurrent live sensors/recording and is not a real-time guarantee.

## MuJoCo

From `/home/osher/hv2`, run one at a time:

```bash
.venv/bin/python g1_deploy/steering_options.py --option 1
.venv/bin/python g1_deploy/steering_options.py --option 2
.venv/bin/python g1_deploy/steering_options.py --option 3
```

Magenta shows the actual commanded pelvis trajectory. Depth and the fixed-world
view remain. R repeats; N changes planner noise only. There is no default video
recording or time limit. Close the window to finish.

**S25 rate comparison**, same face-up start and route3, one command at a time:

```bash
cd /home/osher/hv2
.venv/bin/python g1_deploy/steering_options.py --model r8_s25_diffusion --option 3 --replan-hz 2.5
.venv/bin/python g1_deploy/steering_options.py --model r8_s25_diffusion --option 3 --replan-hz 5
.venv/bin/python g1_deploy/steering_options.py --model r8_s25_diffusion --option 3 --replan-hz 10
```

Use `--option 1` or `--option 2` for the other full pelvis trajectories. Those
S25 routes have not been evaluated here. Keep seed42 to compare rates; N or
`--seed random` adds sampling variation without changing the saved start.

Override resolution explicitly, preserving the selected route:

```bash
.venv/bin/python g1_deploy/steering_options.py --option 1 --waypoints coarse
.venv/bin/python g1_deploy/steering_options.py --option 2 --waypoints dense
```

Both use linear XYZ interpolation and quaternion SLERP. After the final timestamp,
the final full pelvis pose remains the target, as in Stanley's original schedule.
The earlier single-axis `simple` profile has been removed.

## Robot

**Terminal1 — depth:** reuse the existing stream, or follow
[depth terminal 1](deploy_diffusion.md#1-depth-stream--terminal-1), including
the two-file server update. Use raw transport on gigabit, or the PNG fallback
on a 100 Mb/s link. Both preserve the same filtered policy pixels.

**Terminal2 — sensor check first:**

```bash
cd /home/osher/hv2
.venv/bin/python g1_deploy/steering_options.py \
  --robot --sensor-only --feedback estimated --option 1 --seconds 15
```

This sends no motor commands. LiDAR, DLIO and all recorders start automatically.
The September15 S25/S50R ground recordings include live XYZ and all sensor streams.
Dynamic XYZ accuracy and independent clock bounds remain unverified; these runs
show substantial tilt disagreement during recovery, described in the diagnosis above.

**Ground trials, one at a time:**

```bash
.venv/bin/python g1_deploy/steering_options.py --robot --feedback estimated --option 1
.venv/bin/python g1_deploy/steering_options.py --robot --feedback estimated --option 2
.venv/bin/python g1_deploy/steering_options.py --robot --feedback estimated --option 3
```

**S25 hardware rate comparison**, using the same terminal1 depth stream:

```bash
.venv/bin/python g1_deploy/steering_options.py --robot --feedback estimated --model r8_s25_diffusion --option 3 --replan-hz 2.5
.venv/bin/python g1_deploy/steering_options.py --robot --feedback estimated --model r8_s25_diffusion --option 3 --replan-hz 5
.venv/bin/python g1_deploy/steering_options.py --robot --feedback estimated --model r8_s25_diffusion --option 3 --replan-hz 10
```

Run one per trial, with the robot face up on flat ground. These are experimental
tests, not demonstrated successful S25 steering recoveries. Options1/2 select
the other paths independently of rate. Start with2.5 Hz for more compute margin.

For each trial:

1. Stop other motor-control processes. Place the robot face up on flat ground,
   in the same marked position and heading. Launching enters zero torque.
2. **START is optional**: use it to move slowly to the common joint preset for
   a repeatable comparison. Skip it to recover directly from the current pose.
3. Wait for **Live inference ready** and **Steering feedback ready (estimated)**.
   Press **A**. Confirm **Steering active: estimated measured feedback**.
4. **A again / SELECT / Ctrl+C** stops and saves; duration also stops at60 s.
   Wait for **Saved and verified** before starting another run.

Joint positioning cannot reproduce the source base pose, contacts or velocities.
The held start is about7 cm lower than the source's moving first decision. The
unchanged reference includes that initial drop; tight physical-time agreement
is not established merely by successful get-up. Keep a fixed external camera
and floor marks to compare actual motion with estimated XYZ.

`--feedback estimated` explicitly accepts fresh geometry-valid estimates with
unqualified clock bounds. It does not certify accuracy or change `tracking_valid`.
Temporary stale/tilt/LowState samples pause guidance; valid data from the same
frame can resume it without changing the original origin. Missing initial XYZ,
estimator restarts, clock resets or jumps latch unguided recovery. These feedback
failures do not stop the policy. Qualified mode retains stricter eligibility.

## Recording and replay

Unique robot folders go under `logs/real_world/`; sensor checks go under
`logs/deployment_validation/`. Robot state, original processed depth, external
camera, raw LiDAR/IMU, live XYZ and model inputs/noise/outputs remain recorded.
No extra recorder terminal is needed. `--no-external-camera` omits only the Brio.

```bash
.venv/bin/python g1_deploy/replay_motion.py RUN_NAME
```

Viser shows the commanded path alongside measured pelvis motion, camera and depth.
Check steering-status logs: a displayed path alone does not prove guidance ran.
Keep seed42 for the first comparison; `--seed random` tests subsequent variation.
New folder names include the model, requested planning rate and route number.
