# Two additional fixed ground starts

These teacher-derived starts differ from the original demonstration presets:

- `prone.json`: face down, Bones source, teacher setup 321.
- `supine.json`: face up, LAFAN source, teacher setup 684.

Each was transferred to flat MuJoCo, passively settled, and saved at zero velocity.
Source filenames and hashes are inside the JSON files. These are simulation
snapshots, not hardware joint-target presets.

From `/home/osher/hv2`, run either command on the graphical desktop:

```bash
.venv/bin/python g1_deploy/sim_diffusion.py \
  --model r8_s25_diffusion --backend tensorrt --start prone \
  --start-state g1_deploy/config/multimodal/r8_s25_additional/prone.json \
  --seed random --sensor-seed 42 --pose-seed 42 --viewer
```

```bash
.venv/bin/python g1_deploy/sim_diffusion.py \
  --model r8_s25_diffusion --backend tensorrt --start supine \
  --start-state g1_deploy/config/multimodal/r8_s25_additional/supine.json \
  --seed random --sensor-seed 42 --pose-seed 42 --viewer
```

The window shows the actual planner depth input. No video is recorded and there
is no automatic episode timeout. **N** restores the same full pose and starts a
new planner noise sequence; **R** repeats the current seed; **Space** pauses.
Close the window before running the other command. The camera seed and reset
histories remain fixed. The sampled planner seed is saved in each episode's
`config.json`; replace `random` with that integer to reproduce it.

Validation: two seeds per pose, 12 seconds each, using R8_S25 TensorRT with the
existing bounded commands. Both poses had zero initial joint velocity and
identical first model conditions across seeds, with different noise. One seed
per pose reached the one-second upright criterion; one did not within 12 seconds.
All four finished without physics warnings or saturated action ticks. This is a
small functional check, not a recovery-rate estimate or guarantee of distinct modes.
The noise/reset regression suite passed 10 tests.

Detailed results: `logs/multimodal_recovery/additional_starts_20260914/validation.json`.

Follow-up: eight additional random prone seeds all recovered, with identical
initial model inputs and different noise. Their strongest contrast was still
leg placement/timing within a similar push-up recovery; distinct strategies and
hardware multimodality are not established. Real joint targets also do not ensure
identical measured starts. [Checks and experiment protocol](../../../../notes/validation/prone_same_start_20260914.md).

## Use the additional prone joint targets on the robot

`prone_robot.json` contains the same 29 joint angles as `prone.json`, in the
hardware preset format. All pass the R8_S25 deployment joint and action bounds;
no joint angles were adjusted. This preset has not run on hardware. Positioning
does not reproduce MuJoCo's base pose or contact state; the achieved joint error
is printed at A and recorded.

Keep the usual depth stream running. Place the robot face down, then run:

```bash
cd /home/osher/hv2
SESSION="logs/real_world/r8_prone321_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python g1_deploy/deploy_diffusion.py \
  --model r8_s25_diffusion --backend tensorrt \
  --start-pose g1_deploy/config/multimodal/r8_s25_additional/prone_robot.json \
  --seed random --actuate --release-controller --duration-s 60 \
  --live-odometry --record-sensors --record-label prone_teacher_321 \
  --external-camera --camera-size 1920x1080 --camera-fps 30 \
  --output "$SESSION"
```

After startup readiness, **START** slowly moves to the saved joint targets.
When the ramp finishes and the robot is still, **A** starts recovery and its
recording episode. **A again / SELECT / Ctrl+C** stops and saves; the command
also stops at 60 seconds. Wait for **Saved and verified raw sensors**.
Repeat the command block for the same targets and fresh planner noise.
