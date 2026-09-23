# Manual standing mode — Y

For Stanley's 20 selected runs, use [Lab steering](lab-steering.md).
Those presets enable manual Y by default and preserve their individual schedules.

**Purpose: press Y after get-up to stop further recovery motion and settle into standing.**
It does not wait for an automatic standing detector on the robot. It works with
steered or unguided recovery. Enable it with `--standing-hold` on your existing
robot command; the normal recovery is unchanged until you press Y.

- **A:** start recovery and recording. START remains optional joint preparation.
- **Y after get-up:** stop planner requests, discard queued plans, and blend the
  last executed latent into the paired tracker's original encoded standing
  reference over 0.5 seconds. The frozen tracker keeps balancing with live
  joints, IMU, depth and actual issued-action history.
- **A again / SELECT / Ctrl+C:** existing stop behavior. Y does not stop motors.
- Y before A is ignored. Repeated Y does not restart the transition. Standing
  mode remains latched until the run ends; a later fall does not resume planning.

The reference requests the checkpoint's default standing joints, zero joint
velocities, upright pelvis and the heading at Y. It uses the original encoder
and frozen normalizer. PD gains, action scaling and the startup-only limiter are
unchanged. There is no fixed motor-target hold or additional post-Y slew limit.
The encoder is loaded and warmed before startup; Y does no loading/compilation.

Standing does not require DLIO: the reference uses local coordinates. It does
**not** lock a world position. Steering before Y retains its existing feedback
requirements. Sensor/model/external-camera recording remains unchanged; the Y
activation, original encoder inputs, native codes and tracker outputs are recorded
for model replay. New robot sessions remain under `logs/real_world`.

## Measured lab scene, September 22

Results, model hashes, geometry verification, alternatives and the paired online
tests are in:
`logs/deployment_validation/standing_online_20260922/measured_lab/README.md`.

The package specifies **S50R**, CLI `s50r_diffusion`, with the P42 tracker.
Do not substitute `s50r_causal_diffusion` or the older R8 S50R.
All three static box sizes/translations were checked against the supplied USD.
The local `assets/` directory preserves `lab_scene.usda` and `lab_terrain.usda`
together. MuJoCo reproduces those supports and starts, not Isaac's batch/DR/noise.
This is simulation evidence; the Y handover still needs hardware testing.

Preview the upper-front supine placement and manually press **Y** once it gets up:

```bash
cd /home/osher/hv2
LAB=logs/deployment_validation/standing_online_20260922/measured_lab
.venv/bin/python g1_deploy/steerable-recovery/sim.py \
  --model s50r_diffusion --backend tensorrt \
  --gallery-start "$LAB/assets/lab26_start.json" \
  --goal-json "$LAB/assets/lab26_goal.json" \
  --start supine --seed 52032 --sensor-seed 42 --replan-hz 5 \
  --standing-hold --fixed-camera --viewer
```

The measured starts are `lab26` and `lab25` (upper-front interface), `lab09`
(lower-front interface), and `lab20` (lower-right interface). Change **both**
JSON names together. The guide's goals stop steering after 3 seconds, but normal
planning continues until Y. Removing steering alone is not a standing command.

## Robot command

Keep the existing [camera terminal](s25r.md#camera-before-hardware) running.
For the same lab26 goal, place the robot supine across the upper-front interface
shown by the simulation above. This command starts from the measured physical
pose; it does not reposition the joints. A starts; **you press Y after get-up**.

```bash
cd /home/osher/hv2
LAB=logs/deployment_validation/standing_online_20260922/measured_lab
.venv/bin/python g1_deploy/steerable-recovery/deploy.py \
  --model s50r_diffusion --backend tensorrt \
  --goal-json "$LAB/assets/lab26_goal.json" \
  --seed random --noise-policy fresh --replan-hz 5 \
  --standing-hold --actuate --duration-s 60 \
  --depth-processing vigor_legacy \
  --live-odometry --steering-feedback estimated \
  --record-sensors --record-depth --record-model --external-camera \
  --record-label lab26_s50r_manual_y
```

This retains the source steering scale. Simulation uses true pelvis feedback;
robot steering uses the existing estimated-feedback adapter. These tests qualify
neither that odometry nor hardware steering strength. To keep your existing
steering setup instead, simply add `--standing-hold` to its command.

The offline ankle-history diagnosis remains separate:
[standing-diagnosis.md](standing-diagnosis.md). A stable online ending does not
establish that every recorded ankle reversal had a planner cause.
