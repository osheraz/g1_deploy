# P11 recovery in MuJoCo

Run from `/home/osher/hv2`. No robot, sensors, SSH tunnel or LiDAR process is needed.
P11 diffusion20k + N17k depth tracker is the default, using the same local bundle
as deployment: PyTorch TF32, EMA DDIM8, fresh noise, K20/blend0.

## Live fork3 terrain and depth

```bash
cd /home/osher/hv2
.venv/bin/python g1_deploy/sim_diffusion.py \
  --model p11_diffusion --terrain fork3 --start prone --viewer
```

Requires a graphical display (`DISPLAY` on Linux). No video recording is enabled.
Live viewing runs continuously until you reset or close the window. Use
`--seconds 10` only when you want a timed episode. Headless runs default to 10 s;
`--seconds 0` runs either mode without a time limit.
The robot settles on the terrain before policy time zero. The inset displays the
latest arrived depth frame supplied to the planner: black is near, white is
far, purple is invalid, and dark purple means no frame has arrived yet. This
palette only affects the display; model depth values remain unchanged. A face-down
camera can be above the floor but within its 10 cm minimum range, producing
purple pixels. The startup console reports camera clearance and too-close/no-hit
fractions so this is distinguishable from being inside terrain.

| Key | Action |
| --- | --- |
| R / Backspace | Repeat the same posture, terrain and seed |
| N | New planner seed and a nearby starting pose |
| M | New terrain tile; preserve planner seed |
| P | Next model, when running `--model both` |
| 1 / 2 / 3 | Standing / face down / face up |
| 4 / 5 / 6 | Left side / right side / seated |
| 7 | Random ground posture |
| Space | Pause / resume |

M briefly reloads the window to rebuild terrain collision; weights stay loaded.
Every reset clears physics, action/code histories, heading and camera buffers.
N samples XY offsets up to ±10 cm, yaw up to ±15°, and roll/pitch up to ±5°
around the original selected posture. It advances the pose and planner seeds;
camera timing randomness stays fixed. Offsets do not accumulate across presses.
The robot settles again before starting, so terrain contacts determine the final
resting pose. R repeats that sampled start exactly. A new seed does not guarantee
a visibly different recovery route.

`fork3` samples all families from the existing `terrain_fall_fork3.npz` cache.
Use `--terrain stairs`, `pyramids`, `rubble`, `feature`, `feature_drop`,
`slope_feature`, `slope_feature_drop`, or `wave` to restrict sampling. For example:

```bash
.venv/bin/python g1_deploy/sim_diffusion.py \
  --model p11_diffusion --terrain stairs --terrain-seed 43 --start prone --viewer
```

The corrected cached triangle surface feeds the **shared training Warp
`PlannerDepthRenderer`**. Robot occlusion uses the training URDF driven by
MuJoCo body poses. Collision uses triangular prisms under that same surface,
not a heightfield. MuJoCo contact dynamics still differ from Isaac. This loads a
local 5 m tile with a small apron, not the full surrounding terrain atlas;
interpret trials that leave the tile accordingly. Camera corruption and physics
domain randomization are disabled. Model calibration, H3 history, 30 Hz captures
and causal delivery remain active.

`--terrain flat` uses a floor. The optional `--terrain step` is the earlier
manually sized box; it is separate from fork3 stairs. Its dimensions are set by
`--step-height`, `--step-edge`, `--step-length`, and `--step-width`.

Diagnostic trajectories/configs are saved in the printed output directory.
For headless execution omit `--viewer`; add `--video` only if you want an MP4.
The upright metric uses height above terrain under the pelvis and is a diagnostic,
not the training success metric. [Fork3 checks](../../notes/validation/mujoco_fork3_20260913.md).

For the optional startup-depth comparison, add `--depth-startup stationary`.
This renders one second of camera prehistory at the settled pose before policy
time zero, with normal capture spacing and delivery delays. Physics, action
history, latent history and planner noise do not advance. Both planner and
tracker receive the arrived depth history. The default `--depth-startup cold`
keeps the original empty-camera reset. This is a diagnostic option, not a
training recipe change or a smoothing mechanism. The [matched supine test](../../notes/validation/mujoco_startup_depth_20260913.md)
found that it did not materially reduce the initial leg command.

## P05 with the blind tracker

```bash
.venv/bin/python g1_deploy/sim_diffusion.py \
  --model p05_diffusion --start supine --viewer
```

P05 diffusion20k uses its exact **blind A24.5k tracker**. The planner still uses
H3 depth. Its reference recipe is EMA DDIM8, TF32, fresh noise, K25/blend0:
**2 Hz** planning. P11 remains the default model. P05 currently uses PyTorch;
there are no P05/A TensorRT exports to substitute for it.

## Change replanning frequency

```bash
.venv/bin/python g1_deploy/sim_diffusion.py \
  --model p11_diffusion --replan-hz 5 --start supine --viewer
```

| Requested rate | Control ticks between plans | P11 setting |
| --- | --- | --- |
| 2.5 Hz | 20 | Selected default |
| 5 Hz | 10 | Comparison |
| 10 Hz | 5 | Comparison |

The same flag works with P05. It changes only cadence: tracker stays at 50 Hz,
horizon stays at 25 ticks, DDIM stays at eight steps, and noise/blending stay as
saved. Requests must fit an integer number of control ticks and the horizon;
unsupported rates are rejected rather than rounded. The saved config records
both reference and actual settings. These comparisons can change recovery quality.

MuJoCo pauses physics during sampling, so a simulated rate is not a compute
qualification. The hardware launcher also accepts `--replan-hz`, retaining its
100 ms reply-age allowance and horizon check. With that allowance, 2.5/5/10 Hz
fit; 2 Hz needs the entire horizon and therefore is simulation-only. P05's
hardware default is adapted from K25 to K20 for this reason. Very high requested
rates may be limited by the single planner worker, even if the horizon fits.

## Multiple starts

```bash
.venv/bin/python g1_deploy/sim_diffusion.py --model p11_diffusion \
  --postures supine left_side right_side prone --episodes 2 --seconds 10 --video
```

Ground starts settle before policy time zero; they do not begin by dropping the
robot. `--joint-jitter .05` perturbs joint poses before settling. A seated preset
can recline during settling; no hidden controller holds it upright.

The plant uses the SONIC G1-29 MJCF, the tracker's saved PD/action scales, effort
limits, armature and joint friction, with 200 Hz physics and 50 Hz control.
Depth ray casting uses the checkpoint's training URDF visual meshes, driven by
MuJoCo body poses. The default asset’s visible and collision head now use the
training URDF fixed-head transform. Head contact has a 3 mm envelope and firmer
response to reduce startup penetration; masses, inertias and PD gains are unchanged.
This changes head-contact dynamics relative to older MuJoCo runs.
[Head-contact check](../../notes/validation/mujoco_head_contact_20260913.md). [Depth audit](../../notes/validation/mujoco_depth_20260913.md).
Gyro is expressed in the pelvis frame used by its quaternion. Both P11 networks
receive H3 calibrated depth at 30 Hz with causal 0–40 ms delivery delays. Valid
far returns saturate at 2 m; no-hit and invalid returns stay masked. Camera corruption and domain randomization are disabled. Physics pauses during inference: it is not a wall-clock hardware
latency test or an exact reproduction of Stanley's validation panel.

For P11 TensorRT FP32 comparison, add `--backend tensorrt`. It changes numerical
precision from the selected PyTorch TF32 recipe; recovery equivalence is not
established. See [inference details](inference_latency.md).

Legacy B20k/C10k diffusion and E20k/C10k flow remain available via
`--model diffusion`, `--model flow`, or `--model both` (both legacy models).
Those aliases keep their original samplers and default to TensorRT FP32.
`compare_diffusion_samples.py` and `evaluate_recovery.py` still target those
legacy models; use the commands above for P11.

Implementation: `g1_deploy/sim_diffusion.py` and
`motion_diffusion/behavioral/mujoco.py`, using the shared `NativeRecoveryPolicy`.
For physical deployment, use the [robot guide](deploy_diffusion.md).
