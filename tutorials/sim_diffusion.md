# Native diffusion and flow in MuJoCo

Run from the parent `hv2` repository with its virtual environment:

```bash
.venv/bin/python g1_deploy/sim_diffusion.py --model both --video
```

This runs both models sequentially for 10 simulated seconds each and saves
`rollout.mp4`, `trajectory.npz`, `plans.json`, `config.json`, and `summary.json`
under the printed `logs/native_recovery/mujoco_flat_*` directory. Video works
headlessly through EGL; a non-video run requires no graphics context.

For a live MuJoCo window on a machine with a display:

```bash
.venv/bin/python g1_deploy/sim_diffusion.py --model diffusion --viewer
.venv/bin/python g1_deploy/sim_diffusion.py --model flow --viewer
```

The window stays open after the episode duration so you can reset:

| Key | Action |
|---|---|
| R / Backspace | Repeat the current posture and seed |
| N | Reset with the next seed |
| 1 / 2 / 3 | Reset standing / face down / face up |
| 4 / 5 / 6 | Reset left side / right side / seated |
| 7 | Reset to a seeded random ground posture |
| Space | Pause / resume |
| M | Advance to the next model when using `--model both` |

Each reset clears physics, previous action/code, policy history, heading and
camera captures. The model weights remain loaded and the window stays open.
Each attempt gets a separate output directory. The selected posture and seed
appear in the window. Close the window or press Ctrl+C to stop. Camera interaction and synchronization
use the [MuJoCo passive viewer API](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer).
Slow inference slows the display rather than advancing physics behind the policy.

To watch recovery from the floor:

```bash
.venv/bin/python g1_deploy/sim_diffusion.py --model both --start supine --seconds 15 --video
.venv/bin/python g1_deploy/sim_diffusion.py --model flow --start prone --viewer
```

`--start supine` (face up on the ground) is the default. Front/back/side starts
use the decoder's default joints; seated starts bend the hips and knees.
`--joint-jitter .05` adds reproducible joint perturbations in radians.
`--start random` chooses a ground posture from the seed. Ground presets now
settle under gravity and contact **before policy time zero and before display**.
Temporary viscous damping removes reset energy; it is restored to the saved
plant settings before policy control. Velocities, forces, solver state and the
clock are cleared at the final supported pose. No fall/settling sequence is
shown or included in policy history. A seated preset can recline during this
passive preparation; it is not held upright by a hidden controller.
The summary records convergence, contacts and zero initial velocity.
There is no hoist or automatic balancing controller during recovery.
`--episodes N` repeats with seeds `seed+i` and fresh physics,
proprioception history, code history, and camera captures.

With the default `--joint-jitter 0`, **N changes the sampling seed but preserves
the exact starting pose** (unless `--start random` is selected). A different
seed does not guarantee a different recovery strategy. For visibly varied
starting conditions, use:

```bash
.venv/bin/python g1_deploy/sim_diffusion.py --model both --viewer --start random --joint-jitter .15
```

Then N resamples the ground posture, joint offsets and planner noise. Random
posture choices can repeat; keys 2–6 explicitly select a different ground posture.

To isolate sampling diversity from the **same resting pose and sensor condition**:

```bash
.venv/bin/python g1_deploy/sim_diffusion.py --model diffusion --viewer --start supine --pose-seed 123 --sensor-seed 123
```

N changes planner noise; R repeats the same sample. DDIM's eta=0 means its update
is deterministic given the initial noise, not deterministic given observations
alone. The generated 64-D native Z sequence is the sampler output, distinct from
the Gaussian noise used to initialize that sampler. Both the state-noise and
code-noise streams vary with the planner seed; no arbitrary code is injected
directly into the tracker.

For an eight-sample video grid and plots of generated Z and executed motion:

```bash
.venv/bin/python g1_deploy/compare_diffusion_samples.py --model diffusion --start supine --samples 8
```

This verifies exact equality of initial physics, history and depth inputs while
varying the planner seed. It saves initial noise, complete first state/Z plans,
pairwise trajectory distances, `diversity.png`, and `samples.mp4`. The video has
a one-second freeze of the common resting pose before policy time begins.
Distinct samples establish stochastic diversity; separate semantic modes need
additional analysis and are not inferred from a PCA scatter plot.

The bundle selected by `configs/diffusion.yaml` supplies the supported recipes:

| Mode | Sampler | Predicted horizon | Execution before replanning | Code blend |
|---|---|---:|---:|---:|
| Diffusion B20k | DDIM, 8 steps | 25 ticks / 0.5 s | Full 25 ticks / 0.5 s | None |
| Flow E20k | Heun, 16 steps | 25 ticks / 0.5 s | 5 ticks / 0.1 s | 3 ticks / 0.06 s |

“Full horizon” means executing each complete chunk before generating the next,
not stopping after one chunk. Both use their paired C10k decoder, frozen EMA
planner weights, and fresh sampling noise. The script reads saved reference
settings directly, without the asynchronous hardware timing adaptation.
The summary records actual replan ticks so the cadence is inspectable.

Implementation is in `motion_diffusion/behavioral/mujoco.py` and
`g1_deploy/sim_diffusion.py`. It reuses the repository SONIC
`humanoidverse2/data/robots/g1/g1_29dof_old.xml` asset and shared
`NativeRecoveryPolicy`. It does not load the older SONIC policy or legacy VIGOR
tracker. `--mjcf PATH` selects another floating-base G1-29 asset with matching
joint/body names and a flat static world. `--bundle PATH`, `--backend pytorch --device cpu`,
`--seed`, `--seconds`, and `--output` are also available.

The plant uses 200 Hz MuJoCo physics and 50 Hz control. Every physics step
computes `kp * (target - measured_q) - kd * measured_dq`, with saved per-joint
effort limits. Targets use saved default positions and action scales; XML motor
actuators are disabled to avoid adding a second actuator model. Saved armature
and joint friction replace XML defaults. Floor sliding friction is 1.
This is a nominal MuJoCo transfer test, not an exact reconstruction of IsaacSim
contact dynamics or training domain randomization.

The camera uses the model's calibrated mount and intrinsics. Batched MuJoCo
rays intersect full robot visual meshes and the plane, then convert ray length
to optical depth. Inputs preserve 64×48 metre-depth and local-surface mask
channels, millimetre quantization, capture-time proprioception, 30 Hz captures,
and deterministic 0–40 ms delivery delays. Missing captures remain masked;
undelivered holes do not compress history. No external camera, robot SDK, DDS
publisher, SSH tunnel, or controller release is used.

Simulation retains training's action clip and torque limits, logs raw decoder
magnitudes, and continues through ordinary falls and saturation. Nonfinite
state/actions or a MuJoCo physics warning end the run with a recorded status.
There are no hardware pose, velocity, preview, or command-age gates. A completed
duration is not a success score: inspect videos, height/upright metrics, and
saturation counts.

For a matched sweep without a viewer:

```bash
.venv/bin/python g1_deploy/sim_diffusion.py --model both --postures prone supine left_side right_side seated --episodes 3 --joint-jitter .05 --video
```

TensorRT FP32 with CUDA graphs is the default backend. `--backend pytorch` uses
the reference implementation. See [inference latency](inference_latency.md) for
benchmarks and experimental TF32 comparisons.
