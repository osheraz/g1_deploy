# Deployment reference

For the commands to run now, use [the operator guide](deploy_diffusion.md).
This page is for model/export maintenance and runtime details.
Run commands from `/home/osher/hv2` with `source .venv/bin/activate`.

## Model bundle

The default is **P11 diffusion20k + its exact N17k depth tracker**, selected by
Stanley at `43170d4`. It runs unguided with EMA DDIM8, fresh noise per replan,
K20/blend0 and PyTorch TF32. Steering is not enabled by this package.

Local bundle: `logs/native_recovery/bundles/p11_20260913`. To install it on another
workstation with access to the shared filesystem:

```bash
rsync -aL /nfs/turbo/coe-stellayu-extra/clhsieh/software/hv2_diffusion/experiments/fallrec_latent_planner_20260902/deployments/native_recovery_p11_20260913a/ \
  logs/native_recovery/bundles/p11_20260913/
python g1_deploy/deploy_diffusion.py --check
```

Keep the manifest, materialized weights and PD binding together. The check verifies
hashes, paired dimensions and CPU inference, then compares PD gains, default pose
and action scales with the tracker's original resolved training YAML.

The tracker consumes ten chronological proprio/action-history rows (93 each),
a native Z64 and H3 depth features, producing 29 requested joint actions.
The planner uses H10 measured history, five actually executed codes and H3 depth,
and predicts F25 paired S67/Z64 steps at 50 Hz. Only issued requests and executed
codes enter feedback histories. Missing history is handled by the shared runtime.
Relative-attitude conditioning is off; pelvis gyro, gravity and causal heading
remain required. Unguided recovery needs no accumulated XYZ estimator.

The old `diffusion` and `flow` aliases retain the September 10 B20k/E20k models
with the blind C10k tracker. They have separate bundle and export paths in
`configs/diffusion.yaml`; changing the model key selects the whole pair.

## P05 and cadence comparisons

`--model p05_diffusion` selects a local P05 diffusion20k + blind A24.5k bundle.
The planner-bound tracker SHA is verified; the planner still consumes depth.
It uses PyTorch TF32. To recreate the local copy from the shared sources on a
new machine (destination must not exist):

```bash
python -m motion_diffusion.behavioral.hardware.bundle prepare \
  --catalog g1_deploy/configs/p05_candidate.json --models p05_diffusion \
  --asset-mode copy --output logs/native_recovery/bundles/p05_20260913
python g1_deploy/deploy_diffusion.py --model p05_diffusion --check
```

Both launchers accept `--replan-hz 5` or `--replan-hz 10`. The reference recipe
remains in the bundle; session receipts record the change. P05 defaults to K25
in synchronous simulation and K20 in hardware to leave five ticks for reply age.
P11 defaults to K20 in both. No sampler steps, precision, tracker rate or blend
are changed by this option. Faster replanning is a behavioral comparison, not a
claim of improved recovery. P05 has no compiled exports or physical qualification.

## R8_S25, R8 and R2 candidates (September 14)

`--model r8_s25_diffusion` is Stanley's fast short-horizon pick: state-first
diffusion 40k, DDIM 8 per stage, F25/K20, with the same frozen R8+6k depth
tracker as R8 flow, from `logs/native_recovery/bundles/r8_s25_20260914a`
(shared source `.../deployments/native_recovery_r8_s25_20260914a/`). Its
flat-ground simulation record is 511/512 get-ups, median 2.90 s. Since F25 equals
K20 plus the 5-tick reply allowance, slower cadences than 2.5 Hz are not
available; faster ones are.
`--model r8_f50r_flow` and `--model r2_f50r_flow` (blind-tracker
alternative) select the 40k state-first flow planners with their exact
frozen trackers from `logs/native_recovery/bundles/r2_r8_40k_20260914b`
(shared source: `.../deployments/native_recovery_r2_r8_20260914b/`; the `a`
bundle is the immutable 20k fallback with the same model names). Both planners
need depth; only R2's tracker is blind. Inference: EMA, Heun 4 per stage
(S67 first, then Z64), fresh noise, F50, K20, blend 0, PyTorch TF32 by default.
TensorRT FP32 exports of all three trackers and planners are at
`logs/native_recovery/exports_r2r8_20260914/<model>_{controller,planner}`
(parity against PyTorch FP32: state ≤ 3e-5, code ≤ 1e-5, tracker action 4e-4;
receipts in each directory). The same denoiser graph serves both sampling
stages. Measured sensor-only on bluefin, 30 s each: PyTorch planner 39–45 ms
per plan, control p50 3.1 / p95 8.2 ms, 0 late ticks; TensorRT planner
18–20 ms per plan, control p50 2.9 / p95 8.2 ms, 2 late ticks. Master carries
the model-side code of `origin/diffusion_policy` needed for the state-first
factorization; the hardware runtime is unchanged.

```bash
.venv/bin/python g1_deploy/deploy_diffusion.py --model r8_f50r_flow --check
.venv/bin/python g1_deploy/deploy_diffusion.py --model r2_f50r_flow --check
# TensorRT for the real stack or MuJoCo
.venv/bin/python g1_deploy/deploy_diffusion.py --model r8_f50r_flow --backend tensorrt ...
.venv/bin/python g1_deploy/sim_diffusion.py --model r8_f50r_flow --backend tensorrt --terrain fork3 --start prone --viewer
```

## Control and startup behavior

The paired decoder supplies PD gains, default pose and action scale. Requested
joint targets use `default + action_scale * action`; they are not increments from
the current pose. `--command-limits bounded` now shapes those targets using the
same implementation as MuJoCo: mechanical ranges plus an `effort/kp` spring
allowance, and per-tick target changes capped at half the saved joint velocity
times 20 ms initially. After one second, the rate increases smoothly to the saved
velocity over one more second. The first target starts from measured q. Limits remain active
throughout execution, and changed requests feed back into both histories. Gains,
action normalization and measured observations are unchanged. See
[the startup fix](../../notes/validation/robot_startup_fix_20260914.md).

Startup reports large joint deviations and blocks actuation only for extreme
measured limit excursions (more than 20 degrees outside configured limits), or
existing sensor/output faults. The warning threshold is 90 degrees from default
or within 5 degrees of a configured limit. Deeply folded starts remain allowed.
The optional START move ramps to default over two seconds and holds; it has no
separate tracking-error or settling gate. It is a joint hold, not a balance policy.

Finite output checks, the 35 rad/s measured-speed check, sensor
freshness, transport faults and the independent command watchdog remain active.
During bounded execution, finite clipped tracker outputs are shaped before PD
validation instead of being sent directly. Nonfinite commands remain errors.
A rejected startup preview retries on new measurements and leaves START/SELECT
available. Recorded C10k startup inputs also saturated in the original training
actor; bypassing output rejection does not repair that failure. Local evidence:
`logs/native_recovery/implementation_audit_20260912/`.
`--command-limits training` restores the prior direct-target path for explicit
comparisons. Default-pose mode retains its existing two-second ramp.

The planner runs in a separate process. The control loop maintains measured
history and consumes valid planner replies; an independent writer publishes PD
commands. See [compute architecture and measured costs](../../notes/deployment_compute.md).
Clock-age checks use the current estimated mappings. They do not establish an
independent LiDAR/robot-to-host offset bound for steering.

## Shared depth processing

The Jetson publishes raw 640×480 depth at 30 Hz, capture metadata and lossless
PNG. No temporal filtering, hole filling, or color alignment. USB/network frames
use the same calibrated crop/ray sampling, metric range/mask and causal history
as the optimized Warp camera in simulation.

P11 requires **three actual 30 Hz captures for both planner and tracker**.
Its v2 depth contract saturates finite valid returns beyond 2 m to 2 m with
mask=1. No-hit, invalid, near and missing returns remain zero with mask=0.
Legacy v1 checkpoints retain their original far=0 behavior. The encoding is
selected from each checkpoint's contract, never a deployment-only override.
The read-only raw viewer is not this processed model input.

## ONNX and TensorRT

Only neural forwards change backend. Depth preprocessing, history, sampling,
code queues and controls use the shared runtime.

**P11 defaults to PyTorch TF32 to preserve Stanley's selected numerical recipe.**
The new TensorRT FP32 tracker and planner exports are at
`logs/native_recovery/exports_p11_20260913/{controller,planner}`. They use reusable
buffers and CUDA graphs, but are an explicit precision variant, not the selected
TF32 result. Use `--backend tensorrt` only for that comparison. The older engines
cannot be reused for P11 or N17k.

The full TensorRT planner matches PyTorch FP32 (maximum state error 0.000019,
code error 0.000004 in the two synthetic cases rechecked on September 14).
The earlier comparison against selected PyTorch TF32 showed that
state error reached 0.03383 and code error 0.00715. Therefore a passing FP32
export comparison does not establish the same recovery behavior as TF32.
The full depth tracker comparison passes at maximum action error 0.000409.
Receipts are in `logs/native_recovery/exports_p11_20260913/`.

To rebuild and verify on another GPU (use new output directories):

```bash
BUNDLE=logs/native_recovery/bundles/p11_20260913
EXPORT=logs/native_recovery/exports_p11_new
python -m motion_diffusion.behavioral.hardware.export \
  --bundle "$BUNDLE" --model p11_diffusion --component controller \
  --output "$EXPORT/controller" --tensorrt
python -m motion_diffusion.behavioral.hardware.export \
  --bundle "$BUNDLE" --model p11_diffusion --component planner \
  --output "$EXPORT/planner" --tensorrt
python -m motion_diffusion.behavioral.hardware.export \
  --bundle "$BUNDLE" --model p11_diffusion --component controller \
  --output "$EXPORT/controller" --tensorrt --verify-policy
python -m motion_diffusion.behavioral.hardware.export \
  --bundle "$BUNDLE" --model p11_diffusion --component planner \
  --output "$EXPORT/planner" --tensorrt --verify-policy --reference-precision fp32
```

Omitting `--reference-precision fp32` compares against the selected TF32 sampler;
that strict comparison currently fails. Keep this distinction in any results.
Update the P11 export paths in `configs/diffusion.yaml` after rebuilding.
Keep `.onnx`, `.validation.npz` and `export.json` together; engine caches bind
GPU, runtime and precision. `--backend onnx` is the CPU numerical reference.
See [inference latency](inference_latency.md) for the offline benchmark.
