# Deployment status — temporary handoff

Snapshot: **2026-09-17**. Read this before resuming real-world deployment.
Current work in the conversation is moving to simulation / teacher-data curation.
This note can be deleted afterward; permanent commands live in
[Steerable Recovery](steerable-recovery/README.md).

## Start here

Run from `/home/osher/hv2`. `g1_deploy` is a separate Git submodule.

| Task | Current guide / entry |
| --- | --- |
| Robot: depth terminal, both diffusions, prone/supine | [Deploy](steerable-recovery/docs/deploy.md) · `g1_deploy/steerable-recovery/deploy.py` |
| MuJoCo: flat / fork3 / reset controls | [Simulation](steerable-recovery/docs/sim.md) · `g1_deploy/steerable-recovery/sim.py` |
| Three pelvis-trajectory steering options | [Steering](steerable-recovery/docs/steering.md) · `--route 1\|2\|3` |
| Viser / video / depth / model replay | [Recording](steerable-recovery/docs/recording.md) · `g1_deploy/steerable-recovery/replay.py RUN_NAME` |
| Depth and XYZ limitations | [Open issues](steerable-recovery/docs/status.md) |
| Lab blocks, ramps and placement previews | [Lab setup](../notes/validation/lab_recovery_9x24x36.md) |

Old commands such as `g1_deploy/deploy_diffusion.py` were removed, without
forwarding stubs. See [migration](steerable-recovery/docs/migration.md).
After updating the checkout, register the package once:

```bash
.venv/bin/python -m pip install --no-deps --no-build-isolation -e ./g1_deploy
```

## Current experiment settings

- **R8 S25 diffusion40k**: `r8_s25_diffusion`, 0.5 s horizon, usual 2.5 Hz planning.
  **R8 S50R diffusion40k**: `r8_s50r_diffusion`, 1 s horizon, usual 5 Hz planning.
  Both retain their frozen R8+6k depth tracker at 50 Hz.
- The documented unguided commands select TensorRT FP32, fresh noise each plan,
  `--seed random`, and no joint jitter. Omitting the seed uses the preset's saved
  example seed. Always specify the model; the full
  launcher's historical default is P11. Direct launcher backends can also differ
  from preset defaults; use the linked commands.
- Actuated startup enters **zero torque**. Wait for sensor warmup and live
  inference readiness. **A starts from the measured pose**; **START optionally
  moves to saved joints**, then A starts. Presets do not place the robot's body.
- A again / SELECT / Ctrl+C stops and damps. Robot presets default to 60 s.
  Wait for sensor and model-trace saves before closing the terminal.
- Extra startup limiting lasts **0.5 s only**. Preserve training PD, action
  scaling, joint order, paired checkpoints, and issued-action history feedback.
  Tracker history starts from repeated measured state; planner latent history
  resets; delivered depth context is retained. See [architecture](steerable-recovery/docs/architecture.md).
- Depth/timing interruptions during recovery use the existing warning/fallback
  behavior; do not reintroduce automatic shutdown for routine timing delays.
  Before A, missing paired depth still keeps the startup preview waiting.
- Steering follows a pelvis XYZ + orientation trajectory, with XYZ relative to
  the pose at A in initial-heading axes (+Z up). Target Z is displacement,
  not floor height. Guided planner uses PyTorch gradients; tracker uses TensorRT.
  Documented commands use guidance 50; omitting it selects 300 from the goal.
  Invalid feedback can suspend guidance while recovery continues unguided.
  The final target persists; it does not guarantee stationary standing.

## Recording and parallel computation

- Robot presets include live LiDAR/DLIO, raw sensors, full-resolution processed
  depth, external camera and model traces. No second recorder command is needed.
- Accepted A starts the episode. Before A there is warmup, bounded buffered
  context and small setup/status files, not a continuous saved media recording.
- New runs go to **`logs/real_world/`**. Only move approved recordings into
  `paper/real_world/` when requested. Empty launches belong in
  `logs/aborted_deployments/`; checks belong in `logs/deployment_validation/`.
- Replay accepts a bare run name; Viser is on port **8081**. It uses saved live
  XYZ when present. Video export and old-run offline DLIO are explicit options.
  Depth is stored once at source resolution; model history is reconstructed.
- Planner, depth, DLIO and recording run asynchronously from tracking, but still
  share CPU/network/disk. Camera rate (30 Hz) does not establish frame latency.

## Verified versus still open

The organization baseline was pushed as `osheraz`: **hv2 master `1163eedd`**,
pinning **g1_deploy main `206da85`**. Bundles, engines and ROS workspace are not
transferred by Git. This temporary note is newer than that baseline.

Saved checks: 77 preset commands matched after path translation; 19 help checks;
526 existing and 11 migration tests passed. One pre-existing SONIC test lacks
its external profile registry. A saved robot run loaded camera/depth/telemetry;
three tracker and three planner calls matched recorded outputs. Evidence:
`logs/deployment_validation/reorganization_20260917/verification.json`.
These are offline checks, not a new hardware qualification.

Follow-up audit on September 17: **110 focused tests passed**, covering entry
points, history initialization, startup limits, depth clocks, A-triggered capture,
recorder lifecycle, live-odometry integration and presets. All 25 compared
hardware/runtime/action-limit source files match the pre-move checkpoint byte
for byte. Both R8 S25 and S50R passed the new launcher's `--check`: checkpoint
hashes, paired contracts, CPU synthetic inference, training PD gains, default
joints and action scales. This check does not exercise TensorRT engines or live
sensors; the earlier saved-trace comparison covers a limited inference sample.
No deployment regression was found in these checks; live limitations remain below.

- **Depth latency:** latest asynchronous/raw transport passed offline checks;
  measure the updated server with real camera + SSH + LiDAR + recording.
  The reorganization did not upload it to the robot. Camera upload still needs
  two standalone files; the deploy guide gives both. Ports **15556 + 15558**
  are required. Use raw on gigabit, the documented PNG mode on a 100 Mb/s link.
- **XYZ:** live DLIO + waist FK/filter exists; independent clock bounds,
  dynamic accuracy and tilt-rejection causes remain unresolved. Estimated
  steering is experimental. Contacts and ETH elevation-map policy input remain
  deferred; depth is still the policy input.
- **Lab terrain:** supplied blocks are **9 × 24 × 36 inches** (height/depth/width).
  The saved search favors floor-level prone step / downhill ramp candidates,
  not unsupported supine trials. Use the [placement note](../notes/validation/lab_recovery_9x24x36.md#placement-and-viewing)
  and saved starts; flat START presets are not stair-placement sequences.
  Default simulation must remain free of injected camera delay.
- Separate pending simulation improvements: explicit start-offset / held-joint
  settling flags and continuing seed batches after a physics warning. These
  were not included in the organization work.

No robot processes or viewers were launched for the organization or this handoff.
Recheck robot connectivity and placement when deployment resumes. Leave concurrent
paper tooling untouched; do not merge Stanley's training branch just to inspect it.
