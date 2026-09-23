# Pelvis-trajectory steering

For Stanley's September 21 terrain examples and new S50R models, use
[terrain gallery commands](terrain-steering.md). The `--route` examples below
retain the earlier flat-ground models and targets.

Three saved pelvis **XYZ + orientation trajectories**, all from the held
**flat supine151** start. `--route 1|2|3` selects the trajectory.
Depth remains the policy input; live LiDAR/DLIO supplies hardware pelvis feedback.

## Simulation

```bash
.venv/bin/python g1_deploy/steerable-recovery/sim.py --route 3 --model r8_s50r_diffusion --seed 42 --replan-hz 5 --guidance-scale 50
.venv/bin/python g1_deploy/steerable-recovery/sim.py --route 3 --model r8_s25_diffusion --seed 42 --replan-hz 2.5 --guidance-scale 50
```

Choose one command; change `--route` for the other paths or `--replan-hz` for a
cadence comparison. R repeats; N changes noise while keeping the start fixed.

## Robot

Keep [terminal 1: depth](deploy.md#terminal-1-depth) running.
Place the robot **face up on flat ground**. Then choose one:

```bash
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --route 3 --feedback estimated --model r8_s50r_diffusion --seed random --replan-hz 5 --guidance-scale 50
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --route 3 --feedback estimated --model r8_s25_diffusion --seed random --replan-hz 2.5 --guidance-scale 50
```

The launch enters zero torque and starts live XYZ. Wait for readiness.
**START → saved supine151 joints → readiness → A** reproduces the intended
joint setup. **A directly** remains available from the current pose.
A again / SELECT / Ctrl+C stops and saves. Raw sensors, full-resolution depth,
model traces and the external camera are included automatically.

Sensor-only version (no motor commands): append `--sensor-only --seconds 15`.
Replay the printed run name with `replay.py`; see [recording](recording.md).

## Meaning of the settings

- `--guidance-scale 50` is a weaker comparison. **Omitting it uses 300** from the
  source goal. This changes planner guidance, not PD gains or motor torque scaling.
- Default resolutions: route 1 is dense; routes 2/3 use six source waypoints.
  `--waypoints dense|coarse` overrides resolution without changing the route.
- XYZ is **pelvis displacement since A**, in fixed initial-heading axes, +Z up.
  Target Z is a rise relative to the start, not height above the floor.
- Robot deployment automatically releases steering after upright IMU orientation
  and extended hip/knee posture persist for 0.2 s. It stays unguided for the rest
  of the run, including after a later stumble. Tracking and recording continue;
  the current plan bridges the wait for the first unguided plan. The console
  prints `Upright pose confirmed`; no extra flag is needed. See the
  [terrain guide](terrain-steering.md) for the exact posture thresholds.
  The source target itself retains its final waypoint; simulation is unchanged.
- Guided planning uses **PyTorch for gradients** and the **TensorRT tracker**.
- `--feedback estimated` opts into experimental, unqualified XYZ. Invalid or stale
  feedback can suspend steering while recovery continues unguided. Check the log
  for the actual feedback used. [Open measurement work](status.md).

S50R's selected route checks ended upright but were not uniformly stable throughout.
S25 did not stand in the previous route-3 rate checks. These are available test
configurations, not validated hardware success claims. Historical comparisons:
[trajectory checks](../../../notes/validation/pelvis_trajectories_20260915.md).
