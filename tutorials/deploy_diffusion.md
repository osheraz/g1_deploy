# Real-world recovery: commands and test options

Current recipe: **R8_S25 diffusion40k + frozen R8+6k depth tracker**, TensorRT,
**fresh noise every plan, 2.5 Hz replanning, 0.5 s horizon, 50 Hz tracking**.
Specify the model explicitly; the CLI default selects another model.

## 1. Depth stream — terminal 1

```bash
ssh -t -o AddressFamily=any -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:15556:127.0.0.1:5556 \
  -L 127.0.0.1:15558:127.0.0.1:5558 \
  g1_29 \
  '/home/unitree/osher/native_recovery/.venv/bin/python /home/unitree/osher/native_recovery/native_depth_server.py --serial 247122073927 --bind tcp://127.0.0.1:5556 --clock-bind tcp://127.0.0.1:5558'
```

Leave it running. Reuse an existing stream; both forwarded ports are required.

## 2. Recovery and recording — terminal 2

Place the robot at rest or supported. **An actuated launch releases the built-in
controller and enters zero torque after robot connection, before START/A.**
Stop any separate `zero_torque.py` or deployment process first. Keep the external
Brio camera fixed with the robot and floor marks visible.

This starts from the **current pose**, without predefined positioning or joint jitter:

```bash
cd /home/osher/hv2
SESSION="logs/real_world/r8_fresh_$(date +%Y%m%d_%H%M%S)"

.venv/bin/python g1_deploy/deploy_diffusion.py \
  --model r8_s25_diffusion --backend tensorrt \
  --seed random --noise-policy fresh --replan-hz 2.5 \
  --command-limits bounded \
  --actuate --duration-s 60 \
  --live-odometry --record-sensors --record-label fresh_2p5hz \
  --external-camera --camera-size 1920x1080 --camera-fps 30 \
  --output "$SESSION"
```

LiDAR, live XYZ, raw recording and the external camera start automatically;
**no third terminal or separate recorder is needed**. Startup waits for streams
and five seconds of readiness. A/START during warmup are discarded.

- Wait for **Live inference ready**, then press **A** to start recovery and mark the recording episode.
- **A again / SELECT / Ctrl+C** stops, damps and saves. The **60 s duration** also stops and damps.
- Skip **START** for current-pose recovery: without a preset it moves to default standing joint targets.
- Wait for **Saved and verified raw sensors** and check the model-trace save status. `Session closed` alone is not recording completion.

Use a new `SESSION` for each trial. `--release-controller` remains accepted but
is unnecessary: `--actuate` performs release automatically.

### Repeat a fixed prone joint start

Add one of these flags to the command above:

| Start | Flag |
| --- | --- |
| Current measured pose | Omit `--start-pose` |
| Saved face-down joints | `--start-pose prone` |
| Saved face-up joints | `--start-pose g1_deploy/config/multimodal/r8_s25_spatial/supine_robot.json` |

For either preset: **START → slow positioning → Live inference ready → A**.
A/SELECT/Ctrl+C during positioning cancels. After positioning press A once;
another A stops recovery. Pending readiness now reports its reason. Holding
the target and checking inference use separate schedules.

Presets specify joints, not body XYZ, orientation or contact. Place the robot in
the intended pose and floor location yourself. Joint error is printed at A and
saved with the positioning trace. Policy histories start from actual measurements;
camera history is retained. No posture-orientation gate blocks START.

### Noise, replanning and joint variation

Change the existing flag rather than adding a duplicate:

| Test option | Setting and meaning |
| --- | --- |
| Original R8_S25 sampling recipe | `--noise-policy fresh --replan-hz 2.5`: new noise each plan |
| New random sequence each launch | `--seed random`; the selected integer is logged |
| Repeat a sampling sequence | `--seed 42` or another fixed integer; physical outcomes can still differ |
| Compare fixed noise | `--noise-policy fixed_episode`: reuse one initial noise draw while still replanning from observations |
| Compare faster replanning | Replace `--replan-hz 2.5` with `5` or `10`; tracker remains 50 Hz |
| Same preset joint targets | Omit jitter, as in the main command |
| Perturb a preset | Add `--start-pose-jitter-deg 3`; optionally `--start-pose-seed 101` to repeat it |

The original recipe requests a plan every **0.4 s**; its **0.5 s horizon** is not
the execution interval. Full-horizon hardware execution was not retained.
Different noise allows variation; it does not guarantee distinct recovery modes.

`Command limits: bounded` should appear at startup. Target changes start from
measured joints at half the saved velocity limits for one second, then increase
smoothly to the full rate by two seconds. Initial caps for this tracker are
0.20–0.37 rad per 20 ms (about 11–21 degrees): permissive limits, not a guarantee
of gentle physical motion. PD gains retain their training values; the policy
receives the request actually issued. [Implementation and evidence](../../notes/validation/robot_startup_fix_20260914.md).

## 3. Review — after stopping

```bash
.venv/bin/python g1_deploy/replay_motion.py "$SESSION"
```

Accepts the full session path or its `sensors/` path. Opens Viser at
**http://localhost:8081**; forward port 8081 in VS Code. Playback, external camera
and **Show depth** are at the top. The fixed-world pelvis trace uses recorded
live XYZ when available. Missing intervals and late origins are labelled.
Normal playback reuses data without rerunning DLIO or encoding an MP4.
Review checks are saved under `review/`.

| Review option | Flag / command |
| --- | --- |
| Prepare without opening Viser | Add `--prepare-only` |
| Export external video | Add `--export-video`; writes `review/external.mp4` |
| Recompute XYZ from raw LiDAR | Add `--offline-xyz`; slower, offline analysis |
| Replay model decisions | `.venv/bin/python g1_deploy/replay_model.py "$SESSION"` |

Model replay uses recorded exact inputs/noise and requires the referenced
weights/engines. Run it after stopping deployment because it uses the GPU.
If code changed since capture, `--allow-code-changes` explicitly permits a
comparison and records those differences.

Runs save original-resolution lossless depth once, robot/LiDAR/IMU streams,
external camera, live XYZ and exact model traces. Model depth uses the same
preprocessing; the R8_S25 planner/tracker range is **2 m**. Recordings belong in
`logs/real_world/`; aborted no-policy attempts are archived in
`logs/aborted_deployments/`, and offline checks in `logs/deployment_validation/`.
Git transfers code and presets, not recordings or model weights.

## 4. Zero torque

Stop deployment first, then:

```bash
cd /home/osher/hv2
.venv/bin/python g1_deploy/zero_torque.py
```

The robot goes limp. Ctrl+C stops the publisher; it does not restore standing.
Stop this script before starting another deployment.

## Optional checks and live XYZ

Sensor-only check using the same depth stream; no A press or motor commands:

```bash
cd /home/osher/hv2
SESSION="logs/deployment_validation/sensors_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python g1_deploy/deploy_diffusion.py \
  --model r8_s25_diffusion --backend tensorrt \
  --seed random --noise-policy fresh --replan-hz 2.5 --duration-s 20 \
  --live-odometry --record-sensors --record-label sensor_check \
  --external-camera --camera-size 1920x1080 --camera-fps 30 \
  --output "$SESSION"
```

Model-only CPU contract/forward check, without opening sensors:
`.venv/bin/python g1_deploy/deploy_diffusion.py --model r8_s25_diffusion --check`.
It does not validate TensorRT or physical recovery.

Live XYZ runs alongside the policy and is recorded for evaluation. Recovery
remains unguided; depth remains the planner input. Clock bounds and dynamic XYZ
accuracy are unqualified, and measured support contacts are missing. Keep
odometry in its default monitoring mode. Omitting `--live-odometry` retains raw
recording but removes live XYZ; `--offline-xyz` can reconstruct it afterward.

[Agent handoff and remaining work](../../notes/REAL_WORLD_DEPLOYMENT_HANDOFF.md),
[recording details](../RECORDING.md), [model reference](deployment_reference.md),
[MuJoCo commands](sim_diffusion.md).
