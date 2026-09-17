# Real-world recovery: commands and test options

For the three pelvis XYZ/orientation trajectories, use the separate
[steering procedure](steering.md). The selected dense/coarse variants finished
standing in their fixed-seed MuJoCo checks; hardware testing remains experimental.

New flat-supine option: **R8_S50R diffusion40k + frozen R8+6k depth tracker**,
TensorRT, **fresh noise every plan, 5 Hz replanning, 1 s horizon, 50 Hz tracking**.
Use the hardware-adjusted supine 1444 preset below. The earlier R8_S25 option
remains available at 2.5 Hz with a 0.5 s horizon. Specify the model explicitly;
the CLI default selects another model. These are unguided recoveries.

## 1. Depth stream — terminal 1

After updating this checkout, copy **both** camera files once (robot connected):

```bash
cd /home/osher/hv2
scp g1_deploy/core/common/native_depth_server.py g1_deploy/core/common/depth_wire.py \
  g1_29:/home/unitree/osher/native_recovery/
```

For the fast path, connect the robot through **gigabit Ethernet**. On bluefin,
`cat /sys/class/net/enp132s0/speed` should report `1000` or higher; the entire
robot cable/switch path must support that rate. Start the updated server:

```bash
ssh -t -o AddressFamily=any -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:15556:127.0.0.1:5556 \
  -L 127.0.0.1:15558:127.0.0.1:5558 \
  g1_29 \
  '/home/unitree/osher/native_recovery/.venv/bin/python /home/unitree/osher/native_recovery/native_depth_server.py --serial 247122073927 --processing vigor --encoding raw --max-mbps 0 --bind tcp://127.0.0.1:5556 --clock-bind tcp://127.0.0.1:5558'
```

**If the link is still 100 Mb/s**, replace `--encoding raw --max-mbps 0` with
`--encoding png --max-mbps 40` in that command. Do not send uncapped raw depth
over a 100 Mb/s link: depth alone requires about **148 Mb/s** at 640×480×30 Hz.

Leave it running. Reuse an existing stream; both forwarded ports are required.
`--processing vigor` uses VIGOR smoothing but preserves each source pixel's
missing/near/far classification for the policy. Startup must report
`vigor_policy_depth_no_alignment_v2`; the older `vigor_osher_depth_no_alignment_v1`
fills missing pixels and is a different mode. Use `--processing raw` for the
previous unfiltered stream. `--processing vigor_legacy` retains exact old hole
filling for comparisons; it is not the recommended policy input.
Filtering runs on the robot's camera computer; calibration, capture timestamps
and the model's crop/range/mask remain unchanged. Full-resolution recordings
contain the selected output and identify its processing mode.
`--encoding raw` sends those same uint16 pixels without camera-side PNG work;
it does **not** mean unfiltered depth (`--processing raw` is a separate option).
Startup prints `async raw publisher`. The publisher keeps one pending image;
bluefin prepares model observations on its receiver thread, then queues original
pixels to the existing recording thread for lossless PNG storage. No additional
stream, GPU work, or compression is added to the policy control loop. Recording
still starts at A, retaining its existing half-second depth context.
The PNG fallback retains the original transport. Its cap selects the latest
pending exposure: capture stays at 30 Hz, but large PNGs reduce delivery rate.
Bandwidth is per subscriber; close a separate depth viewer during deployment.
The deployment recorder saves depth from its existing receiver.
The September 15 100 Mb/s check showed depth congestion and LiDAR loss; see the
[historical input check](../../notes/validation/dlio_regression_20260915.md).
The raw path has passed offline pixel/replay and local transport checks; actual
camera/SSH latency alongside the policy and LiDAR still needs a connected check.
Clock refresh also runs independently on bluefin. A slow probe or timeout warns
and retries with the last accepted alignment, without stopping depth or the
preset hold. During recovery, lost or unpaired depth warns without stopping
the policy. Captures older than 250 ms are masked unavailable for both models;
valid paired captures resume automatically. A confirmed clock/source fault
quarantines depth until deployment is restarted, while recovery continues
with missing-depth masks. Before A, unavailable depth keeps the preview waiting.
The selected R8_S50R/R8_S25 planners and R8 tracker receive metres: below **0.1 m** or no
return becomes `(0, mask 0)`; **0.1–2 m** stays metric with mask 1; a known return
beyond **2 m** becomes `(2, mask 1)`. There is no VIGOR-style range normalization.

## 2. Recovery and recording — terminal 2

Place the robot at rest or supported. **An actuated launch releases the built-in
controller and enters zero torque after robot connection, before START/A.**
Stop any separate `zero_torque.py` or deployment process first. Keep the external
Brio camera fixed with the robot and floor marks visible.

For the new comparison, place the robot **face up on the flat floor**. The preset
positions the joints using START; there is no joint jitter:

```bash
cd /home/osher/hv2
SESSION="logs/real_world/s50r_supine1444_$(date +%Y%m%d_%H%M%S)"

.venv/bin/python g1_deploy/deploy_diffusion.py \
  --model r8_s50r_diffusion --backend tensorrt \
  --start-pose g1_deploy/config/multimodal/stanley_flat/supine_1444_robot.json \
  --seed random --noise-policy fresh --replan-hz 5 \
  --command-limits bounded \
  --actuate --duration-s 60 \
  --live-odometry --record-sensors --record-depth --record-model \
  --record-label supine1444_s50r_5hz \
  --external-camera --camera-size 1920x1080 --camera-fps 30 \
  --output "$SESSION"
```

LiDAR, live XYZ, raw recording and the external camera start automatically;
**no third terminal or separate recorder is needed**. Startup waits for streams
and five seconds of readiness. A/START during warmup are discarded.
Depth startup waits for the initial history span and a fresh paired capture.
Missing intermediate exposures retain their availability masks and do not
reset live-inference readiness; policy start does not clear the camera buffer.

- After **Live inference ready**, press **A** to recover from the current pose
  and mark the recording episode. **START is optional**: it first moves slowly
  to the selected joint preset and holds; wait for readiness again, then press A.
- **A again / SELECT / Ctrl+C** stops, damps and saves. The **60 s duration** also stops and damps.
- Wait for **Saved and verified raw sensors** and check the model-trace save status. `Session closed` alone is not recording completion.

Use a new `SESSION` for each trial. `--release-controller` remains accepted but
is unnecessary: `--actuate` performs release automatically.

Rerun the whole terminal-2 block for another draw from the same joint preset.
The actual random seed is logged. A fixed `--seed 50` repeats the hardware RNG
initialization while still drawing fresh noise every plan. MuJoCo uses a different
paired-episode RNG initialization, so the same seed number does not reproduce
its noise sequence on hardware. Actual pose, contacts, sensor timing and
asynchronous execution also differ; these commands do not guarantee three modes.

For current-pose recovery, skip START and press A when ready, even with
`--start-pose` configured. Without a preset, optional START requests default standing joint targets.
For S25 on the same preset, change only `--model r8_s25_diffusion` and the session
label/name. Retain 5 Hz for that model comparison.

### Both diffusion models, prone and supine

Use the same terminal-1 depth server. These shortcuts include all recording
flags, TensorRT, fresh noise each plan, and a 60-second duration. Run one at a
time from `/home/osher/hv2`; place the robot on the floor before launch because
actuation enters zero torque. Wait for **Live inference ready → A**. To use the
saved joints, optionally press **START**, wait for the move and readiness, then **A**.

**Face down, flat on the stomach:** both commands select the same saved
flat-prone joints, with no jitter.

```bash
.venv/bin/python g1_deploy/recovery_options.py --robot --start prone --model r8_s25_diffusion --seed random --replan-hz 2.5
.venv/bin/python g1_deploy/recovery_options.py --robot --start prone --model r8_s50r_diffusion --seed random --replan-hz 5
```

**Face up, flat on the back:** both commands select the same earlier **6819**
supine joints. The separate main command above uses **1444**.

```bash
.venv/bin/python g1_deploy/recovery_options.py --robot --start supine --model r8_s25_diffusion --seed random --replan-hz 2.5
.venv/bin/python g1_deploy/recovery_options.py --robot --start supine --model r8_s50r_diffusion --seed random --replan-hz 5
```

The earlier prone catalog used R2 **flow**; specifying `--model` explicitly
selects diffusion. Use optional START for comparisons on fixed joints; new model/rate
combinations are not reproductions of the catalog's selected trajectories.
Results are saved under `logs/real_world/MODEL_POSTURE_TIMESTAMP`.

### Repeat a fixed prone joint start

Replace `--start-pose` in the command above with one of these choices:

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
measured joints at half the saved velocity limits for **the first 0.5 seconds
of policy execution only**. At 0.5 s, both added position and target-rate bounds
turn off; controller targets pass through directly. Timing holds do not restart
or extend this window. Hardware and MuJoCo use the same profile. Initial caps are
0.20–0.37 rad per 20 ms (about 11–21 degrees): permissive limits, not a guarantee
of gentle physical motion. PD gains retain their training values; the policy
receives the request actually issued. Existing finite-value/action validity
checks remain. [Earlier startup investigation](../../notes/validation/robot_startup_fix_20260914.md)
used the previous limiter profile; its results do not qualify this shorter window.

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

For manual recovery boundaries, use **Recovery interval** in the viewer:
**Mark start**, **Mark standing / end**, then **Save & next recording**.
Choose the first standing frame before later drift. Marks reload from each
run's `review/recovery_interval.json`. [Full marking steps](../RECORDING.md#mark-recovery-start-and-first-standing).

Fresh XYZ rejected by the tilt consistency check is shown in **orange**, rather
than hiding the robot. Missing/stale estimates still leave gaps. Orange points
are diagnostic measurements, not qualified steering feedback.

New live runs use the shared 20 ms causal XYZ filter; raw XYZ and filter delay
are also recorded. Delay counts against the existing 50 ms steering freshness
budget. The filter reduces jitter, not drift. DLIO scan corrections remain at
10 Hz (three camera-independent LiDAR packets); pose propagation runs faster.
These changes are installed on bluefin and require no extra launch flag.
Keep the external camera fixed with floor marks visible for the next recovery:
we still need an independent check of actual displacement before enabling XYZ
steering. The commands here remain unguided recovery with XYZ monitoring.

| Review option | Flag / command |
| --- | --- |
| Browse every trial in one viewer | `replay_motion.py logs/real_world` (or one category folder); sessions are found recursively, sorted by path, and switched with a dropdown or Previous/Next buttons. Add `--preload` to load them all first (about 40 s for 29) so switching is instant. Anything under a folder named `hidden` is skipped; move sessions there to keep them out of the viewer. Moving folders while viewing is fine: every Previous/Next rescans, **Rescan folder** picks up new sessions, and a session that vanishes mid-playback is skipped. Launches where A was never pressed and no episode was recorded are moved to `logs/aborted_deployments/` (receipt `cleanup_no_policy_<date>_browse.json`) |
| Prepare without opening Viser | Add `--prepare-only` |
| Export external video | Add `--export-video`; writes `review/external.mp4` |
| Export depth aligned with the video | Add `--export-depth`; writes `review/depth_aligned.mp4` with one frame per external camera frame (same fps and length as `external.mp4`, purple = no valid depth). `replay_motion.py logs/real_world --prepare-only --export-video --export-depth` does every trial |
| Export every frame as images | Add `--export-frames`; writes `review/frames/external/NNNNNN.jpg` (original JPEG bytes) and `review/frames/depth/NNNNNN.png` with the same numbering as the videos, plus `frames/index.json` with exposure times |
| Motion overlay figure | Add `--export-overlay`; builds overlays from the frames you picked in `real_world_hightlight/<run>/NN_*.png` (or `.jpg`): `overlay_12345` uses all picks and every subset keeping the first and last pick is written too (`overlay_15` ... `overlay_1345`). Each pick's robot is cut out where it differs from the per-pixel median of the picks (people, told apart from the robot by colour, are excluded) and pasted onto that median scene, earlier picks at `--overlay-opacity OLDEST NEWEST` (default 0.40 0.70), the last pick opaque, cropped around the robot. PNGs in the highlight folder, JPEGs in `review/`; `overlay` is the all-picks version. `--export-strip` writes the picks side by side with the same crop. Runs without highlight picks get neither |

All exports start at the first camera frame after **A** (`review/export.json` records that
frame index) and keep the footage after the stop; the raw `frames.mjpg` still holds the
pre-A idle. Existing exports are reused unless the raw data or the start frame changed.
| Recompute XYZ from raw LiDAR | Add `--offline-xyz`; slower, offline analysis |
| Replay model decisions | `.venv/bin/python g1_deploy/replay_model.py "$SESSION"` |

Model replay uses recorded exact inputs/noise and requires the referenced
weights/engines. Run it after stopping deployment because it uses the GPU.
If code changed since capture, `--allow-code-changes` explicitly permits a
comparison and records those differences.

Runs save original-resolution lossless depth once, robot/LiDAR/IMU streams,
external camera, live XYZ and exact model traces. Model depth uses the same
preprocessing; these R8 planner/tracker inputs saturate at **2 m**. Recordings belong in
`logs/real_world/`; aborted no-policy attempts are archived in
`logs/aborted_deployments/`, and deployment checks in `logs/deployment_validation/`.
Git transfers code and presets, not recordings or model weights.

The camera and sensors warm up before A. Accepted policy A starts video,
depth, model-call and measured-pose recording; waiting previews are not saved.
Only bounded replay context is kept in RAM beforehand (0.5 s depth and 5 s raw
LiDAR/IMU/joints). Setup/status receipts are small and may appear before A.

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
