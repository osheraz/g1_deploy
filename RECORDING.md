# Recording and replay options

For the current recovery and recording commands, use [the operator guide](tutorials/deploy_diffusion.md).
It includes the depth command, policy command, A/stop controls and synchronized video/motion replay.
Do not run a second recorder beside deployment.

New hardware runs default to `logs/real_world/`; all artifacts and processed review
stay inside the run. Old results and model assets are not moved.

## What is saved

| File inside the session | Contents |
| --- | --- |
| `telemetry.jsonl.gz` | Policy state/actions, heading, timing and live odometry when enabled |
| `sensors/raw/sensors.npz` | Raw LiDAR, pelvis/torso IMUs, joint q/dq/estimated torque and timestamps |
| `sensors/raw/bag/` | Timestamped LiDAR and Livox IMU for offline processing |
| `sensors/raw/summary.json` | Sensor counts, gaps and capture errors |
| `sensors/recording_result.json` | Whether recording started and saved completely |
| `sensors/depth/{frames.bin,frames.jsonl,metadata.json,summary.json}` | Original-resolution lossless depth, calibration/units, capture/receipt times and recorder errors/drops |
| `odometry_mount.json` | Snapshot of the selected mount calibration, shared by live/offline pelvis conversion |
| `model_trace/control/` | Tracker history, native code, selected depth frame references/metadata/masks, raw/clipped output, issued request/target and command shaping |
| `model_trace/planner/` | Live/preview planner inputs, histories/masks, depth frame references, explicit noise, generator state, predictions and anchor |
| `model_trace/{manifest.json,source.tar.gz}` | Resolved settings, dependency versions, checkpoint hashes and the actual source including uncommitted changes; each component also saves engine identities in `runtime.json` |
| `model_trace/summary.json` | Completion and drop/error counters for both processes |

Actuated deployment records raw sensors by default. `--live-odometry` also runs
DLIO and pelvis conversion; without that flag, XYZ can be reconstructed offline.
Sensor-only runs record when `--live-odometry` or `--record-sensors` is selected.
Preflight/default-pose mode does not start this recorder.

The recorder buffers up to five seconds before A and saves after motor cleanup.
It runs separately at lower CPU priority; [measured compute cost](../notes/deployment_compute.md)
is nonzero. Support contacts are not estimated by this recorder.

Recorded deployments save the **original 640×480 uint16 depth**, losslessly,
once per exposure. The network camera queues the original PNG bytes already
received; a writer thread handles checksums and disk writes. USB capture copies
the original pixels and encodes PNG on the writer thread. There is no second
stream, GPU work or codec in the capture/control loop. A full queue drops
recording frames and reports it in `sensors/depth/summary.json`; it never stalls
the policy. Capture begins when the camera connects and ends after motor cleanup.
Use `--no-record-depth` to disable it. Viser applies the same calibrated crop
and finite-positive validity handling at native pixel resolution: **576×464**
with the current 32/8-pixel border crop. The complete 640×480 source remains in
the archive. Near/far display limits affect visualization only.
Model traces store each exposure's exact capture reference, age, availability
and capture-state metadata. Replay uses the same calibrated ray sampling to
64×48 and the same model pixel transform; it does not use a generic resize or
normalized video pixels. No repeated H3 image stacks or
second control journal are saved. The small proprioceptive history is retained
to diagnose incorrect resets or ordering directly. Frame checksums detect damage.
Metadata schema `hv2.recorded_native_depth.v2` identifies new full-resolution
captures. Older `hv2.recorded_metric_depth.v1` recordings still replay at their
saved 64×48 resolution; their missing original detail cannot be recovered.

## Reproduce model decisions offline

Model recording is **on by default**, including sensor-only inference and preview.
Depth recording defaults on with model recording. Explicit `--no-record-depth`
retains the older inline model-image trace so model replay remains possible;
use the defaults for compact storage and depth visualization.
Synthetic inference warmup is excluded. `--no-record-model` opts out. Each process
writes owned binary tensors through a bounded asynchronous queue; there is no
compression or second model inference during control. Tracker snapshots are
flushed after command publication. Logging loss never stops the policy, but makes
the model replay check fail. On exit, look for **Model trace saved without drops**
as well as the raw-sensor completion message.

After stopping, run:

```bash
.venv/bin/python g1_deploy/replay_model.py "$SESSION"
```

This opens no sensors or viewer. It re-runs each live tracker/planner call with
its recorded inputs, exact noise and generator state using the recorded backend.
It compares unclipped/clipped actions, planner states/codes and issued PD targets.
Results go to `review/model_replay.json`; a mismatch or incomplete recording exits
with failure. `--component tracker --max-calls 100` checks the first 100 tracker
calls; full integrity checks still cover the entire trace.

Keep the referenced model bundles and compiled exports: their hashes are saved,
but weights are not duplicated into every run. The dirty source is archived;
replay refuses changed local code unless you explicitly use `--allow-code-changes`
to compare a fix. This reproduces the neural decisions at the measured states,
not the physical recovery dynamics. Raw IMUs, joints/torque, camera, depth, live XYZ
and command timing remain available to investigate the actual motion.

## External camera

Add `--external-camera` (no value selects the Logitech Brio 101,
`/dev/v4l/by-id/usb-046d_Brio_101_2540APD0M3H8-video-index0`; give another
device path to override) with optional `--camera-size 1280x720 --camera-fps 30`
to a recorded policy session. `check_camera.py --list-modes` prints the MJPG sizes
and rates the camera offers; the driver clamps a request it cannot serve (the
Brio 101 is 30 fps at every size) and the report shows `requested_fps`, the
effective `fps` and `measured_fps`. The
recorder then starts one `v4l2-ctl` child at the same low priority that writes
the camera's MJPEG frames straight from the driver buffers to
`sensors/camera/frames.mjpg` and logs every frame's driver sequence number and
start-of-exposure timestamp to `sensors/camera.log`. No decoding, no encoding,
no per-frame user-space work (measured 0.00 s CPU for 300 frames; ffmpeg's
V4L2 input spins at about 30% of a core and was rejected). The control process
never touches the camera. Recording runs from recorder launch to stop, so the
pre-A idle is included.

Timestamps are `CLOCK_MONOTONIC`, the same clock as `time.monotonic()` in the
policy telemetry, the raw recorder's `receipt_monotonic_s` and
`sensors/policy_start.json`; no wall-clock conversion is needed. The logged stamp
is the driver's start-of-exposure time, not an independently calibrated one.
After the policy has stopped, the recorder indexes the dump into
`sensors/camera/frames.npz` (`seq`, `bytes`, `offset`, `ts_monotonic_s`,
`policy_start_monotonic_s`), cuts a frame left incomplete at stop, and remuxes
a playable `sensors/camera/external.mkv` (uniform 30 Hz timeline; use
`frames.npz` for timing). `recording_result.json` reports frames, duration,
`dropped` (driver sequence gaps, exact), warm-up delay, mean/max interval, and
the frame index and video second at the A press. Camera failure is reported with
the saved recording; it does not stop the policy. Logitech cameras lower their
frame rate in dim light unless `exposure_dynamic_framerate` is off; the recorder
sets it off before streaming.

Before the first session with a new camera, check its MJPG modes and take a
short capture:

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video0 -c exposure_dynamic_framerate=0 --set-fmt-video=width=1280,height=720,pixelformat=MJPG \
  --set-parm=30 --stream-mmap=4 --stream-to=/tmp/cam_test.mjpg --stream-count=300 --verbose 2> /tmp/cam_test.log
grep -c dqbuf /tmp/cam_test.log      # 300 frames; each line carries seq and ts
```

Prepare the full video and synchronized robot replay after stopping:

```bash
.venv/bin/python g1_deploy/replay_motion.py "$SESSION"
```

This saves `review/review.json` and `review/README.md` and opens Viser on port 8081.
Playback, camera and depth appear first; diagnostics start collapsed. It uses
recorded live XYZ when present, otherwise cached offline XYZ. Default checks
validate the whole frame index and decode only three camera samples. Add
`--full-validation` to decode every frame, `--prepare-only` for checks without
Viser, or `--export-video` to encode `review/external.mp4` for sharing.
Viser reads the indexed raw JPEGs on demand, using exposure timestamps
rather than assuming frame number equals a policy tick. It keeps only the current
decoded frame and sends a preview at up to 960 pixels wide; the MP4 stays full resolution.

Camera exposure and USB transfer delay remain unqualified. Do not assume a fixed
one-frame delay or treat commanded motor onset as an independent clock reference;
use an independently timestamped visible event for synchronization qualification.
Keep the camera fixed and include measured floor marks for the gross translation
check; a single uncalibrated view is not 3D truth.

## XYZ options and older recordings

Use only after the policy has stopped. Set `SESSION` to the existing run path:

```bash
cd /home/osher/hv2
SESSION=logs/real_world/YOUR_SESSION_DIRECTORY
.venv/bin/python g1_deploy/replay_motion.py "$SESSION" --port 8081
```

This opens saved motion without recomputing DLIO. It starts no sensor driver or
robot connection. Use `--offline-xyz` explicitly to reconstruct raw LiDAR when
live XYZ is absent, or to compare against it. Offline interpolation differs from
what was available during control; default live replay uses the packets the policy
actually received. `--live-estimates` forces that source. `--pose-only` skips XYZ.
Old runs without depth images cannot display them; Viser labels this absence.

The ROS workspace defaults to `~/.local/share/hv2/lidar/ros2_ws`; `LIDAR_WS`
overrides it. [Build instructions](config/lidar/README.md).

If DLIO has no estimate at A, the offline viewer labels its later position origin
and hides the earlier XYZ interval. That trace is not displacement since A.
Deployment automatically waits for the requested streams and five seconds of
pre-roll before allowing startup. This applies to sensor-only checks too. The
countdown is before motor takeover; it adds no wait/check to running control.
Its receipt is saved in `sensors/startup_warmup.json`.
Offline conversion uses the run's `odometry_mount.json` when present. For older
runs, pass `--mount-calibration g1_deploy/config/lidar/mount_candidate_20260912.json`
to use the same candidate as the current live path. Otherwise it uses nominal
geometry. Replay remembers the selected offline result in `review/xyz_selection.json`;
subsequent playback does not silently switch back to nominal geometry.
