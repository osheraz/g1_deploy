# Recording and replay options

For the current recovery and recording commands, use [the operator guide](tutorials/deploy_diffusion.md).
It includes the depth command, policy command, A/stop controls and synchronized video/motion replay.
Do not run a second recorder beside deployment.

New hardware runs default to `logs/real_world/`; all artifacts and processed review
stay inside the run. After reviewing a run, move it to `paper/` manually if you
want to retain it for the paper. Existing recordings and model assets are not moved.
Replay accepts an explicit path to a reviewed run under `paper/`.

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
| `model_trace/planner/` | Policy planner inputs, histories/masks, depth frame references, explicit noise, generator state, predictions and anchor |
| `model_trace/{manifest.json,source.tar.gz}` | Resolved settings, dependency versions, checkpoint hashes and the actual source including uncommitted changes; each component also saves engine identities in `runtime.json` |
| `model_trace/summary.json` | Completion and drop/error counters for both processes |

Actuated deployment records raw sensors by default. `--live-odometry` also runs
DLIO and pelvis conversion; without that flag, XYZ can be reconstructed offline.
Sensor-only runs record when `--live-odometry` or `--record-sensors` is selected.
Preflight/default-pose mode does not start this recorder.

Sensors and estimators warm up before A; video, depth and model-call recording
start with the accepted policy A. Small setup/status files remain available for
startup diagnosis. Raw LiDAR/IMU/joints retain up to five seconds in RAM for
odometry initialization and save that context only for a started run.
It runs separately at lower CPU priority; [measured compute cost](../notes/deployment_compute.md)
is nonzero. Support contacts are not estimated by this recorder.

Recorded deployments save the **original 640×480 uint16 depth**, losslessly,
once per exposure. PNG network transport queues the original bytes already
received; a writer thread handles checksums and disk writes. Raw network transport
and USB capture copy the original pixels and encode PNG on that writer thread. There is no second
stream, GPU work or codec in the capture/control loop. A full queue drops
recording frames and reports it in `sensors/depth/summary.json`; it never stalls
the policy. Archiving begins at policy start and ends after motor cleanup.
A bounded half-second of prior depth is retained in RAM and saved at A so the
first model call's causal depth history can still be reproduced.
Use `--no-record-depth` to disable it. Viser applies the same calibrated crop
and finite-positive validity handling at native pixel resolution: **576×464**
with the current 32/8-pixel border crop. The complete 640×480 source remains in
the archive. Near/far display limits affect visualization only.
With `--processing vigor`, this source is the filtered camera image as received,
with source missing/near/far categories preserved, not the unfiltered sensor
image. `vigor_legacy` recordings contain the original hole-filled output and
retain their original processing label. `metadata.json/source_processing` stores the
filter recipe and SDK version once. Network PNG bytes are saved unchanged;
raw network pixels become lossless PNG on bluefin. `frames.jsonl` records
`wire_encoding`, payload size and decode/preparation time separately from the
archive's PNG encoding. Replay derives
both the video and model views from these bytes without filtering again. Use
`--processing raw` at server launch to record the previous unfiltered source.
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

Model recording is **on by default** for policy and sensor-only inference.
Waiting-for-A previews are evaluated but not archived as model calls.
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
recorder keeps one `v4l2-ctl` stream warm in a separate low-priority process.
The existing camera-check byte-count reader forwards original MJPEG frames to
`sensors/camera/frames.mjpg` and logs every frame's driver sequence number and
start-of-exposure timestamp to `sensors/camera.log` only for saved frames.
There is no decoding or encoding. The byte copies and file writes run in that
child, never the control process. Frames before the accepted A timestamp are
discarded; the device stays open, avoiding a camera restart at policy onset.
The new forwarding path has offline checks but no new hardware CPU measurement.

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
If a run stopped before recording policy observations but saved external camera
frames, the same command opens camera-only playback with scrubbing and Restart.
Robot pose and XYZ remain unavailable for that recording.
When browsing a folder, only runs with recorded policy commands are included;
startup-only attempts and sensor checks are excluded.
Playback, camera and depth appear first; diagnostics start collapsed. It uses
recorded live XYZ when present, otherwise cached offline XYZ. Default checks
validate the whole frame index and decode only three camera samples. Add
`--full-validation` to decode every frame, `--prepare-only` for checks without
Viser, or `--export-video` to encode `review/external.mp4` for sharing.
You can pass just the run name instead of a full path; replay looks it up under
`logs/real_world/` in this checkout. Existing directory paths still work.

For a stopped run whose telemetry gzip was interrupted, the review command
preserves the original and writes complete recoverable records to
`review/telemetry_recovered.jsonl.gz`, with a recovery report. The end may be
missing; no stop event or robot state is invented. Reused sensor timestamps
are retained and replayed using the recorded control clock, with an explicit
diagnostic. Camera and depth keep their original exposure timestamps.
Viser reads the indexed raw JPEGs on demand, using exposure timestamps
rather than assuming frame number equals a policy tick. It keeps only the current
decoded frame and sends a preview at up to 960 pixels wide; the MP4 stays full resolution.

Camera exposure and USB transfer delay remain unqualified. Do not assume a fixed
one-frame delay or treat commanded motor onset as an independent clock reference;
use an independently timestamped visible event for synchronization qualification.
Keep the camera fixed and include measured floor marks for the gross translation
check; a single uncalibrated view is not 3D truth.

## XYZ options and older recordings

### Mark recovery start and first standing

```bash
cd /home/osher/hv2
.venv/bin/python g1_deploy/replay_motion.py logs/real_world --port 8081
```

Open **http://localhost:8081** (forward port 8081 in VS Code). The viewer starts
paused. Already reviewed runs open from their saved index; only the selected
recording is loaded. Do not add `--prepare-only` (it does not open a viewer),
`--preload`, or export flags when marking clips. In each recording:

1. The start defaults to **0 seconds**. Use **Mark start** only to change it.
2. Scrub to the end and click **Mark next**. This marks the end and opens the
   next recording. Repeat for each recording.
3. When finished, click **Save all**.

The **Mark clip** panel shows **NOT MARKED**, **START MARKED**,
**✅ MARKED — not saved yet**, or **✅ MARKED — saved**.
Drafts stay in this viewer until **Save all**; save before closing or restarting.
The recording dropdown lets you revisit a clip. Saved marks reload when you
reopen a recording. For recovery measurements,
mark the end at first standing, before subsequent drifting.

To finish only recordings that have no saved marks:

```bash
.venv/bin/python g1_deploy/replay_motion.py logs/real_world --unmarked --port 8081
```

Each run stores `review/recovery_interval.json`: the exact robot sample indices,
replay timestamps, displayed external-camera exposure references and your label.
Marking does not modify or trim the recording. The analysis uses the marked start
as its origin and the marked standing/end as its cutoff; later drift is excluded.

Once all marks are saved, create trimmed copies:

```bash
.venv/bin/python g1_deploy/trim_marked_videos.py
```

Clips appear under `logs/real_world_trimmed/`, with the same category/run folders.
Each has `external.mp4`, matching `depth_aligned.mp4` when depth was recorded,
and a manifest of the exact source frames. The marked end frame is included.
Original recordings stay unchanged. Already exported, unchanged clips are reused.

Check marking progress or compute the ten-per-posture summary after labeling:

```bash
.venv/bin/python scripts/fall_recovery/validation/summarize_real_world_standing.py --status
.venv/bin/python scripts/fall_recovery/validation/summarize_real_world_standing.py
```

The summary requires saved **First standing** intervals for ten prone and ten
supine recordings. It never falls back to automatic endpoints. Results go to
`logs/analysis/real_world_standing_marked/`; XYZ is reported only when both
marked endpoints are valid. No recovery-time metric is reported.

### XYZ sources

Use only after the policy has stopped. Set `SESSION` to the existing run path:

```bash
cd /home/osher/hv2
SESSION=logs/real_world/YOUR_SESSION_DIRECTORY
.venv/bin/python g1_deploy/replay_motion.py "$SESSION" --port 8081
```

Pass a folder instead of one session (`logs/real_world`, or a category folder
inside it) to browse every recording under it in one viewer, with a dropdown and
Previous/Next buttons; `--preload` loads them all before opening.
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
warmup before allowing startup. This applies to sensor-only checks too. The
countdown is before motor takeover; it adds no wait/check to running control.
Its receipt is saved in `sensors/startup_warmup.json`.
Offline conversion uses the run's `odometry_mount.json` when present. For older
runs, pass `--mount-calibration g1_deploy/config/lidar/mount_candidate_20260912.json`
to use the same candidate as the current live path. Otherwise it uses nominal
geometry. Replay remembers the selected offline result in `review/xyz_selection.json`;
subsequent playback does not silently switch back to nominal geometry.
