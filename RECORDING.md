# Recording and replay options

For the normal live XYZ trial, use [the operator guide](tutorials/deploy_diffusion.md).
It includes the depth command, policy command, A/stop controls and live replay.
Do not run a second recorder beside deployment.

## What is saved

| File inside the session | Contents |
| --- | --- |
| `telemetry.jsonl.gz` | Policy state/actions, heading, timing and live odometry when enabled |
| `sensors/raw/sensors.npz` | Raw LiDAR, pelvis/torso IMUs, joint q/dq/estimated torque and timestamps |
| `sensors/raw/bag/` | Timestamped LiDAR and Livox IMU for offline processing |
| `sensors/raw/summary.json` | Sensor counts, gaps and capture errors |
| `sensors/recording_result.json` | Whether recording started and saved completely |

Actuated deployment records raw sensors by default. `--live-odometry` also runs
DLIO and pelvis conversion; without that flag, XYZ can be reconstructed offline.
Sensor-only runs record when `--live-odometry` or `--record-sensors` is selected.
Preflight/default-pose mode does not start this recorder.

The recorder buffers up to five seconds before A and saves after motor cleanup.
It runs separately at lower CPU priority; [measured compute cost](../notes/deployment_compute.md)
is nonzero. It does not archive raw depth images or estimate support contacts.

## External camera

Add `--external-camera /dev/v4l/by-id/<camera>-video-index0` (optional
`--camera-size 1280x720 --camera-fps 30`) to a recorded policy session. The
recorder then starts one `ffmpeg` child at the same low priority that copies the
camera's MJPEG packets into `sensors/camera/external.mkv` without decoding or
re-encoding; the control process never touches the camera. Recording runs from
recorder launch to stop, so the pre-A idle is included. It requests V4L2
wall-clock timestamps and preserves them with `-copyts`; this is a driver
timestamp, not a calibrated exposure time. `sensors/session.json` stores
`camera.wall_minus_monotonic_s` and `sensors/policy_start.json` stores the A press
in both clocks. `recording_result.json` reports packet count, first/last timestamps,
packet-span duration and gaps above 1.5 nominal frame periods. These diagnostics
do not prove that the camera dropped no frames. Camera failure is reported with
the saved recording; it does not stop the policy.

Before the first session with a camera, check its formats and record 300 frames (about 10 s at 30 Hz):

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
ffmpeg -copyts -f v4l2 -ts abs -input_format mjpeg -framerate 30 -video_size 1280x720 -i /dev/video0 \
  -c:v copy -vsync passthrough -avoid_negative_ts disabled -frames:v 300 /tmp/cam_test.mkv
ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets /tmp/cam_test.mkv
```

Cut the video at A and convert it for viewing (offline, any machine):

```bash
A=$(python3 -c "import json;print(json.load(open('$SESSION/sensors/policy_start.json'))['wall_s'])")
T0=$(ffprobe -v error -show_entries format=start_time -of csv=p=0 "$SESSION/sensors/camera/external.mkv")
ffmpeg -ss "$(python3 -c "print(max($A-$T0,0))")" -i "$SESSION/sensors/camera/external.mkv" -c:v libx264 -crf 18 "$SESSION/sensors/camera/from_A.mp4"
```

Exposure timing, camera clock conversion and USB delay remain unqualified.
Do not assume a fixed one-frame delay or treat commanded motor onset as an
independent clock reference. Use an independently timestamped visible event for
synchronization qualification. Keep the camera fixed and include measured floor
marks for the gross translation check; a single uncalibrated view is not 3D truth.
The timestamp path has synthetic-file tests; a physical USB camera still needs
validation. See [FFmpeg timestamp preservation](https://ffmpeg.org/ffmpeg.html#Advanced-options)
and [V4L2 timestamps](https://ffmpeg.org/ffmpeg-devices.html#video4linux2_002c-v4l2).

## Older recordings without live XYZ

Use only after the policy has stopped. Set `SESSION` to the existing run path:

```bash
cd /home/osher/hv2
SESSION=logs/native_recovery/YOUR_SESSION_DIRECTORY
.venv/bin/python g1_deploy/replay_motion.py "$SESSION" \
  --mount-calibration g1_deploy/config/lidar/mount_candidate_20260912.json --port 8081
```

This runs DLIO offline and opens Viser. It starts no sensor driver or robot
connection. Offline interpolation differs from what was available during control;
use `--live-estimates` for sessions recorded with live odometry. Add
`--prepare-only` to process without opening Viser. `--pose-only` shows only
joints/orientation when raw data or XYZ is unavailable.

The ROS workspace defaults to `~/.local/share/hv2/lidar/ros2_ws`; `LIDAR_WS`
overrides it. [Build instructions](config/lidar/README.md).
