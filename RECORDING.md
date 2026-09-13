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
