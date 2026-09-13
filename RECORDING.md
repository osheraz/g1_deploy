# A-button recording and motion replay

**Current next step:** the stationary check is complete. Use
[the ground tracking test procedure](../notes/ground_tracking_test.md) for the full
depth/deployment/replay commands and what to inspect. The hoist commands below
describe the earlier stationary verification workflow.

Keep the robot supported on the hoist. The recorder has no ground-pose requirement
and does not send motor commands. These tests exercise capture, timing and motion
estimation; they do not establish foot/hand/knee contact accuracy.

On bluefin, keep the existing depth-stream terminal running. If it is not running,
start it in terminal 1:

```bash
ssh -t -o AddressFamily=any -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:15556:127.0.0.1:5556 \
  -L 127.0.0.1:15558:127.0.0.1:5558 \
  g1_29 \
  '/home/unitree/osher/native_recovery/.venv/bin/python /home/unitree/osher/native_recovery/native_depth_server.py --serial 247122073927 --bind tcp://127.0.0.1:5556 --clock-bind tcp://127.0.0.1:5558'
```

Terminal 2, deployment (choose a new output directory for each run):

```bash
cd /home/osher/hv2
.venv/bin/python g1_deploy/deploy_diffusion.py \
  --model diffusion --backend tensorrt --actuate --release-controller \
  --duration-s 60 --output logs/native_recovery/hoist_A_01
```

If a Livox driver is already running, append `--existing-lidar-driver`. Do not
run a separate `record_motion.py` process alongside this command. This integration
reuses that recorder. Use `--no-record-sensors` for a policy-only baseline; native
policy telemetry remains enabled. Flow uses the same lifecycle with `--model flow`.

Keep the robot still while inference warms up and the recorder discovers streams.
Recorder discovery runs in the background and does not block policy startup.
Missing streams or recorder errors are printed; available data is still saved,
but missing LiDAR means that trial cannot validate XYZ. Prefer a fresh driver
owned by this deployment over reusing one left running across a robot restart.
Once the policy is ready, press **A** to start policy and capture. The next **A**,
**SELECT**, **Ctrl+C**, the duration limit or a policy fault ends the session.
Motor cleanup precedes the blocking sensor save. Wait for **Saved and verified**.
Saving now freezes callbacks and writes arrays before finalizing the bag. The
dedicated capture process exits without invoking Unitree listener destructors,
which can block. Saving progress is printed; unsaved captures are not killed by
the former 45-second timeout.
START retains its optional default-pose behavior; it does not start the policy
recording. First-A requests rejected by policy readiness do not start recording.

The sensor process buffers up to five seconds before A for offline estimator
initialization. Arming does not grow this buffer indefinitely. A sends one small
nonblocking pipe message; there is no ROS startup or estimator work on the policy
thread. Capture runs in separate processes at lower CPU scheduling priority.
Final NPZ compression happens after motor cleanup. CPU/network/disk load still
exists; compare telemetry timing with and without recording on the hoist.

Everything is grouped under the deployment Session directory:

- `telemetry.jsonl.gz`: existing policy/control log, measured joint pose and pelvis
  orientation, commands, heading and timing.
- `sensors/raw/sensors.npz`: raw LiDAR points/timestamps, pelvis and torso IMUs,
  29-joint q/dq/estimated torque and receipt clocks.
- `sensors/raw/bag/`: LiDAR and Livox IMU ROS messages for offline DLIO.
- `sensors/raw/summary.json`: stream rates/gaps, start-request/receipt timestamps,
  save status and optional missing torso stream.
- `sensors/recording_result.json`: whether recording started and saved completely.

Depth remains the planner input. The new raw recorder does not duplicate depth
capture, run DLIO, estimate contacts or launch a viewer during deployment. Raw
depth image playback is not available from native telemetry.

After deployment has stopped, run the viewer yourself:

```bash
cd /home/osher/hv2
.venv/bin/python g1_deploy/replay_motion.py \
  logs/native_recovery/hoist_A_01 --port 8081
```

The first run processes the saved bag through DLIO and the existing fixed-mount /
measured-waist pelvis estimator. Subsequent runs reuse those results. It starts no
sensor driver and connects to no robot. `LIDAR_WS` selects a moved ROS workspace.
Use `--prepare-only` to compute results without launching Viser.

Forward port 8081 in VS Code and open http://localhost:8081. Use Play, Index and
Restart. The camera stays in the world frame. The robot translates relative to its
position at the first policy frame; XY uses steering's recorded initial heading,
and Z remains gravity-aligned. A path and numeric XYZ make that motion visible.
The display also shows the shared steering-heading change and full orientation
relative to the starting pelvis as an xyzw quaternion. These are different quantities.
The grid is a ruler, not an estimated floor height. Position is an **offline LiDAR
estimate**, not verified ground truth. Tilt disagreements remain visible with a
diagnostic label; missing position intervals hide the robot rather than inventing
motion. Optional `--mount-calibration PATH` selects an explicit calibration candidate.

Old telemetry-only sessions (including the interrupted `prone_03` sensor capture)
can still replay joints/orientation with `--pose-only`. They do not acquire XYZ
merely by changing the camera or moving the displayed pelvis. The missing NPZ
from that old capture remains missing; the save bug is fixed for new recordings.

## What this establishes for steering

The default A-button command records raw measurements; DLIO runs afterward.
It does **not** currently provide live LiDAR position to the running planner.
The provider and UDP receiver exist, but absolute timestamp error, live pose age,
dynamic tracking accuracy and the launcher wiring remain unfinished. In particular,
the receiver's 50 ms freshness contract has not been validated on live output.
DLIO publishes IMU-propagated poses at 100 Hz between scan corrections; the 100 ms
scan grouping alone does not determine the pose update period or measurement age.
Offline interpolation must not be presented as a live 50 Hz pose estimate.

For each matched policy time, `motion_replay/pose_nominal/deployment_measurements.npz`
contains `episode_time_s`, `position_reset_m`, `height_rel_m`, `yaw_rel_rad`,
`orientation_relative_xyzw` and `pelvis_quat_in_start_heading_xyzw`.
Its neighboring JSON describes the frames and limitations. `pose_calibrated` is
used when an explicit mount calibration is selected. Here relative height is Z
displacement from the start, not height above the ground.

The existing `causal_forward_spine_heading_v1` is shared with simulation and already
runs in hardware. Keep it: a tipped/inverted robot needs a consistent heading
convention. It handles geometric degeneracy, not IMU bias or DLIO registration drift.
The initial world-yaw alignment is applied once; it must not be repeatedly reset to
make disagreement disappear. The pelvis IMU defines the runtime frame, the torso
IMU checks waist/mount geometry, and the Livox IMU is used by DLIO. These are not a
completed joint fusion of all three IMUs.

The supported-hoist test should include still startup, normal operator-started policy
motion, and a fixed-camera video if possible. We will inspect stationary drift,
trajectory versus visible pelvis movement, estimator/IMU yaw disagreement, sensor
gaps and policy latency. Legs can move while pelvis XYZ barely changes on a hoist;
that is not by itself a tracking failure. This does not validate loaded contacts or
the extreme occlusion/impact conditions of floor recovery. Six-channel hardware
contact estimation remains unimplemented.
