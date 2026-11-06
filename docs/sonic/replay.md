# Replay tools

For A-triggered recovery recording and pelvis XYZ replay, see
[RISE recording](../rise/recording.md). `g1_deploy/rise/replay.py SESSION`
prepares pelvis motion for the same episode player.

`replay_robot_diagnostics.py` opens a Viser comparison of measured and
commanded 23-DoF robot joint state from a completed physical recording. It is
read-only and does not connect to the robot.

The existing episode player also reads native diffusion/flow deployment telemetry
(all recorded joints; no second recorder or conversion needed):

```bash
cd /home/osher/hv2
.venv/bin/python g1_deploy/core/review/viewer.py \
  --telemetry logs/real_world/RUN \
  --urdf humanoidverse2/data/robots/g1/g1_29dof_sonic.urdf --port 8081
```

Open forwarded port 8081. Use Play/Pause, Index and Restart for playback.
Native logs contain measured joints and pelvis IMU orientation. Root position is
fixed for display; it is not measured world translation.

Native telemetry has no raw depth or torso IMU. Use `--episode` for recordings
in the camera/tactile episode format.
