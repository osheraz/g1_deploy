# Replay tools

For the current A-triggered diffusion/flow capture and relative-XYZ replay, see
[RECORDING.md](../recovery/recording.md). `g1_deploy/steerable-recovery/steerable_recovery/review/session.py SESSION` prepares
offline pelvis motion and feeds it to this same episode player.

`replay_robot_diagnostics.py` opens a Viser comparison of measured and
commanded 23-DoF robot joint state from a completed physical recording. It is
read-only and does not connect to the robot.

The existing episode player also reads native diffusion/flow deployment telemetry
(all recorded joints; no second recorder or conversion needed):

```bash
cd /home/osher/hv2
.venv/bin/python g1_deploy/core/review/viewer.py \
  --telemetry logs/native_recovery/robot_diffusion_tensorrt_1789255983555602214 \
  --urdf humanoidverse2/data/robots/g1/g1_29dof_sonic.urdf --port 8081
```

Open forwarded port 8081. Play/Pause, Index and Restart control the saved recording.
Native logs provide measured joints and pelvis IMU orientation. The root stays at a
fixed display position; do not interpret it as measured world translation. Raw depth
images and torso IMU are not present in native telemetry. Original `--episode` input
continues to support the existing camera/tactile episode format.
