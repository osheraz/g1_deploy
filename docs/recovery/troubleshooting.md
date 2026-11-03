# Open work and troubleshooting

This page records limitations; [deploy.md](deploy.md) contains the run commands.

## Depth

- Capture rate and age differ: 30 Hz can still deliver old frames. Inspect capture
  age, arrival age and delivered sequences in the saved run.
- A depth viewer working does not establish clock pairing. Deployment needs both
  forwarded ports **15556 and 15558**. Reuse one server and tunnel.
- Check the workstation link: `cat /sys/class/net/enp132s0/speed`. Raw 640×480×30
  uint16 depth needs about 148 Mb/s per subscriber before overhead. Use gigabit,
  or the documented PNG cap on 100 Mb/s; close extra depth subscribers for a trial.
- Match processing, intrinsics, crop and mount to the selected checkpoint.
  Raw and filtered streams are distinct contracts; hole-filled pixels must not
  silently change a model's missing-depth semantics.
- Before A, missing paired depth keeps preview waiting. During recovery the
  existing missing-depth/timing fallback continues; inspect warnings and masks.

## XYZ / steering

Live DLIO + waist FK and the causal position filter are implemented. `estimated`
feedback is an explicit experimental option, not a claim of verified accuracy.
Independent clock-offset bounds, dynamic XYZ accuracy and the cause of tilt
rejections remain open. Use recorded fixed-camera floor references for evaluation.
Filtering reduces jitter; it does not remove drift or justify filling invalid feedback.

Contact estimation and live ETH elevation-map input remain unqualified/deferred.
Depth is still the planner input. The permanent ROS workspace is
`~/.local/share/hv2/lidar/ros2_ws`; [build recipe](../setup/lidar.md).

## Simulation and lab terrain

MuJoCo uses the training URDF and Warp depth, but differs from Isaac contacts and
hardware timing. Do not transfer selected seeded success rates across these setups.
