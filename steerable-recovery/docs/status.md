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
- The server was updated and tested on the robot on September 20. With LiDAR traffic,
  raw transport with `--processing vigor` delivered 30.03 Hz:
  capture-to-arrival median 68.9 ms / p95 70.3 ms,
  zero sequence gaps or stale frames over 45 seconds; recorded depth replay matched.
  One restart had early exposure gaps before settling. Actual policy-consumed age,
  pose pairing and DLIO/inference load remain unqualified. See the
  [live receipt](../../../logs/deployment_validation/depth_gigabit_20260920_160232/REPORT.md).
- Intermittent Jetson SSH failure on the direct link was caused by failed automatic
  network activation. Its profile now explicitly enables IPv6 link-local; see
  [network setup](../../tutorials/01_network.md#jetson-disappears-periodically-on-the-direct-link-fixed-2026-09-20).
- The operator-selected default is now `--processing vigor_legacy`: original
  VIGOR smoothing and hole filling. Filled positive pixels become valid policy
  depth. Explicit `--processing vigor` still preserves source missing/near/far
  categories. R8 observations use metres: remaining below-0.1 m / missing values
  are masked; finite returns above 2 m saturate at 2 m.
- Hole filling differs from Stanley's written unified-depth-v3 raw-processing and
  missing-pixel contract. This default change does not qualify v3 compatibility or
  improved recovery; the timing numbers above describe the earlier `vigor` run.
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
`~/.local/share/hv2/lidar/ros2_ws`; [build recipe](../config/lidar/README.md).

## Simulation and lab terrain

MuJoCo uses the training URDF and Warp depth, but differs from Isaac contacts and
hardware timing. Do not transfer selected seeded success rates across these setups.
The requested search for 9×24×36-inch lab platforms is separate work; no new safe
starting configuration is claimed by this reorganization.

Dated investigations remain in `notes/validation/`; they are historical evidence,
not an additional sequence of steps required before each run.
