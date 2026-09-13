# Local DLIO initialization fix

These files are the tested recording/replay configuration, not a vendored ROS
installation. Simulation geometry/heading tests do not require ROS. Hardware raw
capture and offline DLIO replay do. Set `LIDAR_WS` to your built ROS 2 Humble
workspace; `record_motion.py::DEFAULT_WS` is only the current bluefin scratch path.
The Unitree Python SDK must also be available on the recording process PYTHONPATH.
`MID360_bluefin.json` contains this workstation's network configuration; adjust it
for a different host before starting a driver.

Source revisions in the tested external workspace:

| Repository | Revision |
| --- | --- |
| `livox_ros_driver2` | `4a1def929e5b59c7a8122d19fce6efba581ce9f7` |
| `direct_lidar_inertial_odometry` | `c8acc37100e349d70a9d8432d656cbce7e5072cd` plus the patch below |

`dlio.yaml` expects SI IMU acceleration; the recording/replay wrappers convert
raw Livox g values before DLIO. They preserve raw units in the capture. The
current input uses 30 Hz raw packets grouped in threes, retaining point timestamps
for deskew. No planner FOV crop is added to odometry. Existing DLIO preprocessing
crop/voxel settings remain enabled; dynamic robot self-return rejection is not
qualified by these tests.

`mount_candidate_20260912.json` is an explicit **diagnostic candidate**, copied
unchanged from `logs/lidar_live/mount_candidate.json` on bluefin. The opt-in
`deploy_diffusion.py --live-odometry` path explicitly selects this candidate and
records its path; the standalone bridge still uses its nominal mount unless
passed `--mount-calibration`. It preserves fit/validation
provenance and limitations. For cached replay outputs, continue using the original
calibration argument or regenerate in a fresh output directory: replay cache
provenance includes the calibration path as well as its contents.

`dlio_initialized_pose.patch` prevents the periodic odometry publisher from emitting
positive-timestamp poses before IMU calibration and DLIO initialization. Previously
the pelvis adapter could establish its permanent yaw alignment from those invalid
poses. This is a sensor initialization fix, not a change to planner heading.

The current ROS workspace has this patch applied and is rebuilt with:

```bash
source /opt/ros/humble/setup.bash
export PYTHONNOUSERSITE=1 CMAKE_BUILD_PARALLEL_LEVEL=2
colcon build --packages-select direct_lidar_inertial_odometry --executor sequential
```

Run the build from the workspace root. On a fresh `feature/ros2` checkout, apply the
patch with `git apply /home/osher/hv2/g1_deploy/config/lidar/dlio_initialized_pose.patch`
from the `direct_lidar_inertial_odometry` source directory first. Existing offline
outputs generated with the old node must be regenerated; they are not fixed by a
new viewer. Keep old outputs separately for comparison.
