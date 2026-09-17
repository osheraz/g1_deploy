# Sensor workspace and local fixes

These files are the tested recording/replay configuration, not a vendored ROS
installation. Simulation geometry/heading tests do not require ROS. Hardware raw
capture and offline DLIO replay do. The default workspace is
`~/.local/share/hv2/lidar/ros2_ws`; set `LIDAR_WS` to override it.
The Unitree Python SDK must also be available on the recording process PYTHONPATH.
`MID360_bluefin.json` contains this workstation's network configuration; adjust it
for a different host before starting a driver.

Source revisions in the tested external workspace:

| Repository | Revision |
| --- | --- |
| `Livox-SDK2` | `08f523c930b2f0ba1e98a6afaa8d7476bf479908` plus `livox_system_spdlog.patch` |
| `livox_ros_driver2` | `4a1def929e5b59c7a8122d19fce6efba581ce9f7` |
| `direct_lidar_inertial_odometry` | `c8acc37100e349d70a9d8432d656cbce7e5072cd` plus the patch below |
| Optional `elevation_mapping_cupy` | `a728e5498876bddce77ea72b1b478e42024cd3ce` |
| Optional `ros2_numpy` | `d5ebf723e1d2457d36fa89bf021b212e340c3c11` |

## Rebuild in a permanent location

On bluefin, ROS 2 Humble and system dependencies are already installed. Rebuild
the pinned driver and DLIO with:

```bash
cd /home/osher/hv2
bash g1_deploy/steerable-recovery/setup/build_lidar_workspace.sh
```

Add `--with-map` to also build the optional ETH packages. Building those packages
does not launch mapping or qualify it for prone recovery. Depth stays the planner
input. The script does not start sensors, install system packages or overwrite a
checkout at a different revision. It applies the two committed patches, builds
the SDK into the permanent `local` prefix, and rebuilds ROS from source so install
paths and library dependencies no longer refer to the scratchpad. Source copies
of the old workspace remain intact for comparison.

`LIDAR_STACK_ROOT` overrides the build root; if changed, set runtime `LIDAR_WS` to
`$LIDAR_STACK_ROOT/ros2_ws`. `LIDAR_BUILD_JOBS` defaults to 2. This is a build recipe
for the installed Humble environment, not a complete OS/dependency installer.
It needs system `libspdlog-dev` alongside the existing ROS/PCL dependencies.

## Livox shutdown fix

The local GDB trace reproduced SIGSEGV in the system `spdlog::logger` destructor
called from the Livox SDK at normal process exit. The SDK bundled a different
spdlog implementation in the same process as ROS. `livox_system_spdlog.patch`
uses the system spdlog headers and linked library for both SDK targets, following
the diagnosis in [Livox-SDK2 issue #44](https://github.com/Livox-SDK/Livox-SDK2/issues/44).
This fixes the library conflict rather than suppressing the crash or skipping
cleanup. No point processing, sensor rates or estimator parameters are changed.

## DLIO initialization and sensor configuration

`dlio.yaml` expects SI IMU acceleration; the recording/replay wrappers convert
raw Livox g values before DLIO. They preserve raw units in the capture. The
current input uses 30 Hz raw packets grouped in threes, retaining point timestamps
for deskew. No planner FOV crop is added to odometry. Existing DLIO preprocessing
crop/voxel settings remain enabled; dynamic robot self-return rejection is not
qualified by these tests.

`mount_candidate_20260912.json` is an explicit **diagnostic candidate**, copied
unchanged from `logs/lidar_live/mount_candidate.json` on bluefin. The opt-in
`deploy.py --live-odometry` path explicitly selects this candidate and
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
patch with `git apply /home/osher/hv2/g1_deploy/steerable-recovery/config/lidar/dlio_initialized_pose.patch`
from the `direct_lidar_inertial_odometry` source directory first. Existing offline
outputs generated with the old node must be regenerated; they are not fixed by a
new viewer. Keep old outputs separately for comparison.
