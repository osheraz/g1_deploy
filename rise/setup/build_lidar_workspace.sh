#!/usr/bin/env bash
# Build pinned sensor dependencies without starting sensors or changing system packages.
set -eo pipefail
config_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../config/lidar" && pwd)"
stack_dir="${LIDAR_STACK_ROOT:-$HOME/.local/share/hv2/lidar}"
with_map=false
if [[ "${1:-}" == --with-map ]]; then with_map=true; shift; fi
if [[ $# != 0 ]]; then
  echo 'Usage: bash g1_deploy/rise/setup/build_lidar_workspace.sh [--with-map]' >&2; exit 2
fi

# Do not embed an active virtualenv or old ROS overlay in the new install.
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
source /opt/ros/humble/setup.bash
export PYTHONNOUSERSITE=1 CMAKE_BUILD_PARALLEL_LEVEL="${LIDAR_BUILD_JOBS:-2}"
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
mkdir -p "$stack_dir/ros2_ws/src"
stack_dir="$(cd -- "$stack_dir" && pwd)"

checkout() {
  local destination="$1" url="$2" revision="$3"
  if [[ ! -e "$destination" ]]; then
    git clone "$url" "$destination"
    git -C "$destination" checkout --detach "$revision"
  fi
  if [[ "$(git -C "$destination" rev-parse HEAD)" != "$revision" ]]; then
    echo "Unexpected revision at $destination; preserving your checkout. Expected $revision" >&2
    exit 1
  fi
}

apply_once() {
  local repository="$1" patch="$2"
  if git -C "$repository" apply --reverse --check "$patch" 2>/dev/null; then return; fi
  git -C "$repository" apply --check "$patch"
  git -C "$repository" apply "$patch"
}

checkout "$stack_dir/Livox-SDK2" https://github.com/Livox-SDK/Livox-SDK2 \
  08f523c930b2f0ba1e98a6afaa8d7476bf479908
checkout "$stack_dir/ros2_ws/src/livox_ros_driver2" https://github.com/Livox-SDK/livox_ros_driver2 \
  4a1def929e5b59c7a8122d19fce6efba581ce9f7
checkout "$stack_dir/ros2_ws/src/direct_lidar_inertial_odometry" https://github.com/vectr-ucla/direct_lidar_inertial_odometry \
  c8acc37100e349d70a9d8432d656cbce7e5072cd
apply_once "$stack_dir/Livox-SDK2" "$config_dir/livox_system_spdlog.patch"
dlio_dir="$stack_dir/ros2_ws/src/direct_lidar_inertial_odometry"
apply_once "$dlio_dir" "$config_dir/dlio_pose_snapshot.patch"

driver_dir="$stack_dir/ros2_ws/src/livox_ros_driver2"
if [[ -e "$driver_dir/package.xml" ]] && ! cmp -s "$driver_dir/package.xml" "$driver_dir/package_ROS2.xml"; then
  echo "Existing $driver_dir/package.xml differs from ROS 2 template; inspect it before rebuilding." >&2; exit 1
fi
cp "$driver_dir/package_ROS2.xml" "$driver_dir/package.xml"

packages=(livox_ros_driver2 direct_lidar_inertial_odometry)
if $with_map; then
  checkout "$stack_dir/ros2_ws/src/elevation_mapping_cupy" https://github.com/leggedrobotics/elevation_mapping_cupy \
    a728e5498876bddce77ea72b1b478e42024cd3ce
  checkout "$stack_dir/ros2_ws/src/ros2_numpy" https://github.com/Box-Robotics/ros2_numpy \
    d5ebf723e1d2457d36fa89bf021b212e340c3c11
  packages+=(elevation_mapping_cupy ros2_numpy)
fi

cmake -S "$stack_dir/Livox-SDK2" -B "$stack_dir/sdk_build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$stack_dir/local"
cmake --build "$stack_dir/sdk_build" --parallel "$CMAKE_BUILD_PARALLEL_LEVEL"
cmake --install "$stack_dir/sdk_build"
cd "$stack_dir/ros2_ws"
colcon build --packages-up-to "${packages[@]}" --executor sequential --cmake-args \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DROS_EDITION=ROS2 -DDISTRO_ROS=humble \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DCMAKE_PREFIX_PATH="$stack_dir/local" -DCMAKE_INSTALL_RPATH="$stack_dir/local/lib"
echo "Built sensor workspace: $stack_dir/ros2_ws"
echo 'No sensor processes were started. Mapping, if built, remains optional and unvalidated live.'
