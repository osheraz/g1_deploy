# Remote

This folder contains the remote-compute deployment code for the G1 robot.

In this repository, responsibilities are split across two machines:

- Robot PC:
  - runs the tactile ROS 2 drivers in `ros_ws/src/jq_tactile_driver`
  - owns the USB/serial connection to the e-skin suit
- Remote PC:
  - runs `remote/deploy_g1.py`
  - talks to the G1 over Unitree DDS
  - receives depth through the existing depth client
  - receives tactile through ROS subscriptions
  - logs synchronized rollout data

## What Runs Here

Main runtime:

- [`deploy_g1.py`](/home/osher/e_skin/remote/deploy_g1.py)

Support modules:

- [`config.py`](/home/osher/e_skin/remote/config.py)
- [`robot/robot_interface.py`](/home/osher/e_skin/remote/robot/robot_interface.py)
- [`policy/policy_runner.py`](/home/osher/e_skin/remote/policy/policy_runner.py)
- [`common/tactile_client.py`](/home/osher/e_skin/remote/common/tactile_client.py)

Utility scripts:

- [`scripts/replay_rollout.py`](/home/osher/e_skin/remote/scripts/replay_rollout.py)
- [`scripts/make_video.py`](/home/osher/e_skin/remote/scripts/make_video.py)

## Environment

The repo already uses Pixi + UV at the workspace level, and that is the preferred path
here too.

Recommended approach:

1. Use Pixi for ROS 2 and system-level robotics dependencies.
2. Use UV for Python packages and editable installs.
3. Install the Unitree SDK into the same virtual environment with `uv pip`.

## Install

### 1. Create the Workspace Environment

From `ros_ws/`:

```bash
cd ros_ws
pixi run setup
source .envrc
```

That gives you:

- ROS 2 Humble from Pixi
- a workspace `.venv`
- Python packages managed by UV

### 2. Install Remote Python Dependencies

Install the extra Python packages used by the deploy/runtime side into the same UV
environment:

```bash
uv add pyyaml scipy onnxruntime omegaconf termcolor pillow imageio viser opencv-python
```

If you want the remote package installed in editable mode:

```bash
uv pip install -e ../remote
```

### 3. Install Unitree SDK

Clone the SDK wherever you keep external repos, then install it into the active UV env:

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
uv pip install -e .
```

### 4. Install Torch If Needed

This repo does not pin a Torch wheel in `pyproject.toml`, because the correct wheel
depends on your CUDA/runtime setup. Install the wheel that matches your machine.

Example for CUDA 12.1:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Robot + Sensor Setup

### G1 Connection

Before running deploy:

1. Put the G1 into the proper low-level / developer mode.
2. Connect the remote PC to the robot over ethernet.
3. Make sure the correct network interface is active on the remote PC.

`deploy_g1.py` currently calls `ChannelFactoryInitialize(0)` directly, so it is not
yet using the `--net` argument to select a DDS interface.

### Tactile ROS Setup

On the robot PC:

```bash
cd ros_ws
source install/setup.bash
ros2 launch jq_tactile_driver jq_tactile_all.launch.py
```

The tactile driver publishes:

- `/jq_tactile/<role>/<sensor>/raw_sensor_bytes`
- `/jq_tactile/<role>/<sensor>/raw_gyro_bytes`
- `/jq_tactile/<role>/<sensor>/frame_info`
- `/jq_tactile/<role>/<sensor>/capture_time_ns`
- `/jq_tactile/<role>/<sensor>/frame_seq`

On the remote PC:

- source the same ROS environment before running deploy
- make sure both machines can see each other in ROS 2
- use the same `ROS_DOMAIN_ID`
- make sure DDS discovery/networking is allowed between the two machines

## Config

The default remote config is:

- [`configs/g1_23dof.yaml`](/home/osher/e_skin/remote/configs/g1_23dof.yaml)

Important fields:

- `use_depth_obs`
- `use_tactile`
- `tactile_sensor_order`
- `tactile_max_age_s`
- `policy_path`
- `lowcmd_topic`
- `lowstate_topic`

To enable tactile logging:

```yaml
use_tactile: true
tactile_sensor_order: ["LF", "RF", "WB"]
tactile_max_age_s: 0.05
```

## Run

From the repo root on the remote PC:

```bash
source ros_ws/.envrc
python3 remote/deploy_g1.py --config g1_23dof.yaml --log TEST.npz
```

## Logs

Real rollout logs are written under:

- `remote/logs/real_rollouts/`

## Utility Scripts

Replay a rollout:

```bash
python3 remote/scripts/replay_rollout.py --log balance_board/TEST_1234567890.npz
```

Render rollout videos:

```bash
python3 remote/scripts/make_video.py
```

## Notes

- The remote code still contains some hardcoded paths to the HumanoidVerse/ASAP training
  tree, especially for `policy_path` and URDF assets.
- [`config.py`](/home/osher/e_skin/remote/config.py) still points at an external config
  root.
- The `--net` CLI argument is present in `deploy_g1.py` but is not currently wired into
  `ChannelFactoryInitialize`.
