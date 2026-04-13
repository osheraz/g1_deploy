# Remote

Remote-side G1 deploy runtime.

Short version:

- Shared quickstart in `e_skin`:
  [`osheraz/e_skin/docs/quickstart.md`](https://github.com/osheraz/e_skin/blob/master/docs/quickstart.md)

Copy this folder to the remote machine.

## Setup

```bash
cd remote
pixi run bootstrap
source .envrc
uv pip install -e .
```

Install Unitree SDK into the same environment:

```bash
export REMOTE_PIXI_PREFIX="$PWD/.pixi/envs/default"
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
export CYCLONEDDS_HOME="$REMOTE_PIXI_PREFIX"
export CMAKE_PREFIX_PATH="$REMOTE_PIXI_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
uv pip install -e .
```

Example Torch install:

```bash
uv pip install pyzmq
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Run

```bash
cd remote
source .envrc
python3 deploy_g1.py --config g1_23dof.yaml --log TEST.npz
```

Safe stream check without sending robot actions:

```bash
cd remote
source .envrc
python3 check_streams.py --config g1_23dof.yaml
```

## Inputs

- tactile: ROS 2 topics from the robot-side workspace
- depth: ZMQ stream from the robot-side depth server

Set [`configs/g1_23dof.yaml`](configs/g1_23dof.yaml) so `depth_server_ip` points to the robot PC.

## Notes

- `check_streams.py` is a read-only checker. It never sends robot commands.
- The robot state comes from Unitree DDS (`rt/lowstate`).
- Tactile comes from ROS 2 topics published by `ros_ws/`.
- Depth comes from the robot-side ZMQ server.
- The checker isolates the tactile ROS subscriber so one command can show robot state and tactile together in Viser.
- `deploy_g1.py` uses the same three inputs, but it also runs the policy and sends actions to the robot.

## Files

- [`deploy_g1.py`](deploy_g1.py)
- [`config.py`](config.py)
- [`configs/g1_23dof.yaml`](configs/g1_23dof.yaml)
- [`common/tactile_client.py`](common/tactile_client.py)
- [`common/depth_client.py`](common/depth_client.py)
- [`scripts/bootstrap.sh`](scripts/bootstrap.sh)
- [`tools/replay_rollout.py`](tools/replay_rollout.py)
- [`tools/make_video.py`](tools/make_video.py)
