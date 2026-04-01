# Remote

Remote-side G1 deploy runtime.

Copy this folder to the remote machine.

## Setup

```bash
cd remote
pixi run bootstrap
source .envrc
uv pip install -e .
```

Install extras into the same environment if needed:

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
uv pip install -e .
```

Example Torch install:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Run

```bash
cd remote
source .envrc
python3 deploy_g1.py --config g1_23dof.yaml --log TEST.npz
```

## Inputs

- tactile: ROS 2 topics from the robot-side workspace
- depth: ZMQ stream from the robot-side depth server

Set [`configs/g1_23dof.yaml`](/home/osher/e_skin/remote/configs/g1_23dof.yaml) so `depth_server_ip` points to the robot PC.

## Files

- [`deploy_g1.py`](/home/osher/e_skin/remote/deploy_g1.py)
- [`config.py`](/home/osher/e_skin/remote/config.py)
- [`configs/g1_23dof.yaml`](/home/osher/e_skin/remote/configs/g1_23dof.yaml)
- [`common/tactile_client.py`](/home/osher/e_skin/remote/common/tactile_client.py)
- [`common/depth_client.py`](/home/osher/e_skin/remote/common/depth_client.py)
- [`scripts/bootstrap.sh`](/home/osher/e_skin/remote/scripts/bootstrap.sh)
- [`scripts/replay_rollout.py`](/home/osher/e_skin/remote/scripts/replay_rollout.py)
- [`scripts/make_video.py`](/home/osher/e_skin/remote/scripts/make_video.py)
