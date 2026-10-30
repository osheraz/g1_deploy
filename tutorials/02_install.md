# Environment

## This workstation

Use the existing hv2 environment. Do not reinstall Torch, Isaac, or the robot
SDK merely to launch deployment.

```bash
cd /home/osher/hv2
source .venv/bin/activate
python -m pip install --no-deps --no-build-isolation -e ./g1_deploy
python g1_deploy/steerable-recovery/deploy.py --help
python -c "import torch, numpy, cv2, zmq, unitree_sdk2py, viser"
```

The Python entrypoint adds the parent checkout and its bundled
`dependencies/isaac_utils` helpers to the import path. Those helpers do not boot
IsaacSim. The model implementation remains shared with simulation.

## Matching source checkout

Deployment needs both the parent hv2 code and its pinned `g1_deploy` submodule.
In a clean deployment checkout following `master`:

```bash
git pull --ff-only
git submodule update --init g1_deploy
```

This is not an instruction to merge or overwrite a separate training branch.
For a fresh checkout, clone hv2 and initialize only this submodule:

```bash
git clone https://github.com/osheraz/hv2.git hv2-deploy
cd hv2-deploy
git submodule update --init g1_deploy
```

On Bluefin, the existing environment, model bundle and TensorRT exports are
configured already. Another machine needs access to the bundle path in
`g1_deploy/steerable-recovery/config/models.yaml` (or a local materialized copy), its own network
interface, and newly validated TensorRT engines. No private SSH control socket
is required by the documented camera command. The camera host itself still
requires authorized SSH access.

## Another workstation

Use Python 3.11 or newer and a CUDA/Torch environment appropriate for that
machine. The model dependency list is
[`requirements.in`](../steerable-recovery/hardware-requirements.in);
it is not a pinned replacement for an existing training environment.
`g1_deploy/pyproject.toml` lists viewer/robot-tool dependencies.

Real robot access additionally needs Unitree SDK2 Python and CycloneDDS.
Preserve a working installation and its `CYCLONEDDS_HOME` configuration.
Bundle checks do not require camera hardware or the robot SDK.

For compiled inference, follow the [export tooling](../steerable-recovery/docs/performance.md).
Build/validate engines on the target GPU; copying an engine alone is insufficient.

## Jetson camera host

The isolated camera environment is
`/home/unitree/osher/native_recovery/.venv`. It needs NumPy, OpenCV, pyzmq and
pyrealsense2, not Torch or the robot SDK. See [camera commands](04_cameras.md)
and the [migration record](08_robot_migration.md). Do not replace vendor Python.
