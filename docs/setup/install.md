# Installation

Use the existing HV2 environment when available:

```bash
cd /home/osher/hv2
source .venv/bin/activate
python -m pip install --no-deps --no-build-isolation -e . -e ./g1_deploy
python g1_deploy/rise/deploy.py --help
```

Another workstation needs the matching HV2 checkout, pinned `g1_deploy` submodule,
model bundle, and a compatible Torch/CUDA environment. Dependencies are listed in
[`pyproject.toml`](../../pyproject.toml) and
[`hardware-requirements.in`](../../rise/hardware-requirements.in).
Install Unitree SDK2 Python and CycloneDDS for robot access; preserve an existing
working installation. Rebuild TensorRT engines on the target GPU.

The camera host uses an isolated environment with NumPy, OpenCV, pyzmq and
pyrealsense2. It does not need Torch or Unitree DDS.
