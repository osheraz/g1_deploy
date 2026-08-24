# 2. Install

Python ≥ 3.11. Plain [uv](https://docs.astral.sh/uv/) — no pixi, no conda.

```bash
cd g1_deploy
uv venv
source .venv/bin/activate
uv pip install -e .
```

That covers numpy, opencv, onnxruntime, viser, omegaconf, scipy.

## Torch

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Required even for ONNX policies — `vigor/policy_runner.py` imports torch unconditionally.

## Unitree SDK

DDS comms. CycloneDDS has to be built first:

```bash
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x ~/cyclonedds
cd ~/cyclonedds && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/cyclonedds/install && cmake --build . --target install

export CYCLONEDDS_HOME=~/cyclonedds/install
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git ~/unitree_sdk2_python
uv pip install -e ~/unitree_sdk2_python
```

Keep `CYCLONEDDS_HOME` exported in your shell profile — the SDK needs it at import time.

## Check

```bash
python -c "import unitree_sdk2py, cv2, numpy, torch; print(numpy.__version__, cv2.__version__)"
```

## Running scripts under `sonic/scripts/`

Python puts the *script's* directory on `sys.path`, not your cwd, so scripts importing
`common` need the repo root:

```bash
PYTHONPATH=. python sonic/scripts/<script>.py
```

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `_ARRAY_API not found` | numpy 2 with a conda opencv 4.6 built against the 1.x ABI — upgrade opencv or `pip install "numpy<2"`. Shouldn't happen in a clean uv venv. |
| `No module named 'common'` | run with `PYTHONPATH=.` |
| SDK import fails on cyclonedds | `CYCLONEDDS_HOME` not exported |

---

Next: [3. Config](03_config.md)
