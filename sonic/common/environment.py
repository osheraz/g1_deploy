"""Environment for SONIC experiment subprocesses."""
import os
from pathlib import Path


def experiment_environment(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "research/repos/GR00T-WholeBodyControl")
    env["PYTHONNOUSERSITE"] = "1"
    tensorrt = Path(env.get("TENSORRT_ROOT", "/data/clhsieh/software/nvidia/tensorrt/10.13.3.9"))
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, (
        str(tensorrt / "lib"), "/usr/local/cuda/lib64", "/usr/local/lib",
        env.get("LD_LIBRARY_PATH", ""),
    )))
    return env
