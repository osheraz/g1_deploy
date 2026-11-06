"""Exercise launcher arguments without a simulator, DDS or motor publisher."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def launch(tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts/joystick"
    scripts = tmp_path / "g1_deploy/sonic/scripts/joystick"
    shutil.copytree(source, scripts)
    companion = tmp_path / "sonic_cpp/teleop"
    companion.mkdir(parents=True)
    (companion / "core.common.sh").write_text("""
validate_sonic_model_profile_name() {
    case "$1" in low-latency|original-release|sonic-v1.1) return 0;; *) return 1;; esac
}
normalize_sonic_adaptation() { printf '%s\\n' "$1"; }
""")
    capture = tmp_path / "capture"
    capture.write_text(f"#!{sys.executable}\n" + """
import json, sys
if sys.argv[1:2] == ['matrix']:
    print('E_SKIN_SELECTED_PYTHON_CONFIG=official.yaml')
    print('E_SKIN_SELECTED_FORWARD_STABLE_23DOF_CONFIG=forward.yaml')
    print('E_SKIN_SELECTED_MJCF=robot.xml')
    print('E_SKIN_SELECTED_MODEL_PROFILE=original-release')
else:
    print(json.dumps(sys.argv[1:]))
""")
    capture.chmod(0o755)
    shutil.copy2(capture, companion / "t1_planner_joystick_robot_23dof.sh")
    env = dict(
        os.environ,
        G1_PYTHON=str(capture),
        E_SKIN_TELEOP_PYTHON=str(capture),
        E_SKIN_DEPLOYMENT_MATRIX="matrix",
        TensorRT_ROOT=str(tmp_path / "tensorrt"),
    )

    def run(script, *args):
        return subprocess.run(
            ["bash", str(scripts / script), *args],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
        )

    return run


def command(result):
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.splitlines()[-1])


def test_mujoco_keeps_selected_inputs_and_adaptation(launch):
    args = command(
        launch(
            "mujoco.sh",
            "--adaptation",
            "forward-stable-23",
            "--sagittal-scale",
            "0.85",
            "--knee-gait-offset-deg",
            "4",
            "--unitree-remote",
            "--interface",
            "test0",
            "--ground-friction",
            "0.7",
            "--planner-replan-period",
            "0.4",
            "--no-idle-hold",
            "--no-record",
            "--viser",
        )
    )
    assert args[0].endswith("/g1_deploy/sonic/sim.py")
    for name, value in {
        "--config": "forward.yaml",
        "--reference-joint-scale": "sagittal=0.85",
        "--reference-joint-offset-deg": "knee=4",
        "--interface": "test0",
        "--ground-friction": "0.7",
        "--planner-replan-period": "0.4",
    }.items():
        assert args[args.index(name) + 1] == value
    assert {"--remote", "--viser", "--no-idle-hold"}.issubset(args)
    assert "--record-dir" not in args


def test_mujoco_keeps_slow_gait_and_recording(launch):
    args = command(launch("mujoco.sh", "--model-profile", "original-release",
                          "--embodiment", "g1_23dof", "--adaptation", "forward-stable-23",
                          "--slow-gait"))
    assert {"--slow-gait", "--record-dir", "--native-gamepad"}.issubset(args)
    assert args[args.index("--config") + 1] == "forward.yaml"


@pytest.mark.parametrize("script", ["robot.sh", "robot_cpp.sh"])
def test_robot_launcher_does_not_arm_by_default(launch, script):
    args = command(launch(script))
    assert "--arm-real-robot" not in args
    if script == "robot.sh":
        assert "--preflight-only" in args


@pytest.mark.parametrize(
    "args",
    [
        ["--landing-hold"],
        ["--adaptation", "stable-slow-23"],
        ["--adaptation", "forward-stable-23", "--slow-gait", "--hip-pitch-scale"],
        ["--slow-gait"],
    ],
)
def test_mujoco_rejects_retired_or_incompatible_options(launch, args):
    result = launch("mujoco.sh", *args)
    assert result.returncode == 2
    assert "ERROR:" in result.stderr
