# Script index

This directory contains compatibility entry points and specialized utilities.
New operator workflow implementations belong under `g1_deploy/workflows/`;
new repeatable parameter sweeps belong under `g1_deploy/experiments/`.

## Compatibility launchers

These stable absolute commands delegate to canonical workflows/presets:

| Script | Canonical responsibility |
| --- | --- |
| `run_forward_stable_23_mujoco.sh` | Current corrected-forward Original Release 23-DoF MuJoCo preset |
| `run_forward_stable_23_cpp.sh` | Current corrected-forward Original Release 23-DoF physical C++ preset |
| `run_planner_joystick_mujoco.sh` | Generic joystick MuJoCo workflow |
| `run_planner_joystick_robot_cpp.sh` | Generic C++ physical joystick workflow |
| `run_planner_joystick_robot.sh` | Generic Python physical/preflight workflow |
| `run_robust_small_gait_23_mujoco.sh` | Fixed robust 23-DoF MuJoCo preset |
| `run_robust_small_gait_23_cpp.sh` | Fixed robust 23-DoF physical C++ preset |
| `run_robust_small_gait_23_robot.sh` | Fixed robust 23-DoF physical Python/preflight preset |

`run_robot_23_cpp.sh` is the locked official-parity Original Release C++
baseline. `run_stable_slow_23_robot.sh` is retained to reproduce the superseded
motor-shaped research profile.

The generic MuJoCo and C++ physical workflows are separate scripts. Each
accepts independent selectors for model, embodiment, and adaptation.
`run_planner_joystick_mujoco.sh` never publishes a physical `LowCmd`.
`run_planner_joystick_robot_cpp.sh` never starts MuJoCo. Both unified
launchers accept `--slow-gait` on `forward-stable-23`. Prefer those
commands for controlled comparisons; fixed presets are convenience aliases.

## Simulation engines and input tools

- `sim2sim_sonic.py`: Python MuJoCo/SONIC engine used by joystick workflows and
  controlled experiments; not a second workflow selector.
- `fake_pico.py`: synthetic PICO-compatible publisher.
- `prepare_sonic_motion.py` / `export_sonic_motion.py`: motion preparation.
- `extract_gait_cycle.py`: **legacy.** Builds mocap-cycle npz for the
  withdrawn hybrid-gait attempt (`--gait-templates`). Not the current
  slow-walk method.
- `eval_tracking_mujoco.py`: headless tracking/stability gate. Its
  `--gait-templates` flag is the same legacy path.

## Calibration and sensing

- `run_zero_torque_monitor_robot.sh`: no-SONIC, zero-torque joint monitor.
- `run_*calibration*.sh|py`: suspended/review-gated calibration execution.
- `generate_*calibration*.py`: deterministic measurement-plan generation.
- `fit_suspended_calibration.py` and `merge_external_calibration_truth.py`:
  offline fitting and external true-angle integration.

Calibration uses no SONIC policy. It identifies the sensor/plant boundary
without confounding it with learned targets.

## Offline analysis and replay

Scripts prefixed `analyze_` or `replay_` read recordings and do not publish a
robot command. `make_video.py` renders derived visualization. Experiment-scale
runners and their matching analyses live in
[`../experiments/`](../experiments/README.md).

Current MuJoCo flags are in the
[`MuJoCo collaborator guide`](../../docs/runbooks/mujoco.md). Planner/robot
commands are in the
[`planner runbook`](../../docs/runbooks/joystick_planner.md); architecture and
lessons are in the
[`SONIC learning guide`](../../docs/learning/sonic_deployment.md).
