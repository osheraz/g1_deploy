# Planner joystick workflows

## Layout

| Path | Purpose | Sends physical commands |
| --- | --- | --- |
| [`mujoco.sh`](mujoco.sh) | Generic Python MuJoCo launcher | No |
| [`robot_cpp.sh`](robot_cpp.sh) | Generic C++ physical joystick launcher | Only with `--arm-real-robot` |
| [`robot.sh`](robot.sh) | Generic Python physical preflight/runtime | Only with `--arm-real-robot` |
| [`presets/`](presets/README.md) | Fixed, reviewed stack selections | Depends on preset and arming flag |

The MuJoCo and physical C++ launchers are separate scripts. They share
adaptation names, but neither accepts the other's environment flags.
`mujoco.sh` never publishes a physical `LowCmd`. `robot_cpp.sh` never starts
MuJoCo.

Each launcher selects one atomic model profile (`low-latency`,
`original-release`, or `sonic-v1.1`), one embodiment (`g1_23dof` or
`g1_29dof`), and one adaptation. `official-parity` / `naive` is valid on the
full 3x2x2 matrix. `forward-stable-23` / `stable` is valid for any released
model on `g1_23dof` and additionally accepts `--slow-gait`,
`--sagittal-scale`, and `--hip-pitch-scale`. Native
29-DoF rejects those 23-DoF adaptations. `robust-small-gait-23` and
`stable-slow-23` are withdrawn legacy attempts: blocked on the physical
robot, and blocked in MuJoCo unless `--legacy-reproduction` is passed.

## Unified MuJoCo launcher

```bash
/home/clhsieh/e_skin/g1_deploy/sonic/scripts/run_planner_joystick_mujoco.sh --model-profile original-release --embodiment g1_23dof --adaptation official-parity
/home/clhsieh/e_skin/g1_deploy/sonic/scripts/run_planner_joystick_mujoco.sh --model-profile original-release --embodiment g1_23dof --adaptation forward-stable-23
/home/clhsieh/e_skin/g1_deploy/sonic/scripts/run_planner_joystick_mujoco.sh --model-profile original-release --embodiment g1_23dof --adaptation forward-stable-23 --slow-gait
```

## Unified physical C++ launcher

```bash
/home/clhsieh/e_skin/g1_deploy/sonic/scripts/run_planner_joystick_robot_cpp.sh --adaptation official-parity --interface enp132s0
/home/clhsieh/e_skin/g1_deploy/sonic/scripts/run_planner_joystick_robot_cpp.sh --adaptation forward-stable-23 --interface enp132s0
/home/clhsieh/e_skin/g1_deploy/sonic/scripts/run_planner_joystick_robot_cpp.sh --adaptation forward-stable-23 --slow-gait --interface enp132s0
```

Add `--arm-real-robot` only after the matching no-DDS preflight. C++ is the
current physical recommendation because the Python path experienced 112--115 ms
LowState stalls on the robot.

## Current corrected-forward 23-DoF preset

Interactive keyboard MuJoCo:

```bash
/home/clhsieh/e_skin/g1_deploy/sonic/workflows/joystick/presets/forward_stable_23_mujoco.sh
```

Physical C++ controller (moves the robot):

```bash
/home/clhsieh/e_skin/g1_deploy/sonic/workflows/joystick/presets/forward_stable_23_robot_cpp.sh --arm-real-robot --slow-gait --interface enp132s0
```

Run the same physical command without `--arm-real-robot` for its no-DDS
preflight. The preset locks Original Release, `g1_23dof`, and
`forward-stable-23`. Append `--slow-gait` for the healthy-plant encoder-facing
recipe (sagittal 0.28 + knee +8° + split yaw). C++ is the current physical
recommendation because the Python path experienced 112--115 ms LowState stalls
on the robot.

## Withdrawn robust 23-DoF preset

`robust-small-gait-23` is a **legacy attempt** from a backward-selected
campaign. Physical presets refuse to start. The MuJoCo preset starts only
with `--legacy-reproduction`. Do not use it for a new walk / turn / stand
run. Current recipe: `forward-stable-23 --slow-gait`.

Never run Python and C++ low-level controllers concurrently.

## Adaptation status

| Adaptation | Status |
| --- | --- |
| `official-parity` | General default for the complete model x operation x embodiment matrix |
| `forward-stable-23` | Current 23-DoF forward Slow Walk candidate on any released model; all-six masks, optional `--slow-gait` / sagittal / hip-pitch scale, direct output |
| `robust-small-gait-23` | **Withdrawn legacy attempt.** Blocked on the robot; MuJoCo needs `--legacy-reproduction` |
| `stable-slow-23` | **Withdrawn legacy attempt.** Blocked on the robot; MuJoCo needs `--legacy-reproduction` |

See the [model guide](../../model/README.md) for policy components.
Compatibility commands under `g1_deploy/sonic/scripts/run_planner_joystick_*.sh`,
`run_forward_stable_23_*.sh`, `run_robot_23_cpp.sh`, and
`run_robust_small_gait_23_*.sh` delegate here without changing their operator
interface.
