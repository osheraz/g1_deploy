# Reviewed joystick presets

These launchers lock a complete experimental stack so model, embodiment, and
adaptation cannot drift between commands. Generic MuJoCo selection remains in
[`mujoco.sh`](../mujoco.sh). Generic physical C++ selection remains in
[`robot_cpp.sh`](../robot_cpp.sh). The Python physical path remains in
[`robot.sh`](../robot.sh). The MuJoCo and robot launchers stay separate.

| Preset | Runtime | Environment | Physical commands |
| --- | --- | --- | --- |
| `forward_stable_23_mujoco.sh` | Python | MuJoCo | No |
| `forward_stable_23_robot_cpp.sh` | C++ | Unitree G1 | Only with `--arm-real-robot` |
| `robust_small_gait_23_mujoco.sh` | Python | MuJoCo | No |
| `robust_small_gait_23_robot_cpp.sh` | C++ | Unitree G1 | Only with `--arm-real-robot` |
| `robust_small_gait_23_robot_python.sh` | Python | Unitree G1 | Only with `--arm-real-robot` |

The current MuJoCo and C++ physical presets lock Original Release, `g1_23dof`,
and `forward-stable-23`: all six unavailable reference/history axes are masked,
with no ankle/output shaping. Both accept optional `--hip-pitch-scale`; omitting
it keeps the base bilateral hip-pitch reference scale at 1.00. Prefer the
unified [`../robot_cpp.sh`](../robot_cpp.sh) launcher when comparing
`official-parity` against `forward-stable-23`. The three `robust_small_gait`
presets are historical backward-selected profiles. The Python physical preset
is retained for controlled runtime A/B and diagnostics; C++ is the current
physical recommendation.

Compatibility entry points in `g1_deploy/scripts/` delegate here so existing
absolute commands continue to work.
