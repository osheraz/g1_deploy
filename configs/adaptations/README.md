# SONIC adaptation profiles

The ordinary embodiment configs implement `official-parity`; it is not a
separate YAML in this directory. Files here are explicit, opt-in research
profiles with narrower contracts.

| Profile | Allowed stack | Status | What changes |
| --- | --- | --- | --- |
| `forward-stable-23` | Any released model + planner + G1 23-DoF | Current corrected-forward MuJoCo and C++ selection | Shared YAML `g1_23dof_forward_stable.yaml`; masks all six absent reference/history axes; optional `--slow-gait` / sagittal / hip-pitch encoder-facing recipe; direct output |
| `robust-small-gait-23` | Each released checkpoint + planner + G1 23-DoF | **Withdrawn legacy attempt.** Blocked on the robot. MuJoCo needs `--legacy-reproduction` | Backward-selected wrist-mask / hip-only scale from a wrong-sign campaign |
| `stable-slow-23` | Original Release + planner + G1 23-DoF | **Withdrawn legacy attempt.** Blocked on the robot. MuJoCo needs `--legacy-reproduction` | All-six masks plus weak ankle output shaping |

`robust-small-gait-23` is implemented by one registry-selected YAML per model:

| Checkpoint | Config | Wrist history | Hip scale |
| --- | --- | --- | ---: |
| Low-Latency | [`low_latency_23dof_robust_small_gait.yaml`](low_latency_23dof_robust_small_gait.yaml) | native | 0.80 |
| Original Release | [`original_release_23dof_robust_small_gait.yaml`](original_release_23dof_robust_small_gait.yaml) | masked | 0.80 |
| SONIC v1.1 | [`sonic_v1_1_23dof_robust_small_gait.yaml`](sonic_v1_1_23dof_robust_small_gait.yaml) | masked | 0.90 |

All three mask only the four absent wrist future-reference channels. They do
not change planner recurrence, decoder output, action scale, PD gains, SMPL
pose reconstruction, final IDLE centering, or captured standing hold. The hip
reference scale is active only for planner gait; wrist masking is a tracker
observation contract and remains applicable in POSE mode.

Python/MuJoCo and C++/robot both accept `--slow-gait`, `--sagittal-scale`,
and `--hip-pitch-scale` on `forward-stable-23`. Those three flags are
mutually exclusive. `--slow-gait` is the healthy-plant Original Release
recipe (sagittal 0.28 + knee +8° + walk yaw 0.15 / in-place 0.35) from the
[`2026-08-18 slower-gait study`](../../../docs/experiments/mujoco/2026-08-18_slower_gait/README.md).
Bare `--sagittal-scale` is 0.85, the only scale that survives +10° knee
bias. `--landing-hold` / `--landing-hold-dvel` are a **legacy landing-hold
attempt**. Pose freeze destroys swing; the 160 ms dvel zero is
gait-neutral and does not beat `--slow-gait`. Reproduction only. Python
only; not on the C++ robot path.

`sim2sim_sonic.py --gait-templates` / `policy/gait_reference.py` is a
**legacy hybrid-gait attempt** (mocap lower-body cycles over the planner
upper body). Do not use it for walk / turn / stand. Reproduction only.

`forward-stable-23` is the current 23-DoF observation contract. Pair it with
`--model-profile`; do not treat the YAML as a model lock.
`original_release_23dof_forward_stable.yaml` is a compatibility alias for the
shared file. It leaves all outputs unscaled. The unified MuJoCo and C++
physical launchers accept `--slow-gait` / `--sagittal-scale` /
`--hip-pitch-scale`. Omission keeps encoder-facing amplitude at 1.00.
Collaborator MuJoCo flags are in
[`../../../docs/runbooks/mujoco.md`](../../../docs/runbooks/mujoco.md). The
matching physical command is
`g1_deploy/scripts/run_planner_joystick_robot_cpp.sh`. It is not yet a
physical-robot qualification.
`robust-small-gait-23` and `stable-slow-23` are withdrawn legacy attempts.
They are blocked on the physical robot. MuJoCo reproduction needs
`--legacy-reproduction`. Do not make either the default for the model x
operation x embodiment matrix.

The rationale and evidence are in
[`../../../docs/learning/sonic_deployment.md`](../../../docs/learning/sonic_deployment.md)
and the
[`cross-model adaptation report`](../../../docs/experiments/mujoco/2026-08-16_cross_model_23dof_adaptation/README.md).
Read the superseding Original Release forward result in the
[`2026-08-18 correction`](../../../docs/experiments/mujoco/2026-08-18_forward_direction_and_contact_correction/README.md).
