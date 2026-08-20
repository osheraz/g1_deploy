# SONIC adaptation profiles

The ordinary embodiment configs implement `official-parity`; it is not a
separate YAML in this directory. Files here are explicit, opt-in research
profiles with narrower contracts.

| Profile | Allowed stack | Status | What changes |
| --- | --- | --- | --- |
| `forward-stable-23` | Any released model + planner + G1 23-DoF | Current corrected-forward MuJoCo selection | Shared YAML `g1_23dof_forward_stable.yaml`; masks all six absent reference/history axes; base scale 1.00 or optional bilateral hip-pitch scale 0.80--1.00; direct output |
| `robust-small-gait-23` | Each released checkpoint + planner + G1 23-DoF | Historical backward-selected profiles; forward recommendation withdrawn | Masks absent wrist reference axes, preserves waist task context, uses checkpoint-specific wrist history and bilateral hip-reference scale; direct output |
| `stable-slow-23` | Original Release + planner + G1 23-DoF | Historical research profile | Masks all absent axes plus weak ankle action shaping; corrected forward tests attribute its useful behavior to the masks, not ankle shaping |

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

The MuJoCo launcher also accepts `--sagittal-scale` on `forward-stable-23`.
That scales hip pitch, knee, and ankle pitch together. Bare flag 0.85 is the
current smaller-gait default from the
[`2026-08-18 small-gait study`](../../../docs/experiments/mujoco/2026-08-18_small_consistent_gait/README.md).
On a known-healthy plant, `--sagittal-scale 0.33` is the slowest uniform walk;
`--slow-gait` is the slower 0.28 + knee-offset + split-yaw recipe from the
[`2026-08-18 slower-gait study`](../../../docs/experiments/mujoco/2026-08-18_slower_gait/README.md)
(Original Release; the other two checkpoints do not inherit that floor).
Those flags are mutually exclusive with `--hip-pitch-scale`.
`--landing-hold` / `--landing-hold-dvel` are a **legacy landing-hold
attempt**. Pose freeze destroys swing; the 160 ms dvel zero is
gait-neutral and does not beat `--slow-gait`. Reproduction only.

`sim2sim_sonic.py --gait-templates` / `policy/gait_reference.py` is a
**legacy hybrid-gait attempt** (mocap lower-body cycles over the planner
upper body). Do not use it for walk / turn / stand. Reproduction only.

`forward-stable-23` is the current 23-DoF observation contract. Pair it with
`--model-profile`; do not treat the YAML as a model lock.
`original_release_23dof_forward_stable.yaml` is a compatibility alias for the
shared file. It leaves all outputs unscaled. The unified MuJoCo and C++
physical launchers accept `--hip-pitch-scale`; no value selects 0.85, chosen
by a 75-run signed knee-bias sweep, while omission keeps the base reference
scale at 1.00. Collaborator MuJoCo flags are in
[`../../../docs/runbooks/mujoco.md`](../../../docs/runbooks/mujoco.md). The
matching physical command is
`g1_deploy/scripts/run_planner_joystick_robot_cpp.sh`. It is not yet a
physical-robot qualification.
`stable-slow-23` remains only for reproduction of the 2026-08-13 study. Do not
make either historical profile the default for the general model x operation x
embodiment matrix.

The rationale and evidence are in
[`../../../docs/learning/sonic_deployment.md`](../../../docs/learning/sonic_deployment.md)
and the
[`cross-model adaptation report`](../../../docs/experiments/mujoco/2026-08-16_cross_model_23dof_adaptation/README.md).
Read the superseding Original Release forward result in the
[`2026-08-18 correction`](../../../docs/experiments/mujoco/2026-08-18_forward_direction_and_contact_correction/README.md).
