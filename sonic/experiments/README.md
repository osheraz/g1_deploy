# Controlled experiment runners

This directory contains repeatable multi-run studies. Each runner writes raw
recordings under `/data/clhsieh/logs/e_skin/experiments/`; its reviewed
protocol, derived tables/figures, limitations, and conclusions live under
`docs/experiments/`.

| Runner / analysis | Reviewed report |
| --- | --- |
| `run_cross_model_23dof_adaptation.py` / `analyze_cross_model_23dof_adaptation.py` | [`2026-08-16 cross-model adaptation`](../../docs/experiments/mujoco/2026-08-16_cross_model_23dof_adaptation/README.md) |
| `run_robust_23dof_adaptation.py` | [`2026-08-15 robust adaptation`](../../docs/experiments/mujoco/2026-08-15_robust_23dof_adaptation/README.md) |
| `run_knee_offset_sweep.py` / `analyze_knee_offset_diagnosis.py` | [`2026-08-15 post-adjustment diagnosis`](../../docs/experiments/physical/2026-08-15_post_left_knee_offset_diagnosis/README.md) |
| `run_slow_walk_training_free.py` / `analyze_slow_walk_training_free.py` | [`2026-08-13 training-free Slow Walk`](../../docs/experiments/mujoco/2026-08-13_training_free_23dof_slow_walk/README.md) |
| `run_small_consistent_gait.py` / `analyze_small_consistent_gait.py` | [`2026-08-18 smaller consistent gait`](../../docs/experiments/mujoco/2026-08-18_small_consistent_gait/README.md) |
| `run_slower_gait.py` / `analyze_landing_hold.py` | [`2026-08-19 landing hold`](../../docs/experiments/mujoco/2026-08-19_landing_hold/README.md) (legacy attempt; withdrawn) |

Keep experiment orchestration here and reusable controller math in
`g1_deploy/policy/`. Do not copy a production launcher into an experiment;
call the canonical workflow/engine with a recorded configuration instead.
