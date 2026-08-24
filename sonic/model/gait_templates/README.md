# Legacy gait templates

These `slow_walk.npz` / `walk.npz` cycles belong to a **withdrawn hybrid-gait
attempt**: replace the planner's lower-body reference with a phase-indexed
mocap loop (`sonic/gait_reference.py`, `sonic/scripts/extract_gait_cycle.py`,
`sim2sim_sonic.py --gait-templates`).

That is not the current slow-walk method. Official Slow Walk stays 0.4. The
operator recipe is encoder-facing sagittal scale (`--slow-gait` /
`--sagittal-scale`) on the planner clip. Evidence:
[`../../../../docs/experiments/mujoco/2026-08-18_slower_gait/README.md`](../../../../docs/experiments/mujoco/2026-08-18_slower_gait/README.md).

Keep the files so the attempt can be reproduced. Do not wire them into
launchers or recommend them for walk / turn / stand.
