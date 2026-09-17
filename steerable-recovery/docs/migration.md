# September 17 deployment organization

Update hv2 together with its pinned `g1_deploy` submodule. Do not merge this
cleanup into Stanley's model branch just to run it; use the matching deployment
checkout and leave model/training ownership unchanged.

```bash
# In a clean deployment checkout following master:
git pull --ff-only
git submodule update --init g1_deploy
.venv/bin/python -m pip install --no-deps --no-build-isolation -e ./g1_deploy
```

This task creates local commits; publication is a separate step. Git does not
transfer model weights, engines, recordings or the external ROS workspace.

| Previous command/location | Replacement |
| --- | --- |
| `g1_deploy/deploy_diffusion.py` | `g1_deploy/steerable-recovery/deploy.py` |
| `g1_deploy/sim_diffusion.py` | `g1_deploy/steerable-recovery/sim.py` |
| `g1_deploy/replay_motion.py` | `g1_deploy/steerable-recovery/replay.py` |
| `recovery_options.py --robot --start prone` | `deploy.py --preset prone` |
| `recovery_options.py --start supine` | `sim.py --preset supine` |
| `steering_options.py --robot --option 3` | `deploy.py --route 3` |
| `steering_options.py --option 3` | `sim.py --route 3` |
| `replay_model.py` | `python -m steerable_recovery.review.model` |
| `record_motion.py` | `steerable_recovery.recording.session` (automatic through deploy) |
| `configs/diffusion.yaml` | `steerable-recovery/config/models.yaml` |
| `config/` recovery assets | `steerable-recovery/config/` |
| `sonic/common` episode/camera/annotations | `core/recording/` |
| `sonic/common/unitree_mode.py` | `core/robot/mode.py` |
| `sonic/scripts/replay_episode.py` | `core/review/viewer.py` (shared recovery/SONIC viewer) |

The [complete path map](../migration_paths.json) also covers diagnostics and
four ROS capture/replay workers moved from hv2 into the submodule. Their original
sources are retained in hv2 checkpoint **9dd44b8e**. Deployment checkpoint:
**b88a9b6**. Pure relocation: deployment **9333f3b**, parent **e2f4d8b7**.
The pure-move commits are intermediate history; use the completed integration.

Python imports use `steerable_recovery` (underscore); the project folder uses
`steerable-recovery` (hyphen). No old root-script forwarding shims are installed.
Callers that import files by path must update; paper showcase tools are maintained
separately. Figure-only overlays moved to `tools/media/figures.py` and load lazily.

Old recordings remain readable. Model replay maps renamed sources but still
checks their bytes, source archive, weights and engines. Edited source requires
an explicit `--allow-code-changes`; the report lists differences.

The robot-side camera is still a two-file standalone upload; copy both files
using [the deployment guide](deploy.md#terminal-1-depth). This cleanup did not
contact the robot or install camera files there.

No model pair, seed, replanning rate, history reset, PD parameter, limiter window,
recording root or stop behavior was intentionally changed. Simulation start-offset,
joint-hold-during-settling and batch-error handling are separate requested work,
not bundled into this organization.

Local check artifacts: `logs/deployment_validation/reorganization_20260917/`.

## Verification

- 77 resolved preset commands matched the checkpoint after translating moved paths.
- 19 public/module/worker `--help` checks passed, including system-Python ROS workers.
- Existing recovery/shared-tool suite: 526 passed, 2 skipped; 11 new migration tests passed.
- The final recorder/review/worker subset passed 66 tests after import cleanup.
- One SONIC profile test fails on its missing external registry
  `hv2/config/sonic_deployment_profiles.json`. The pre-move source gives the same error.
- A September 16 robot recording loaded 497 camera frames, 495 full-resolution
  depth captures and control telemetry. Three tracker and three planner calls
  matched the saved trace with explicit code-change comparison; all 842 control
  and 41 planner records passed trace-byte validation.
- Current guide links and moved files resolve. Editable install and wheel build passed.

No robot processes, camera server, ROS driver or live viewer were started.
