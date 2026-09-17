# Same start, different recoveries

For the current **three prone and three supine examples on flat ground**, use
[the September 15 options](../config/multimodal/flat_three/README.md):

```bash
.venv/bin/python g1_deploy/recovery_options.py --start prone
.venv/bin/python g1_deploy/recovery_options.py --start supine
```

N cycles the three selected samples. That guide also contains the matching
robot presets and commands. The earlier selections below are retained as
historical experiments; their limiter settings and starting poses differ.

For **large spatial path differences**, start with the
[verified supine/prone spatial examples](../config/multimodal/r8_s25_spatial/README.md).
Their camera stays fixed; add `--random` to explore unselected seeds.

For the **two additional prone/supine starts with fresh random planner seeds**,
use [these live MuJoCo commands](../config/multimodal/r8_s25_additional/README.md).
The commands below retain the earlier selected-seed demonstrations.

Run one of these from `/home/osher/hv2` on a graphical desktop:

```bash
.venv/bin/python g1_deploy/multimodal_recovery.py --start supine
.venv/bin/python g1_deploy/multimodal_recovery.py --start sideways
.venv/bin/python g1_deploy/multimodal_recovery.py --start prone
```

Each opens live MuJoCo with the actual depth input displayed. No video recording
is enabled. **N** cycles the two selected diffusion seeds; **R** repeats the current
seed; **Space** pauses. Close the window to exit. There is no automatic time limit.
Run the next command after closing the previous window.

| Start | Selected seeds | Observed contrast |
| --- | --- | --- |
| Supine | 42, 44 | Strongest contrast: different roll/rise direction; about 170° heading separation |
| Sideways (left side) | 43, 45 | Different turning/body paths; about 95° heading separation |
| Prone | 50, 51 | Similar push-up rise, different turn and leg placement; about 58° heading separation |

These are selected successful **12-second simulated examples**, not success-rate
estimates. Prone provides weaker body-path diversity. Longer live continuations
have not been evaluated. The poses have not been qualified as hardware presets.

The model is **R8_S25 diffusion 40k + frozen R8+6k depth tracker**, TensorRT FP32,
DDIM8, fresh noise, 0.5-second horizon and 2.5 Hz replanning. Existing bounded PD
commands remain enabled. No steering or reference trajectory is supplied.

The starts come from teacher-realized training trajectories with contrasting
futures. They were transferred to flat MuJoCo, passively settled once and saved.
Every reset restores the same joint positions, full base pose, zero velocities,
camera delivery seed and initial model histories. Only the diffusion noise changes.
This differs from ordinary `sim_diffusion.py`, where **N** also changes the pose.
The default window follows the pelvis; use the floor grid to judge translation.

Explore additional noise from the same start:

```bash
.venv/bin/python g1_deploy/multimodal_recovery.py --start supine --seeds 42 43 44 45 46 47
```

Other seeds need not recover successfully or produce a new mode. For a finite
headless sweep, add `--headless`; it runs each seed for 12 seconds and saves
trajectories. Add `--video` only if you want MP4 files. `--output PATH` sets a new
output directory; the default is under `logs/multimodal_recovery/`.

The small start files and selection receipts are in
[`config/multimodal/r8_s25`](../config/multimodal/r8_s25). Showing them needs the
existing R8_S25 model bundle, not the teacher corpus. The launcher checks the model
and sampling recipe; the snapshot checks the MuJoCo plant and version. If those
change, regenerate the search rather than assuming the selected seeds still work.

To repeat the teacher search (requires the release on shared storage):

```bash
.venv/bin/python g1_deploy/search_multimodal_recovery.py \
  --output logs/multimodal_recovery/new_search --mine --candidates-per-posture 8
.venv/bin/python g1_deploy/search_multimodal_recovery.py \
  --output logs/multimodal_recovery/new_search --search --seeds 42 43 44 45 --seconds 12
```

`--ids ID ...` restricts a search to particular candidates. Additional seed sweeps
in the same directory are combined without counting repeated seeds as new modes.
After inspecting the trajectories, `--export PATH` packages the best available
pair per posture; `--ids` can restrict which reviewed poses are exported.

[Search evidence and implementation](../../notes/validation/multimodal_recovery_20260914.md).
