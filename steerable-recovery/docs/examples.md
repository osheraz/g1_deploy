# Saved starting poses

Use the same model, rate and initial joint targets to compare random sampling.
A new `--seed random` selects a new recorded seed. Fresh noise is drawn each plan.
Different seeds permit variation; they do not guarantee distinct recoveries.

| Selection | Saved start | Notes |
| --- | --- | --- |
| `--preset prone` | `config/multimodal/flat_three` prone | Flat face-down start |
| `--preset supine` | `config/multimodal/flat_three` supine | Earlier supine6819 joints |
| `--route 1`, `2`, `3` | `config/steering/trajectories/supine_151*` | Shared face-up steering start |
| Explicit `--start-pose` | Any robot pose JSON | Robot positioning only |
| Explicit `--start-state` | Any simulator state JSON | MuJoCo only |

[Robot commands](deploy.md) · [MuJoCo commands](sim.md) · [Steering](steering.md).
On hardware, START moves to the saved joints; A alone uses measured joints.
Presets do not position the floating base or reproduce simulator contacts.

Advanced saved-example catalogs remain available through:

```bash
.venv/bin/python -m steerable_recovery.presets.multimodal --help
```

They preserve their original selected seeds/models. Prefer the explicit S25/S50R
commands in the guides when running a new comparison. Add `--print-command` to
preset/route commands to inspect the resolved command without opening sensors.
