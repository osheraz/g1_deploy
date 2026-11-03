# Models and saved starts

| Model names | Use |
| --- | --- |
| `ours`, `joint`, `depth_blind`, `blind_blind` | Published pairs; prepare their matching bundle |
| `s25r_diffusion`, `s50r_diffusion` | Earlier S25/S50 experiments |
| `r8_s25_diffusion`, `r8_s50r_diffusion` | Earlier R8 experiments |
| `r2_f50r_flow` | Flat prone comparison |

Names select exact tracker/planner pairs, not aliases to newer weights.
[Bundle preparation](https://github.com/osheraz/hv2/blob/proposal/reorganization-20261031/docs/guides/deployment.md).

```bash
python g1_deploy/rise/sim.py --preset supine --model s25r_diffusion --seed 42 --print-command
python g1_deploy/rise/deploy.py --gallery list
python g1_deploy/rise/deploy.py --gallery robust_29_left --feedback estimated --print-command
```

`--preset prone|supine` uses saved flat joint configurations. `--gallery` uses
the [measured lab recipes](lab-steering.md). Inspect the resolved command first:
named deploy presets enable motor output, while `--print-command` executes nothing
and gallery `--sensor-only` removes actuation.

For a separate checkout, `HV2_RECOVERY_ASSET_ROOT` selects the workspace containing
saved inputs. New recordings still go to the running checkout's `logs/real_world`.

```bash
python -m rise.tools.verify_lab_gallery
```
