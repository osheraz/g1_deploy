# Models and saved starts

| Model names | Use |
| --- | --- |
| `ours`, `joint`, `depth_blind`, `blind_blind` | Published pairs; prepare their matching bundle |
| `s25r_diffusion`, `s50r_diffusion` | Earlier S25/S50 experiments |
| `r8_s25_diffusion`, `r8_s50r_diffusion` | Earlier R8 experiments |
| `r2_f50r_flow` | Flat prone comparison |

Names select exact tracker/planner pairs, not aliases to newer weights.
[Bundle preparation](https://github.com/osheraz/hv2/blob/master/docs/guides/evaluation/deployment.md).

Retained visual models declare `depth_input: metric_state`; their saved metric
depth, mask and capture-state inputs stay unchanged. New image-only pairs use
`depth_input: images` and normalized depth. Loading checks the selected mode.

```bash
python g1_deploy/rise/sim.py --preset supine --model s25r_diffusion --seed 42 --print-command
python g1_deploy/rise/deploy.py --gallery list
python g1_deploy/rise/deploy.py --gallery robust_29_left --feedback estimated --print-command
```

`--preset prone|supine` selects saved flat joint configurations.
`--gallery` selects a [measured lab recipe](lab-steering.md).

Named deploy presets enable motor output. Inspect them first with `--print-command`,
which runs nothing. For gallery recipes, `--sensor-only` disables actuation.

For a separate checkout, `HV2_RECOVERY_ASSET_ROOT` selects the workspace containing
saved inputs. New recordings still go to the running checkout's `logs/real_world`.

```bash
python -m rise.tools.verify_lab_gallery
```
