# Retained deployment consumers

The published catalog (`ours`, `joint`, `depth_blind`, `blind_blind`) and Osher's
retained deployment records have different checkpoint identities. Select the
original name and bundle; no old name redirects to a published pair.

| Consumer | Status | Maintained interface |
| --- | --- | --- |
| VIGOR and SONIC | Retained execution | Existing `deploy_vigor.py` and `sonic/` workflows |
| S25R/P42, S50R/P42 | Retained execution and recorded replay | `--model s25r_diffusion`, `--model s50r_diffusion` |
| R8 S25, R8 S50R | Retained execution and same-start comparison | `--model r8_s25_diffusion`, `--model r8_s50r_diffusion` |
| R2 F50R | Retained baseline for the flat prone preset | `--model r2_f50r_flow` |
| 32 measured-lab runs | Retained execution and replay inputs | `--gallery list`, [lab commands](lab-steering.md) |
| Flat prone/supine starts | Retained same-start comparison | `--preset prone`, `--preset supine` |
| R8 S25 spatial starts | Retained reset/replay inputs | Explicit `--start-state config/multimodal/r8_s25_spatial/…json` |
| Historical robot and paper recordings | Retained replay | `replay.py RUN_NAME` or an absolute session directory |
| Stanley flat/terrain galleries, three route presets and producer import/search workflows | Retired by incoming revision | Git history; not restored as current entry points |

The retained profiles keep checkpoint paths, gain/depth contracts, cadence and
export bindings. The original lab goals keep progress release, timed release,
upright-cutoff exceptions, and Y settings. Runtime inference and transitions
remain owned by HV2. Historical results do not qualify this integrated revision
on hardware.

From the HV2 checkout, inspect commands without launching a viewer or robot:

```bash
python g1_deploy/steerable-recovery/deploy.py --gallery robust_29_left --feedback estimated --print-command
python g1_deploy/steerable-recovery/sim.py --gallery robust_30_right --headless --print-command
python g1_deploy/steerable-recovery/sim.py --preset supine --model s25r_diffusion --seed 42 --print-command
python g1_deploy/steerable-recovery/replay.py --help
```

Named **deploy** presets retain their existing motor-enabled command expansion.
`--print-command` executes nothing; gallery `--sensor-only` removes actuation.
The direct deploy runtime requires `--actuate`. Simulation rejects actuation.

For an isolated checkout, set `HV2_RECOVERY_ASSET_ROOT` to a workspace containing
the retained `logs/native_recovery/` inputs. This affects bundle, export and
recorded-terrain inputs only. New sessions remain under the running checkout's
`logs/`; existing recordings can be addressed by absolute path.

```bash
HV2_RECOVERY_ASSET_ROOT=/path/to/asset/workspace \
  python -m steerable_recovery.tools.verify_lab_gallery
```

The verifier reads input bytes and prints a report. Optional `--output FILE`
creates a new report without rewriting an asset or an existing report.

| Previous command/input | Current command/input |
| --- | --- |
| `deploy.py --gallery NAME` / `sim.py --gallery NAME` | Same named lab consumer |
| `sim.py --gallery-start FILE` | `sim.py --recorded-start FILE`; named launcher updated |
| `tools.import_lab_gallery --verify-local` | `tools.verify_lab_gallery` |
| `--route 1\|2\|3` | Retired; explicit supported goals use `--goal-json FILE` |
| `lidar/mount_candidate_20260912.json` | `lidar/mount_candidate.json` |

Python 3.11 shares HV2's NumPy 1.x profile and uses OpenCV 4.11. Python 3.12
shares the NumPy 2.x profile. Install compatible HV2 and g1 dependencies together;
dependency resolution does not establish Torch, simulator or TensorRT binary
qualification. TensorRT remains an optional machine-specific dependency.
