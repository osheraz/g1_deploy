# Measured-lab steering

These S50R recipes bind a starting placement, paired model, steering target,
planning rate and Y standing support. Do not substitute flat start joints for
the measured terrain placement. Start images and local assets are checked by:

```bash
python -m steerable_recovery.tools.verify_lab_gallery
python g1_deploy/steerable-recovery/deploy.py --gallery list
```

| Pair | Placement / start image | Comparison | Planner schedule |
|---|---|---|---|
| 1 | `lab00`, lower-front platform/floor interface; `robust_01_minimal.jpg` | Minimal / right floor turn | 5 Hz, no blend |
| 2 | `lab26`, upper-front block/platform interface; `robust_03_minimal.jpg` | Minimal / right floor turn | **10 Hz, 2-tick (40 ms) latent blend** |
| 3 | `lab31`, upper-front block/platform interface; `robust_05_left.jpg` | Left / right side roll | 5 Hz, no blend |
| 4 | `lab01`, lower-front platform/floor interface; `robust_07_left.jpg` | Left / right side roll | 5 Hz, no blend |
| 5 | `lab25`, upper-front block/platform interface; `robust_09_minimal.jpg` | Minimal / right floor turn | 5 Hz, no blend |
| 6 | `deck10`, fully on lower deck, front; `robust_11_left.jpg` | Left / minimal side roll | 5 Hz, no blend |
| 7 | `deck26`, fully on lower deck, middle; `robust_13_minimal.jpg` | Minimal / right side roll | 5 Hz, no blend |
| 8 | `deck42`, fully on lower deck, right; `robust_15_left.jpg` | Left / minimal side roll | 5 Hz, no blend |
| 9 | `flat25`, unobstructed flat floor away from platforms; `robust_17_left.jpg` | Left / minimal side roll | 5 Hz, no blend |
| 10 | `flat43`, unobstructed flat floor away from platforms; `robust_19_left.jpg` | Left / right side roll | 5 Hz, no blend |
| 11 | `lab01`, lower-front platform/floor interface; `robust_21_left.jpg` | Progress-based left / right side roll | 5 Hz, no blend |
| 12 | `lab31`, upper-front block/platform interface; `robust_23_left.jpg` | Progress-based left / right side roll | 5 Hz, no blend |
| 13 | `lab01`, lower-front platform/floor interface; `robust_25_left.jpg` | Seven-second timed left / right side roll | 5 Hz, no blend |
| 14 | `lab31`, upper-front block/platform interface; `robust_27_left.jpg` | Seven-second timed left / right side roll | 5 Hz, no blend |
| 15 | `lab19`, lower-right platform/floor edge; `robust_29_left.jpg` | Recipe 04 progress left / right side roll | 5 Hz, no blend; seeds 54001 / 54002 |

Pair 15 uses recipe 04 progress steering. `robust_32_right` uses the pair 05
placement with its own saved pose/goal. The catalog printed above gives all names.
Pairs 11–12 use recipe 03 progress; pairs 13–14 use seven-second timed steering.
Progress is measured from pelvis orientation, not elapsed wall time.

Inspect a recipe, run its sensor session, then use the motor command when ready:

```bash
python g1_deploy/steerable-recovery/deploy.py --gallery robust_29_left --feedback estimated --print-command
python g1_deploy/steerable-recovery/deploy.py --gallery robust_29_left --feedback estimated --sensor-only
python g1_deploy/steerable-recovery/deploy.py --gallery robust_29_left --feedback estimated
python g1_deploy/steerable-recovery/deploy.py --gallery robust_30_right --feedback estimated
```

These named robot commands enable actuation. A starts recovery and recording;
START optionally moves to saved joints; Y requests standing; A again/SELECT stops.
`estimated` permits experimental odometry bounds while retaining freshness checks.
See [depth setup](deploy.md#terminal-1-depth) and [standing mode](standing-hold.md).

For simulation, replace `deploy.py` with `sim.py`. The same target and Y reference
are used; MuJoCo contacts and sensor timing are not an exact Isaac/hardware replay.
