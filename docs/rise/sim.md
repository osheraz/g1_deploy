# MuJoCo

Prepare a bundle and set `HV2_RECOVERY_BUNDLE` using the
[deployment guide](https://github.com/osheraz/hv2/blob/master/docs/guides/evaluation/deployment.md).
MuJoCo supports interactive inspection of a published pair. Paper evaluation uses
[the native Isaac workflow](https://github.com/osheraz/hv2/blob/master/docs/guides/evaluation/README.md).

```bash
python g1_deploy/rise/sim.py --bundle "$HV2_RECOVERY_BUNDLE" --model ours --backend pytorch \
  --terrain flat --start supine --seed 42 --seconds 10 --no-depth-view
```

Add `--viewer` for a window or `--video` for an MP4. Supply a new `--output` directory
for generated files. `--bundle PATH` selects another verified bundle;
its manifest must contain the requested model. The saved inference recipe is used
unless an override is explicit.

| Control | Effect |
| --- | --- |
| R / N | Repeat / change noise; `--noise-only` fixes the start |
| M | Sample another terrain, except for a recorded or measured scene |
| `--terrain stairs --terrain-seed 43` | Select a cached terrain tile |
| `--replan-hz 5` | Replan at 5 Hz with the 50 Hz tracker |
| `--start-state PATH.json` | Restore a flat resting state |
| `--recorded-start PATH.json` | Restore recorded pose, velocities and exact terrain |
| `--terrain-geometry PATH.json` | Load measured static-box geometry |
| `--goal-json PATH.json` | Apply an explicit steering command |

Recorded starts keep their terrain checksum and reject reset/preroll changes.
The format remains `hv2.recorded_sim_start.v1`. MuJoCo contacts and synchronous
inference differ from Isaac evaluation and hardware. Depth uses the model's
saved settings and shared Warp renderer. See
[steering](steering.md) and [standing mode](standing-hold.md) for optional controls.

Timed and upright guidance release use the hardware criteria. Both request an
unguided plan immediately and preserve tracker history. Upright release stays
active until reset. Progress recipes without an upright cutoff use their scheduled release.

Y bypasses planning and selects the encoded standing reference. The simulation
summary records the guidance-release reason, time and upright thresholds.
