# MuJoCo

Prepare a bundle and set `HV2_RECOVERY_BUNDLE` using the
[deployment guide](https://github.com/osheraz/hv2/blob/master/docs/guides/deployment.md).
MuJoCo supports interactive inspection of a published pair. Paper evaluation uses
[the native Isaac workflow](https://github.com/osheraz/hv2/blob/master/docs/guides/evaluation.md).

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

Recorded starts preserve their terrain checksum and forbid hidden reset/preroll
changes. The `hv2.recorded_sim_start.v1` format is unchanged. MuJoCo contacts and
synchronous inference differ from native Isaac evaluation and robot deployment.
Depth uses the checkpoint contract and the shared Warp renderer. See
[steering](steering.md) and [standing mode](standing-hold.md) for optional controls.

Configured timed and upright guidance release use the same criteria as hardware.
They request an unguided solve immediately while preserving measured tracker
history. Upright release latches until reset. Progress recipes that disable the
upright cutoff retain their scheduled unguided solve. Y remains a separate
encoded-standing intervention that bypasses planning. The simulation summary
records the release reason, time and active upright thresholds.
