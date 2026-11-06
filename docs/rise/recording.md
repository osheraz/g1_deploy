# Recording and replay

The [deployment commands](deploy.md) support recording. Sensors warm up before A;
**accepted policy A starts the episode**. Only bounded context is buffered first
(0.5 s depth, 5 s raw motion), plus small setup/status receipts.
A again / SELECT / Ctrl+C stops the policy; wait for sensor and model-trace saves.

| Location | Contents |
| --- | --- |
| `logs/real_world/RUN/` | Real trials: config, telemetry, stop/save status |
| `RUN/sensors/` | Raw LiDAR, IMUs, joints, live XYZ, original-resolution depth, external camera |
| `RUN/model_trace/` | Model inputs, sampled noise, outputs, issued actions, configuration, Git revision and weight identities |
| `RUN/review/` | Derived videos, frames, recovery marks and review checks |
| `logs/deployment_validation/` | Sensor-only / offline checks |
| `logs/aborted_deployments/` | Archived launches with no policy episode |
| `paper/real_world/` | Only recordings the user explicitly moves there |

Depth is saved once at source resolution after the selected server processing;
model histories reference captures and reconstruct the checkpoint's crop/mask.
It is not stored as a stack of repeated resized images.

## Viser

```bash
.venv/bin/python g1_deploy/rise/replay.py RUN_NAME
.venv/bin/python g1_deploy/rise/replay.py RUN_NAME --speed 0.25 --loop
.venv/bin/python g1_deploy/rise/replay.py logs/real_world
```

A bare name resolves under `logs/real_world`; full paths and `sensors/` paths
also work. Forward port **8081** in VS Code and open **http://localhost:8081**.
Playback, external camera and depth controls are at the top. XYZ uses recorded
live odometry when available. Orange / missing sections retain estimator validity
information; display smoothing does not establish position accuracy.

**Play** and **Loop** default on. **Speed (×)** changes synchronized robot/camera
playback; `0.25` is quarter speed. Uncheck a control to pause or stop looping;
`--no-loop` disables looping at launch. Playback does not modify recordings.

Mark **start** and **standing / end**, then **Save & next recording**.
Marks are saved in `RUN/review/recovery_interval.json`.
Browse mode skips hidden folders and archives no-policy launches.

## Exports after stopping

```bash
.venv/bin/python g1_deploy/rise/replay.py RUN_NAME --prepare-only --export-video --export-depth
```

Writes `review/external.mp4` and synchronized `review/depth_aligned.mp4`.
`--export-frames` adds source JPEGs and depth PNGs. Existing exports are reused.
Normal replay does not encode video or rerun DLIO. Use `--offline-xyz` explicitly
for an older raw recording that needs odometry processing.

Optional figure exports: `--export-overlay` / `--export-strip`. These load
`tools/media/figures.py` only on request; normal replay does not load Mask R-CNN.
Picked images remain under `logs/real_world_hightlight/` (historical spelling).

## Replay model decisions

```bash
.venv/bin/python -m rise.review.model logs/real_world/RUN_NAME --max-calls 5
python -m rise.tools.diagnostics.analyze_run --help
python -m rise.tools.diagnostics.standing_cause --help
```

Uses saved observations/noise and verified model weights/engines; run after
stopping deployment because it uses the GPU. This checks inference and issued
commands, not physical dynamics. New recordings save Git revisions and modified-tree
status; they do not copy the source tree. Exact reproduction requires the recorded
checkout. Add `--allow-code-changes` for a comparison against current code;
differences are listed in `review/model_replay.json`. Input-swap diagnostics probe
network sensitivity; they do not establish a physical cause or replay contacts.
Keep derived results separate from the original recording.
