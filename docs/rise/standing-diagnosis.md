# Diagnose a recording

Inspect measured motion and external video first:

```bash
python g1_deploy/rise/replay.py RUN_NAME --speed 0.25
python -m rise.review.model logs/real_world/RUN_NAME --max-calls 5
python -m rise.tools.diagnostics.analyze_run --help
python -m rise.tools.diagnostics.standing_cause --help
```

Model replay compares saved observations/noise with inference outputs. It does
not replay contact dynamics. Input-swap diagnostics identify sensitivity of the
network; establishing a physical cause requires a controlled simulation or trial.
Keep derived diagnostics separate from the original recording.
