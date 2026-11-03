# Timing measurements

Measure planner sampling, tracker latency and asynchronous scheduling separately.
A lower replanning frequency changes policy behavior as well as compute load;
use the saved recipe for reproduction.

Prepare the selected pair with the [deployment guide](../../../docs/guides/deployment.md)
and keep its `HV2_RECOVERY_BUNDLE` setting.

```bash
python -m steerable_recovery.tools.benchmarks.inference \
  --bundle "$HV2_RECOVERY_BUNDLE" --model ours \
  --profiles pytorch --iterations 30 --warmup 10 --concurrent-seconds 10 \
  --output "$NEW_BENCHMARK"
```

Each published pair requires its matching prepared bundle. TensorRT profiles require
matching validated exports in `config/models.yaml`; no legacy export is selected
automatically. Synthetic timing excludes sensors, network transport and DDS.
Use a recorded sensor-only session to measure the complete input pipeline.
