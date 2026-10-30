# Timing measurements

Measure planner sampling, tracker latency and asynchronous scheduling separately.
A lower replanning frequency changes policy behavior as well as compute load;
use the saved recipe for reproduction.

```bash
python -m steerable_recovery.tools.benchmarks.inference --model ours \
  --profiles pytorch --iterations 30 --warmup 10 --concurrent-seconds 10 \
  --output "$NEW_BENCHMARK"
```

For another published pair, pass its `--bundle`. TensorRT profiles require
matching validated exports in `config/models.yaml`; no legacy export is selected
automatically. Synthetic timing excludes sensors, network transport and DDS.
Use a recorded sensor-only session to measure the complete input pipeline.
