# Timing checks

Deployment timing is saved with each run. Replay writes a timing report under
`motion_replay/`; control telemetry contains capture, receipt, inference and
publish timestamps. Capture-to-publish excludes robot receipt and motor response.

The planner runs separately from the 50 Hz tracker. Depth reception, image saving,
external capture and DLIO also run asynchronously. They still compete for host
CPU, network and disk. Measure with the selected model and actual sensor load.

Sensor-only combined check (keep the depth terminal running; no motor commands):

```bash
.venv/bin/python g1_deploy/steerable-recovery/deploy.py \
  --model r8_s25_diffusion --backend tensorrt --duration-s 20 \
  --live-odometry --record-sensors --record-depth --record-model --external-camera
```

Guided synthetic worker check (no sensors):

```bash
.venv/bin/python -m steerable_recovery.presets.steering --option 3 --model r8_s50r_diffusion --benchmark
```

Legacy/P11 backend comparison tool:

```bash
.venv/bin/python -m steerable_recovery.tools.benchmarks.inference --model p11_diffusion --profiles pytorch trt_graph --iterations 30 --warmup 10 --concurrent-seconds 10
```

That benchmark currently accepts P11/P05/legacy aliases, not the R8 candidates.
Its compute timings exclude sensors. Do not quote them as R8 end-to-end latency.
No sampler steps, precision, cadence or buffering policy changed in this cleanup.

For a new GPU, rebuild and verify exports through the shared tool; select the
bundle/model explicitly and write a new directory before updating `models.yaml`:

```bash
.venv/bin/python -m motion_diffusion.behavioral.hardware.export --help
```

Keep `export.json`, ONNX, validation samples and matching TensorRT engines together.
