# Native recovery deployment interfaces

The operator workflow is [Steerable Recovery](../README.md).
`steerable_recovery.hardware` owns robot transports, device workers and hardware
session orchestration. The parent `motion_diffusion.inference` package owns
checkpoint-bound model loading, causal measured histories, asynchronous S→Z
sampling, PD/action conversion and portable sensor contracts.

Use the [canonical handoff](../../../docs/guides/deployment.md) for selected
pairs and training/evaluation commands. `scripts/run_native_recovery.sh check`
verifies a portable bundle without connecting sensors or motors. Always specify
the model; old launcher aliases are historical models, not the selected pair.

Keep fresh replanning noise, exact tracker/planner depth histories and saved
normalizers. Neural acceleration is optional and receipt-bound. Analytic planner
guidance requires PyTorch gradients; it cannot silently use TensorRT/ONNX.
Sensor clocks, missing depth, feedback loss and stale plans remain explicit.
Experimental measured-pelvis steering is not independently qualified dynamic
accuracy. ROS estimator diagnostics live in `steerable_recovery.hardware.dlio_simulation`;
portable simulation measurements live in `humanoidverse2.observations.pelvis`.
Neither supplies a live hardware fallback estimate.

Historical operator guides and previous source are recoverable through
[storage and history](../../../docs/guides/storage.md).
