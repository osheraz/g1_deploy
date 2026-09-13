# G1 deployment

**[Run diffusion / flow on the robot](tutorials/deploy_diffusion.md)** — depth,
policy + recording, buttons, evaluation and Viser. Start here on bluefin.

| Task | Entry point |
| --- | --- |
| Test recovery in MuJoCo | [Simulation guide](tutorials/sim_diffusion.md) |
| Check what is validated / still missing | [Deployment status](tutorials/deployment_status.md) |
| Set up a new machine or robot | [Setup](tutorials/GETTING_STARTED.md) |
| Preview depth / LiDAR without policy | [Sensor viewers](tutorials/04_cameras.md) |
| Understand recorded files / replay older runs | [Recording reference](RECORDING.md) |
| Change models or exports | [Configuration](tutorials/03_config.md), [runtime reference](tutorials/deployment_reference.md) |
| Benchmark inference and concurrent sensors | [Inference](tutorials/inference_latency.md), [full deployment compute](../notes/deployment_compute.md) |
| Integrate measured steering in simulation | [Stanley's contract](../notes/stanley_steering_measurement_contract.md) |

The shared diffusion runtime is in the parent hv2 repository at
`motion_diffusion/behavioral/hardware/`. Use hv2 with its pinned `g1_deploy`
submodule; this is not an independent copy of the model runtime.

SONIC teleoperation has [separate workflows](sonic/workflows/README.md).
`deploy_vigor.py` is legacy and moves during startup;
[deprecated](deprecated/README.md) is not the diffusion deployment path.
