# Code map

RISE, SONIC and VIGOR share robot I/O and recording tools. HV2 owns training and
RISE model inference. This repository connects models to devices and viewers.

```text
g1_deploy/
├── rise/                  RISE recovery
│   ├── deploy.py          robot command, configuration and lifecycle setup
│   ├── sim.py             MuJoCo deployment preview
│   ├── replay.py          recording preparation, exports and viewer launch
│   ├── models.py          model selection; bundle prepare/check
│   ├── paths.py           checkout, configuration and output roots
│   ├── hardware/          runtime loop, startup preview, Unitree adapter
│   ├── sensing/           paired depth, clocks, DLIO and measured pelvis pose
│   ├── recording/         capture processes, depth frames and telemetry
│   ├── review/model.py    offline model replay and input interventions
│   ├── presets/           flat and lab recipe argument expansion
│   ├── config/            model pairs, saved poses/goals, camera/LiDAR settings
│   ├── setup/             pinned ROS/LiDAR workspace build
│   └── tools/             sensors/, diagnostics/, benchmarks/, media/
├── sonic/                 SONIC tracking and teleoperation
│   ├── deploy.py, sim.py   robot and simulation entrypoints
│   ├── sonic_runner.py    encoder/decoder execution
│   ├── sonic_reference.py, pico_reference.py
│   ├── model/             observation layout, adapters, history and backends
│   ├── inputs/, sensing/  joystick/PICO, robot and tactile input
│   ├── calibration/       joint mapping and calibration protocols
│   ├── scripts/           input/, motion/, calibration/, analysis/, replay/
│   ├── assets/            release/, low_latency/, motions/
│   └── tests/
├── vigor/                 visual policy, depth preprocessing, recording/replay
├── core/
│   ├── config.py          VIGOR/SONIC robot configuration
│   ├── robot/             DDS, commands, buttons, IMU, mode handover and CRC
│   ├── sensing/           camera/LiDAR transport and depth wire encoding
│   ├── recording/         camera capture/indexing, episodes and annotations
│   └── review/            shared Viser display and playback
├── configs/               robot joint maps, gains and SONIC model profiles
├── zero_torque.py         explicit motor command
├── tests/                 deployment integration checks
└── docs/                  this map, setup and application guides
```

## Read RISE in this order

1. `deploy.py` reads arguments and connects configuration, recording and runtime.
2. `hardware/runtime.py` runs startup, control and shutdown. `startup.py` checks
   readiness; `unitree.py` handles state, remote buttons and motor commands.
3. `sensing/` receives depth and odometry. `camera.py` pairs depth with robot state;
   `measured_pose.py` converts DLIO pose to pelvis pose.
4. `recording/session.py` manages recording processes and the A trigger.
   `depth.py` and `telemetry.py` read and write their streams.
5. `replay.py` opens recordings. Optional media exports stay outside the control loop.

## What is shared with HV2?

| Operation | Owner |
| --- | --- |
| Planner, guidance, progress and plan queues | `motion_diffusion/planner/` |
| Model-pair loading and executable recovery policies | `motion_diffusion/inference/` |
| Tracker encoder/decoder and standing-reference encoding | `humanoidverse2/agents/action_vae/` and `motion/references/` |
| Observation histories, state/depth assembly | `humanoidverse2/observations/` |
| Joint/PD contract | `humanoidverse2/robots/g1/` |
| MuJoCo physics, raw sensors, terrain and scene cache | `humanoidverse2/simulator/mujoco/` |
| Camera geometry, corruption, rotations and terrain math | `hv2_common/` |
| DDS/remote, hardware capture and recording processes | this repository |

RISE's `sim.py` uses HV2's MuJoCo physics and the same policy, tracker, steering
and Y mode as deployment. Simulation advances on its own clock; hardware uses a
planner worker and DDS in real time. Contacts and sensor measurements still differ.

SONIC has its own model/reference interface. Its calibration simulator measures
joint mappings and actuator response, separately from recovery simulation.

## Checks and boundaries that remain

- **At model load:** check dimensions, joint order, normalization, sampling and PD settings.
- **During control:** check sensor age, clock uncertainty, finite actions and startup limits.
- **During recording/replay:** check array shapes and indices for incomplete writes.
- **Model inputs:** reuse transport and preserve the checkpoint's depth mode and preprocessing.
- **SONIC rotations:** use shared NumPy WXYZ helpers without importing Torch.
  Keep SONIC's observation layout with its model.
- **Logs:** distinguish requested/applied actions and simulator truth/measured pelvis estimates.

The Jetson server needs only `rise/sensing/native_depth_server.py` and
`core/sensing/depth_wire.py`, copied together. ROS workers use system Python and
bootstrap the checkout. Importing a model does not start ROS, cameras or DDS.

Use the original Git revision to run historical commands. Current commands use
current paths, without forwarding modules or SDK-path patches.

## Where to add something

- **Checkpoint or experiment:** add model settings or a saved goal/start.
  Reuse deployment and simulation commands.
- **Planner/tracker behavior:** change its HV2 library so simulation and hardware share it.
- **Device:** add transport in `core/sensing`, recovery pairing in `rise/sensing`,
  or robot control in `core/robot` and its application adapter.
- **Recording field:** extend its stream writer/reader. Reuse `core/review` for display.
- **Operator command:** use the application's tools/scripts directory and existing guide.

For inference or control changes, compare model outputs and simulation trajectories
before and after. For sensors and recording, test startup, shutdown, timestamps
and exports. Hardware validation is a separate robot trial.
