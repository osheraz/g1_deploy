# Code map

Three applications share robot I/O and recording primitives. Training and learned
RISE inference belong to HV2; this repository connects them to devices and viewers.

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

1. `deploy.py`: arguments → model/site configuration → recorder → runtime.
2. `hardware/runtime.py`: startup → active control → stop and cleanup.
   `startup.py` handles preview/readiness; `unitree.py` handles robot state,
   remote buttons and motor publication.
3. `sensing/network_depth.py` and `odometry_receiver.py`: timestamped measurements
   and freshness. `camera.py` pairs depth with robot observations; `measured_pose.py`
   converts DLIO/camera-body geometry to pelvis pose.
4. `recording/session.py`: child-process ownership and the A recording trigger.
   `depth.py` and `telemetry.py` write/read their respective streams.
5. `replay.py`: the single operator entrypoint for reviewing recordings.
   Media tools are optional exports; they are not imported by the control loop.

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

**RISE's `sim.py` is an application, not another physics engine or planner.**
It creates the HV2 MuJoCo plant and calls the same recovery policy, tracker,
guidance and Y standing implementation as deployment. Simulation advances on its
own clock; robot deployment uses a wall-clock planner worker and DDS. Sharing
model code does not make simulated contacts or sensor truth equal to hardware.

SONIC has its own model/reference interface. Its calibration plant measures
joint mappings and actuator responses; it does not implement RISE inference.
Keep that experiment separate from a recovery simulator.

## Checks and boundaries that remain

- Model dimensions, joint order, normalization, sampling and PD bindings must
  agree. Check them when loading a model, not repeatedly inside each motor tick.
- Depth/odometry age, clock uncertainty, finite actions and startup limits change
  while running. They remain live checks. Recording shape/index checks protect
  playback from incomplete writes.
- VIGOR depth normalization and RISE metric depth/masks are different trained
  inputs. They share transport and physical sensor definitions, not an assumed
  identical neural input. SONIC's NumPy rotations use shared WXYZ helpers without
  importing Torch; model-specific observation layout remains with SONIC.
- Requested actions and applied commands are different values. Simulator truth
  and measured pelvis estimates are different sources. Preserve both in logs.

The Jetson server needs only `rise/sensing/native_depth_server.py` and
`core/sensing/depth_wire.py`, copied together. ROS workers use system Python and
bootstrap the checkout. Importing a model does not start ROS, cameras or DDS.

There are no old-path forwarding modules or SDK-path patches. Use the original
Git revision for historical execution; the current entrypoints use current paths.

## Where to add something

- New checkpoint or experiment: model config, saved goal/start, or preset.
- New planner/tracker behavior: its HV2 library; both sim and robot consume it.
- New device: `core/sensing` for reusable transport; `rise/sensing` for recovery
  measurement pairing. New robot control: `core/robot` and its application adapter.
- New recording field: the owning stream writer/reader. Use `core/review` for
  shared display primitives and the application replay command for composition.
- New operator command: its application's tools/scripts directory and existing
  topic guide. Avoid another launcher chain or a second collection of notes.

Changes to inference/history/control require matching old/new model outputs and
simulation trajectories, not only import tests. Sensor and recording changes
also require startup, stop, timestamp and export tests. Hardware qualification
is a separate run on the robot.
