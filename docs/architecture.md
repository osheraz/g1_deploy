# Code ownership

| Directory | Responsibility |
| --- | --- |
| `core/robot` | DDS state/commands, remote buttons, IMU transforms and SDK setup |
| `core/sensing` | Shared depth, external-camera and LiDAR transport |
| `core/recording` | Episode files, camera playback, recovery annotations and logging |
| `core/review` | Shared Viser displays and recorded-episode viewer |
| `configs` | Robot I/O, joint mapping, gains and SONIC adaptations |
| `vigor` | VIGOR visual-policy loading and robot execution |
| `sonic/model` | SONIC encoder/decoder, observation layouts and reference adaptations |
| `sonic/inputs`, `sonic/sensing` | Joystick/PICO commands, tactile input and robot feedback |
| `sonic/calibration` | Joint-calibration protocol and MuJoCo plant |
| `sonic/scripts` | Joystick launchers, calibration, motion preparation, analysis and replay |
| `steerable-recovery/config` | Recovery model pairs, camera settings, start poses and goals |
| `steerable_recovery/cli`, `presets` | Recovery commands and named experiment recipes |
| `steerable_recovery/hardware` | Unitree lifecycle, startup, stopping and control loop |
| `steerable_recovery/sensing` | Clock pairing, depth acquisition and pelvis odometry |
| `steerable_recovery/recording`, `review` | Triggered capture, telemetry and replay |
| `steerable_recovery/tools` | Sensor commands, benchmarks, diagnostics and media exports |
| `docs` | Setup and operator instructions |

`steerable_recovery` is the package inside `steerable-recovery/`.
Learned recovery inference lives in the parent `motion_diffusion/inference` and
`humanoidverse2/agents/action_vae`. Robot and MuJoCo commands use those same models,
histories, steering objectives and standing-reference implementation.

The planner, depth receiver, DLIO and recorders run separately from tracking.
Recording starts at accepted A; pre-start capture is bounded. Y requests the
original encoder's standing reference. The startup limiter lasts 0.5 seconds.
See [standing](recovery/standing-hold.md) and [recording](recovery/recording.md).

The Jetson depth server remains standalone: copy `native_depth_server.py` and
`core/sensing/depth_wire.py` together. ROS workers bootstrap the checkout under
system Python; model imports do not start ROS, cameras or DDS.
