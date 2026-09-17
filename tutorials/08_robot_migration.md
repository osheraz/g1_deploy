# Move onboard streamers from G1-23 to G1-29

The robot runs sensor publishers. Bluefin runs inference and DDS. The native
planner/action decoder is 29-DoF: do not test motor output on the old G1-23.

## Saved from the old robot

Verified snapshot on Bluefin:
`/home/osher/hv2/logs/robot_migration/g1_23_20260911_rtt9Eu/`.
Its README and `receipt.json` list files, dependencies and SHA-256 checks.
The snapshot is local/ignored, not part of the Git push.

| Reusable | Do not transfer as defaults |
|---|---|
| Camera publisher source | Camera serial, intrinsics, mount calibration |
| Tactile Python driver and install instructions | Board-to-limb USB serial assignments |
| Message protocols and workstation clients | 23-DoF motor map, encoder offsets, PD/firmware settings |
| Dependency/version inventory | Whole old ARM venv, credentials, vendor startup services |

The snapshot's RGB-D server is the old **filtered, untimestamped** protocol.
For native planner depth use `core/common/native_depth_server.py` from this
repository. Its raw image is processed by the shared sim/hardware builder.
Both publishers open the same camera; never start them simultaneously.
Tactile is optional: the current native depth planner does not consume it.

## Before copying to the new robot

The new G1-29 currently has IPv6-only Ethernet on its Jetson. Use the
[verified Bluefin SSH command](01_network.md#new-g1-29-ssh-over-ipv6-verified-2026-09-11).
Do not assume the old `g1` alias pointing to `192.168.123.164` reaches this robot.

1. Disconnect the old robot from this network before connecting another robot
   with the same IP. Confirm the new onboard address and SSH identity; a host-key
   mismatch must be verified, not bypassed with `StrictHostKeyChecking=no`.
2. Read OS/architecture, Python and installed camera dependencies. Reuse source,
   not the old binary environment. Check the actual RealSense serial/USB link.
3. Use the camera clock-reply endpoint and startup robot-tick alignment described
   in the camera guide. Do not set either machine's wall clock as a deployment
   step. Absolute host dates need not match; the measured mapping must remain
   stable while streaming.
4. Verify physical 29-DoF mapping and reported mode independently. The old
   LowState reported mode_machine=4; do not promote that value to the new robot.

No robot files or services were changed during the backup. No motor commands
were sent. The old robot currently exposes no RealSense device or tactile ports;
streaming cannot be validated until the actual sensors are connected.

## Stage source on the new robot

Completed on 2026-09-11 over IPv6:

- Native depth source: `/home/unitree/osher/native_recovery/native_depth_server.py`.
- Optional tactile source: `/home/unitree/osher/e_skin/jq_tactile_driver/`.
- Isolated Python 3.8 environment: `/home/unitree/osher/native_recovery/.venv`.
  It reuses system NumPy 1.17.4 and OpenCV 4.2.0; installed ARM wheels are
  pyrealsense2 2.55.1.6486, pyzmq 25.1.2 and pyserial 3.5. Vendor Python/Conda,
  startup services, firmware and network configuration were not changed.
- All 14 copied source/package files matched local SHA-256 hashes. Camera CLI
  and tactile server imports pass; that does not establish working physical sensors.

Local package cache: `logs/robot_migration/g1_29_20260911_lbkr9q/` in hv2.
The new Jetson also booted with its UTC clock in 1970 and NTP unsynchronized.
Its wall clock was set once during sensor bring-up, with streaming stopped.
No RTC, timezone or NTP configuration was changed. Do not repeat that as a
streaming setup step. Deployment now aligns camera host time through its clock
reply port and estimates robot tick alignment from startup DDS packets.

After connecting the camera, the Jetson detected **D435I serial 247122073927**,
firmware 5.13.0.55, with a USB 2.1 connection. A depth-only check at 640x480
captured 90 measured frames after 15 warmup frames: 30.03 Hz host arrival rate,
no frame-number gaps, and 96.1% nonzero depth pixels in that scene. Capture was
stopped cleanly afterward; no motor channel was opened. This short USB capture
check is not yet a network-stream, clock-alignment or inference qualification.
Measured stream intrinsics were fx=fy=388.811767578125, cx=326.99853515625,
cy=238.08714294433594, with zero distortion coefficients and approximately
0.001 metres per raw depth unit. The operator confirmed using the mounting position and
rotation already in code; no photo is required. This confirmation does not make
the new camera's measured intrinsics identical to the old checkpoint calibration.

## View robot state and depth

Use [the camera/viewer commands](04_cameras.md). The read-only check received
LowState and raw 640×480 depth at 30 Hz with all 29 joints correctly mapped.
It does not qualify planner inference or motor output.

Only after the identity above is confirmed and the `g1` SSH alias points to the
new robot, from `/home/osher/hv2`:

```bash
ssh g1 'mkdir -p /home/unitree/osher/native_recovery'
rsync -a g1_deploy/steerable-recovery/steerable_recovery/sensing/native_depth_server.py \
  g1:/home/unitree/osher/native_recovery/
```

No autostart unit or robot service replacement is needed. The camera-only
launch, once its serial is known, is:

```bash
source /home/unitree/osher/native_recovery/.venv/bin/activate
python /home/unitree/osher/native_recovery/native_depth_server.py --serial ACTUAL_CAMERA_SERIAL
```

This needs NumPy, OpenCV, pyzmq and a RealSense binding built for that robot's
Python/OS. The old runtime versions are recorded in the receipt; do not blindly
install its mixed system/Python librealsense versions on another machine.

If tactile is needed, copy only the recovered `jq_tactile_driver/` package to
`/home/unitree/osher/e_skin/jq_tactile_driver/`, then:

```bash
source /home/unitree/osher/native_recovery/.venv/bin/activate
cd /home/unitree/osher/e_skin
python -m jq_tactile_driver.server --role left_hand right_hand
```

That command requires the correct `/dev/jq_hand_*` assignments. The existing
bare command from the login directory does not import the package on the old
robot: it is source-copied, not installed. Do not run the old USB-rule script
unless those exact physical boards are being reused and the assignments match.

Use the [diffusion deployment guide](../steerable-recovery/docs/deploy.md) for the normal YAML settings, automatic timestamp alignment,
sensor-only inference and separate actuation commissioning. No old 23-DoF
calibration or old VIGOR2 planner configuration belongs in that native profile.
