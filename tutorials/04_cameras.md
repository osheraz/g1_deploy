# Camera and read-only robot viewer

The current G1-29 has a D435I, serial `247122073927`, connected to its Jetson.
Use the mounting position and rotation already in code, as confirmed by the
operator. Lens intrinsics and timestamp alignment are separate checks.

## Terminal 1: camera publisher and tunnel

Run on Bluefin. This starts only the camera publisher on the Jetson. Leave it
running, and do not open the same camera with another server.

```bash
ssh -t -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:15556:127.0.0.1:5556 \
  -L 127.0.0.1:15558:127.0.0.1:5558 \
  unitree@fe80::d150:4901:5a1a:cb5%enp132s0 \
  '/home/unitree/osher/native_recovery/.venv/bin/python /home/unitree/osher/native_recovery/native_depth_server.py --serial 247122073927 --bind tcp://127.0.0.1:5556'
```

Omit `-6` on this forwarding command: the literal remote address selects IPv6,
while the local forward listens on IPv4 loopback. The old `g1` SSH alias may
still point at the disconnected robot; use [the verified address](01_network.md#new-g1-29-ssh-over-ipv6-verified-2026-09-11).

## Terminal 2: robot and depth in Viser

```bash
cd /home/osher/hv2
source .venv/bin/activate
python g1_deploy/sonic/check_streams.py \
  --config g1_29dof_sonic.yaml --net enp132s0 --viser --no-tactile \
  --depth-server 127.0.0.1 --depth-port 15556 \
  --urdf humanoidverse2/data/robots/g1/g1_29dof_sonic.urdf
```

Open **http://localhost:8080**. Over remote VS Code, forward port 8080.

This displays all 29 joints, pelvis IMU orientation and uncropped 640×480 raw
depth. Position is anchored in the viewer, not measured odometry. RGB is blank
because this publisher sends depth only. No policy or motor publisher is created.
Stop both terminals with Ctrl+C.

## Depth contract

Current lens at **640×480**: `[fx, fy, cx, cy] =
[388.811767578125, 388.811767578125, 326.99853515625, 238.08714294433594]`.
The physical camera is configured in
[`configs/diffusion.yaml`](../configs/diffusion.yaml), under
`camera.base_intrinsics`. New simulated planner depth uses
[`DEFAULT_DEPTH_CAPTURE`](../../motion_diffusion/terrain/planner_depth.py);
the environment default is also updated in
[`legged_base.yaml`](../../humanoidverse2/config/env/legged_base.yaml).
These are full-resolution intrinsics, before cropping/resizing. Use the shared
builder to derive the processed rays; do not paste these values as 64×48 intrinsics.
Existing checkpoints keep their saved calibration. For new training, regenerate
any cached depth with the new capture settings; changing metadata alone does
not change old rendered images.

[`native_depth_server.py`](../core/common/native_depth_server.py) publishes
`[JSON metadata, lossless uint16 PNG]` at 30 Hz. Metadata carries capture time,
clock domain, sequence, serial, intrinsics and depth scale. It does not apply
temporal smoothing, hole filling or color alignment.

The viewer shows raw depth in metres. It is **not** the planner's processed
image. Current inference uses the shared calibrated crop/ray sampler,
64×48 range/mask and causal history; see [deployment](deploy_diffusion.md).
The old square-crop VIGOR preprocessing is not used by the current planner.

A receive timestamp is not camera capture time. Deployment automatically uses
the streamer's clock-reply endpoint (port 5558, forwarded as 15558) to align
camera host time; the viewer itself does not require that second port.
