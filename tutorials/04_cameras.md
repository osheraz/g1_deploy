# Camera and read-only robot viewer

The current G1-29 has a D435I, serial `247122073927`, connected to its Jetson.
Use the mounting position and rotation already in code, as confirmed by the
operator. Lens intrinsics and timestamp alignment are separate checks.

## Start depth

Use [terminal 1 in the operator guide](../steerable-recovery/docs/deploy.md#terminal-1-depth).
Leave it running, then use a second terminal for the viewer below.

## Robot and depth in Viser

```bash
cd /home/osher/hv2
source .venv/bin/activate
python g1_deploy/sonic/check_streams.py \
  --config g1_29dof_sonic.yaml --net enp132s0 --viser --no-tactile \
  --depth-server 127.0.0.1 --depth-port 15556 \
  --urdf humanoidverse2/data/robots/g1/g1_29dof_sonic.urdf
```

Open **http://localhost:8080**. Over remote VS Code, forward port 8080.

This displays all 29 joints, pelvis IMU orientation and uncropped 640×480
streamed depth, including the server's selected filters. Position is anchored
in the viewer, not measured odometry. RGB is blank
because this publisher sends depth only. No policy or motor publisher is created.
Stop both terminals with Ctrl+C.

## Depth contract

Current lens at **640×480**: `[fx, fy, cx, cy] =
[388.811767578125, 388.811767578125, 326.99853515625, 238.08714294433594]`.
The physical camera is configured in
[`steerable-recovery/config/models.yaml`](../steerable-recovery/config/models.yaml), under
`camera.base_intrinsics`. New simulated planner depth uses
[`DEFAULT_DEPTH_CAPTURE`](../../motion_diffusion/terrain/planner_depth.py);
the environment default is also updated in
[`legged_base.yaml`](../../humanoidverse2/config/env/legged_base.yaml).
These are full-resolution intrinsics, before cropping/resizing. Use the shared
builder to derive the processed rays; do not paste these values as 64×48 intrinsics.
Existing checkpoints keep their saved calibration. For new training, regenerate
any cached depth with the new capture settings; changing metadata alone does
not change old rendered images.

[`native_depth_server.py`](../steerable-recovery/steerable_recovery/sensing/native_depth_server.py) publishes
`[JSON metadata, lossless uint16 pixels]` with 30 Hz capture. PNG is the default
wire encoding; `--encoding raw` removes camera-side compression on gigabit.
The viewer accepts both. See [server update and launch](../steerable-recovery/docs/deploy.md#terminal-1-depth).
Metadata carries capture time,
clock domain, sequence, serial, intrinsics, depth scale and processing mode.
The default `--processing vigor` applies the old filter path, then restores any
pixel whose missing/near/far classification changed. This retains unavailable
depth rather than inventing valid terrain for the policy. `--processing raw`
bypasses filtering; `--processing vigor_legacy` reproduces the original hole
filling for comparison. No mode applies color alignment or the old unused 3 m threshold.

The viewer shows streamed depth in metres. It is **not** the planner's processed
image. Current inference uses the shared calibrated crop/ray sampler,
64×48 range/mask and causal history; see [deployment](../steerable-recovery/docs/deploy.md).
The old square-crop VIGOR preprocessing is not used by the current planner.

A receive timestamp is not camera capture time. Deployment automatically uses
the streamer's clock-reply endpoint (port 5558, forwarded as 15558) to align
camera host time; the viewer itself does not require that second port.

## LiDAR preview

With no deployment/Livox driver running, this owns a read-only preview driver:

```bash
cd /home/osher/hv2
.venv/bin/python g1_deploy/steerable-recovery/steerable_recovery/sensing/check_lidar.py --live --viser
```

Open the printed Viser URL and forward its port in VS Code. Add
`--depth-server 127.0.0.1 --depth-port 15556` to also show the existing depth stream.
Ctrl+C closes the preview and its owned driver. Do this separately from the policy
trial; it is not required for deployment. The live preview defaults to 50 Hz
packets; the deployment recorder uses 30 Hz. Preview points are not a registered
world map or qualified pelvis XYZ. For motion evaluation use
[the deployment replay](../steerable-recovery/docs/recording.md).
