# Camera and read-only robot viewer

The saved G1-29 configuration uses a D435I, serial `247122073927`, on its Jetson.
Match the camera mounting, intrinsics and clock alignment to the selected model.

## Start depth

Use [terminal 1 in the operator guide](../rise/deploy.md#terminal-1-depth).
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
[`rise/config/models.yaml`](../../rise/config/models.yaml), under
`camera.base_intrinsics`. New simulated planner depth uses
[`DEFAULT_DEPTH_CAPTURE`](../../../humanoidverse2/simulator/sensors/depth_capture.py);
the environment default is also updated in
[`legged_base.yaml`](../../../humanoidverse2/config/env/legged_base.yaml).
These are full-resolution intrinsics, before cropping/resizing. Use the shared
builder to derive the processed rays; do not paste these values as 64×48 intrinsics.
Existing checkpoints keep their saved calibration. For new training, regenerate
any cached depth with the new capture settings; changing metadata alone does
not change old rendered images.

[`native_depth_server.py`](../../rise/sensing/native_depth_server.py) publishes
`[JSON metadata, lossless uint16 pixels]` with 30 Hz capture. PNG is the default
wire encoding; `--encoding raw` removes camera-side compression on gigabit.
The viewer accepts both. See [server update and launch](../rise/deploy.md#terminal-1-depth).
Metadata carries capture time,
clock domain, sequence, serial, intrinsics, depth scale and processing mode.
The server default `--processing vigor_legacy` applies VIGOR smoothing and
hole filling; the RISE launch command explicitly selects `--processing raw`. Filled positive
pixels become valid depth for the viewer and policy. Explicit `--processing vigor`
applies the same filters, then restores pixels whose missing/near/far classification
changed; original holes therefore remain. `--processing raw` bypasses filtering.
No mode applies color alignment or the old unused 3 m threshold.

The hole-filled default differs from the unified-depth-v3 contract,
which specifies raw hardware processing and unknown pixels remaining invalid.
Processing provenance remains in every stream/recording; do not label the two
input contracts interchangeable or infer training compatibility from image shape.

The viewer shows streamed depth in metres. It is **not** the planner's processed
image. Current inference uses the shared calibrated crop/ray sampler,
64×48 range/mask and causal history; see [deployment](../rise/deploy.md).
The old square-crop VIGOR preprocessing is not used by the current planner.

A receive timestamp is not camera capture time. Deployment automatically uses
the streamer's clock-reply endpoint (port 5558, forwarded as 15558) to align
camera host time; the viewer itself does not require that second port.

## LiDAR preview

With no deployment/Livox driver running, this owns a read-only preview driver:

```bash
cd /home/osher/hv2
.venv/bin/python g1_deploy/rise/tools/sensors/check_lidar.py --live --viser
```

Open the printed Viser URL and forward its port in VS Code. Add
`--depth-server 127.0.0.1 --depth-port 15556` to also show the existing depth stream.
Ctrl+C closes the preview and its owned driver. Do this separately from the policy
trial; it is not required for deployment. The live preview defaults to 50 Hz
packets; the deployment recorder uses 30 Hz. Preview points are not a registered
world map or qualified pelvis XYZ. For motion evaluation use
[the deployment replay](../rise/recording.md).
