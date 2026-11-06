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

Hardware calibration lives in [`rise/config/models.yaml`](../../rise/config/models.yaml),
under `camera.base_intrinsics`. Shared simulation defaults live in
[`depth_config.py`](https://github.com/osheraz/hv2/blob/master/hv2_common/sensors/depth_config.py).
The intrinsics describe full-resolution images; the shared sampler derives the
cropped 64×48 input. Preserve each checkpoint's calibration. Regenerate cached
depth when changing capture geometry.

[`native_depth_server.py`](../../rise/sensing/native_depth_server.py) publishes
timestamped uint16 depth at 30 Hz, with intrinsics, scale, sequence and processing
metadata. PNG and raw wire encoding preserve the same pixels; raw avoids
compression but needs gigabit. Encoding and image processing are separate choices.

| Processing | Effect |
| --- | --- |
| `raw` | No filtering; selected explicitly by the RISE launch command |
| `vigor` | Smoothing while preserving source missing/near/far categories |
| `vigor_legacy` | Smoothing and hole filling; the server's standalone default |

Match the selected model's processing contract. Raw and hole-filled images are
different model inputs even when their shapes match.

The viewer shows the full streamed image in metres. The planner receives the
calibrated 64×48 depth/mask and causal history, not that display image.

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
