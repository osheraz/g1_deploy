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

The viewer shows 29 joints, pelvis IMU orientation and uncropped 640×480 depth
with the server's filters. Robot position is fixed for display; it is not odometry.
RGB is blank because the publisher sends depth only. No policy or motor publisher runs.
Stop both terminals with Ctrl+C.

## Depth contract

Hardware calibration lives in [`rise/config/models.yaml`](../../rise/config/models.yaml),
under `camera.base_intrinsics`. Shared simulation defaults live in
[`depth_config.py`](https://github.com/osheraz/hv2/blob/master/hv2_common/sensors/depth_config.py).
The intrinsics describe full-resolution images; the shared sampler derives the
cropped 64×48 input. Preserve each checkpoint's calibration. Regenerate cached
depth when changing capture geometry.

[`native_depth_server.py`](../../rise/sensing/native_depth_server.py) publishes
30 Hz uint16 depth with timestamps, intrinsics, scale, sequence and processing metadata.
PNG and raw transport preserve the same pixels. Raw avoids compression but needs
gigabit. Choose image processing separately:

| Processing | Effect |
| --- | --- |
| `raw` | No filtering; selected explicitly by the RISE launch command |
| `vigor` | Smoothing while preserving source missing/near/far categories |
| `vigor_legacy` | Smoothing and hole filling; the server's standalone default |

Match the selected model's processing contract. Raw and hole-filled images are
different model inputs even when their shapes match.

The viewer shows the full streamed image in metres. The planner receives the
calibrated 64×48 depth/mask and causal history, not that display image.

Receive time differs from capture time. Deployment aligns camera clocks through
port 5558, forwarded as 15558. The viewer does not need this second port.

## LiDAR preview

With no deployment/Livox driver running, this owns a read-only preview driver:

```bash
cd /home/osher/hv2
.venv/bin/python g1_deploy/rise/tools/sensors/check_lidar.py --live --viser
```

Open the printed Viser URL and forward its port in VS Code. Add
`--depth-server 127.0.0.1 --depth-port 15556` to also show the existing depth stream.
Ctrl+C closes the preview and its driver. Run it separately from policy trials.
Preview uses 50 Hz packets; deployment recording uses 30 Hz. These points are
neither a registered world map nor validated pelvis XYZ. For motion evaluation use
[the deployment replay](../rise/recording.md).
