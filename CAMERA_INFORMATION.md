# G1 Onboard Camera Information

Collection date: 2026-08-03 (America/Detroit)

## Status and scope

Collection is complete for the camera identity, available SDK stream profiles, factory
intrinsics, cross-sensor extrinsics, active server configuration, and live RGB/depth
transport. Values in the calibration sections were read from the physical camera through
librealsense on G1 PC2.

No motor command, locomotion client, or policy runtime was started during collection.

The RealSense SDK does not provide the transform from the complete camera assembly to the
G1 torso. That mounting transform requires a separately verified robot mechanical
calibration. The simulation `camera_link` must not be treated as a measurement of the
physical installation.

## Required saved calibration record

This is the compact calibration record for the exact RGB and depth profiles selected by the
robot's RGB-D server. These are factory-calibration values read from this physical camera
through its RealSense SDK; they are not generic D435I specifications.

The official RealSense documentation defines every image stream as having its own 2D pixel
coordinate system and its own 3D camera coordinate system. In a stream's image, `(0, 0)` is
the center of the top-left pixel; in its 3D frame, `+x` points right, `+y` points down, and
`+z` points forward. RGB and depth therefore require separate intrinsics and a calibrated
extrinsic transform between their 3D frames. See the official
[RealSense intrinsic/projection guide](https://dev.realsenseai.com/docs/projection-texture-mapping-and-occlusion-with-intel-realsense-depth-cameras/)
and [RealSense extrinsic/projection API guide](https://dev.realsenseai.com/docs/projection-in-realsense-sdk-2-0/).

### Device and selected stream profiles

| Required field | Saved value |
| --- | --- |
| Camera model | Intel RealSense D435I |
| Serial number | `246422070545` |
| Firmware | `5.13.0.55` |
| RGB stream | 640 x 480, BGR8, 30 FPS |
| Depth stream | 640 x 480, Z16, 30 FPS |
| Depth scale | `0.0010000000474974513` meters per Z16 unit (approximately 1 mm/unit) |

The formats above are the camera-to-server SDK formats. On the network, the server encodes
each BGR8 frame as JPEG and each Z16 depth frame as a lossless PNG; that transport encoding
does not change the saved camera calibration.

### Intrinsics for the selected profiles

`fx` and `fy` are focal lengths in pixels. `cx` and `cy` below are the RealSense SDK's `ppx`
and `ppy` principal-point fields. Coefficients are recorded in the SDK's exact
`coeffs[0..4]` order.

| Stream coordinate system | fx | fy | cx (`ppx`) | cy (`ppy`) | Distortion model | `coeffs[0..4]` |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| RGB, 640 x 480 | 607.367310 | 607.252502 | 322.488281 | 245.726822 | Inverse Brown-Conrady | `[0, 0, 0, 0, 0]` |
| Depth, 640 x 480 | 386.658173 | 386.658173 | 324.116455 | 237.453629 | Brown-Conrady | `[0, 0, 0, 0, 0]` |

The two intrinsic matrices are therefore:

```text
K_rgb = [[607.367310,   0.000000, 322.488281],
         [  0.000000, 607.252502, 245.726822],
         [  0.000000,   0.000000,   1.000000]]

K_depth = [[386.658173,   0.000000, 324.116455],
           [  0.000000, 386.658173, 237.453629],
           [  0.000000,   0.000000,   1.000000]]
```

### Depth-to-RGB extrinsics

For a 3D point expressed in the depth optical coordinate system, the saved transform is
defined as:

```text
p_rgb = R_depth_to_rgb * p_depth + t_depth_to_rgb

R_depth_to_rgb =
[[ 0.999969000, -0.000391363, -0.007842550],
 [ 0.000362701,  0.999993000, -0.003655790],
 [ 0.007843920,  0.003652840,  0.999963000]]

t_depth_to_rgb =
[0.0147143574, -0.0000631260, 0.0004082282] meters
```

The same translation is approximately `[14.7143574, -0.0631260, 0.4082282]` mm. The
rotation above is shown as a mathematical row-by-row matrix. The SDK's flat
`rs2_extrinsics.rotation` storage is column-major and must be converted accordingly.

### SDK fields used to save the record

The official API exposes intrinsics on each selected `video_stream_profile`, extrinsics from
one stream profile to another, and scale on the depth sensor. The corresponding Python calls
used for this record are:

```python
active_profile = pipeline.get_active_profile()
device = active_profile.get_device()

model = device.get_info(rs.camera_info.name)
serial = device.get_info(rs.camera_info.serial_number)
firmware = device.get_info(rs.camera_info.firmware_version)

rgb_profile = active_profile.get_stream(rs.stream.color)
depth_profile = active_profile.get_stream(rs.stream.depth)
rgb_video_profile = rgb_profile.as_video_stream_profile()
depth_video_profile = depth_profile.as_video_stream_profile()

# width(), height(), format(), and fps() identify each selected stream profile.
rgb_intrinsics = rgb_video_profile.get_intrinsics()
depth_intrinsics = depth_video_profile.get_intrinsics()
depth_to_rgb = depth_profile.get_extrinsics_to(rgb_profile)
depth_sensor = device.first_depth_sensor()
depth_scale_m_per_unit = depth_sensor.get_depth_scale()
```

`get_intrinsics()` returns `fx`, `fy`, `ppx`, `ppy`, `model`, and five `coeffs` for that
specific stream profile. `get_extrinsics_to()` returns the rotation and translation mapping
the source profile's 3D frame into the target profile's 3D frame. For Z16 depth sample `d`,
metric depth is `z_meters = d * depth_scale_m_per_unit`; a raw value of zero means that no
valid depth was determined.

Because the verified collection run did not align the streams, an RGB pixel `(u, v)` and a
depth pixel with the same numeric `(u, v)` must not be assumed to observe the same ray.
Correct mapping is: deproject the depth pixel with `K_depth` and its metric depth, transform
the resulting 3D point with `R_depth_to_rgb` and `t_depth_to_rgb`, then project it using
`K_rgb` and the RGB distortion model.

### Alignment-mode interpretation

The robot server constructs `rs.align(rs.stream.depth)`. It calls `align.process(frames)`
only when `align_to_depth=True`; otherwise it publishes the native stream frames. Its
configuration lookup is `config.get("align_to_depth", False)`, and the current `__main__`
configuration does not set the key, so the verified run used the default `False` mode.

| `align_to_depth` | Published depth | Published RGB | Calibration consumers must use |
| --- | --- | --- | --- |
| `False` (verified run and default) | Native depth pixel grid, subsequently depth-filtered | Native RGB pixel grid | Use `K_depth` for depth, `K_rgb` for RGB, and `R_depth_to_rgb` / `t_depth_to_rgb` to map between them. Equal pixel indices are not correspondence. |
| `True` | Depth remains in the depth target grid, subsequently depth-filtered | A synthetic RGB image reprojected by librealsense into the depth pixel grid | For the published aligned pair, both arrays use the depth target pixel geometry (`K_depth` at the current 640 x 480 output). Same pixel indices correspond geometrically where the SDK produced a valid mapping. Do not apply the depth-to-RGB extrinsic again to this already aligned RGB output. |

In both modes, retain the native RGB intrinsics, native depth intrinsics, factory
depth-to-RGB extrinsics, and depth scale saved above. Alignment changes the geometry of the
published RGB image; it does not replace or modify the camera's factory calibration. An
aligned image can contain unmapped pixels because the two physical sensors have different
viewpoints, fields of view, and occlusions.

The table above applies to the current 640 x 480 capture and 640 x 480 publish shape. If
`publish_shape` is changed, OpenCV resizes both arrays after optional RealSense alignment;
the original 640 x 480 intrinsic matrix must not be used unchanged for that resized output.
Save or derive intrinsics for the actual published pixel grid and record the resize rule.

## Device identity

| SDK field | Physical-camera value |
| --- | --- |
| Product name | Intel RealSense D435I |
| Serial number | `246422070545` |
| ASIC serial number | `320143060473` |
| Firmware update ID | `320143060473` |
| Product ID | `0B3A` |
| Product line | D400 |
| Firmware version | `5.13.0.55` |
| Advanced mode | Yes |
| Camera locked | Yes |
| Connection type | USB |
| USB type descriptor | `2.1` |
| Debug opcode | `15` |
| Physical device path | `/sys/devices/platform/3610000.xhci/usb1/1-2/1-2.1/1-2.1:1.0/video4linux/video0` |
| USB enumeration | `8086:0b3a Intel Corp.` |
| Depth scale | `0.0010000000474974513` meters per raw unit |

### Firmware recommendation discrepancy

Two installed SDK interfaces report different recommended firmware versions:

| Interface | SDK version | Recommended camera firmware |
| --- | --- | --- |
| `rs-enumerate-devices` | 2.56.5.0 | `5.17.0.10` |
| `pyrealsense2` | 2.55.1.6486 | `5.16.0.1` |

The installed camera firmware, `5.13.0.55`, is older than both recommendations. No firmware
update was attempted. A firmware update should only be selected after checking compatibility
with PC2's Jetson kernel and the image-server deployment.

## Host and network

| Item | Verified value |
| --- | --- |
| Workstation interface | `enp132s0` |
| Workstation robot-subnet address | `192.168.123.222/24` |
| G1 motion controller / PC1 | `192.168.123.161` |
| PC1 MAC | `7e:1d:75:60:f5:89` |
| G1 development computer / PC2 | `192.168.123.164` |
| PC2 MAC | `3c:6d:66:16:64:c3` |
| SSH endpoint | `unitree@192.168.123.164`, alias `g1` |
| PC2 architecture | AArch64 |
| PC2 OS | Ubuntu 20.04 |
| PC2 kernel | `5.10.104-tegra` |
| Ethernet link | 100 Mb/s, full duplex |

Both robot addresses responded with sub-millisecond round-trip times. Read-only Unitree DDS
on `rt/lowstate` was explicitly bound to `enp132s0` and delivered approximately 1,050
messages per second with message ages of 0.00-0.01 seconds. The observed `mode_machine` was
`4`.

## PC2 camera software

| Component | Version or path |
| --- | --- |
| Python | 3.8.10 |
| `rs-enumerate-devices` | librealsense 2.56.5.0 |
| `pyrealsense2` | 2.55.1.6486 |
| OpenCV | 4.2.0 |
| PyZMQ | 27.1.0 |
| `logging_mp` | `/home/unitree/.local/lib/python3.8/site-packages/logging_mp/` |
| RGB-D server | `/home/unitree/osher/image_server/img_server_rgbd.py` |
| Depth-only server | `/home/unitree/osher/image_server/img_server_osher.py` |

## Supported hardware stream profiles

### Depth and infrared stereo module

| Stream | Resolution | Format | Supported FPS |
| --- | ---: | --- | --- |
| Depth | 1280 x 720 | Z16 | 6 |
| Depth | 848 x 480 | Z16 | 10, 8, 6 |
| Depth | 640 x 480 | Z16 | 30, 15, 6 |
| Depth | 640 x 360 | Z16 | 30 |
| Depth | 480 x 270 | Z16 | 60, 30, 15, 6 |
| Depth | 256 x 144 | Z16 | 90 |
| Infrared 1 | 1280 x 720 | Y8 | 6 |
| Infrared 1 | 848 x 480 | Y8 | 10, 8, 6 |
| Infrared 1 | 640 x 480 | Y8 | 30, 15, 6 |
| Infrared 1 | 640 x 360 | Y8 | 30 |
| Infrared 1 | 480 x 270 | Y8 | 60, 30, 15, 6 |
| Infrared 2 | 1280 x 720 | Y8 | 6 |
| Infrared 2 | 848 x 480 | Y8 | 10, 8, 6 |
| Infrared 2 | 640 x 480 | Y8 | 30, 15, 6 |
| Infrared 2 | 640 x 360 | Y8 | 30 |
| Infrared 2 | 480 x 270 | Y8 | 60, 30, 15, 6 |

### RGB camera

Each row supports RGB8, Y8, BGRA8, RGBA8, BGR8, and YUYV.

| Resolution | Supported FPS |
| ---: | --- |
| 1920 x 1080 | 8 |
| 1280 x 720 | 15, 10, 6 |
| 640 x 480 | 30, 15, 6 |
| 424 x 240 | 60, 30, 15, 6 |

### Motion module

| Stream | Format | Supported FPS |
| --- | --- | --- |
| Accelerometer | MOTION_XYZ32F | 400, 200, 100 |
| Gyroscope | MOTION_XYZ32F | 400, 200 |

The SDK reports zero bias/noise variances and an identity 3x3 sensitivity matrix for both
motion streams. These values should not be interpreted as a measured noise calibration.

## Factory intrinsics

Intrinsics depend on stream and resolution. The server uses the bold rows.

### Depth and infrared intrinsics

Depth, Infrared 1, and Infrared 2 share these calibrated values at a given listed profile.
The distortion model is Brown-Conrady and all five reported coefficients are zero.

| Resolution | fx | fy | ppx | ppy | Horizontal FOV | Vertical FOV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 x 144 depth | 644.430298 | 644.430298 | 134.860779 | 67.756042 | 22.47 deg | 12.75 deg |
| 480 x 270 | 241.661362 | 241.661362 | 242.572784 | 133.408508 | 89.60 deg | 58.38 deg |
| 640 x 360 | 322.215149 | 322.215149 | 323.430389 | 177.878021 | 89.60 deg | 58.38 deg |
| **640 x 480** | **386.658173** | **386.658173** | **324.116455** | **237.453629** | **79.22 deg** | **63.65 deg** |
| 848 x 480 | 426.935059 | 426.935059 | 428.545258 | 237.188385 | 89.60 deg | 58.68 deg |
| 1280 x 720 | 644.430298 | 644.430298 | 646.860779 | 355.756042 | 89.60 deg | 58.38 deg |

Depth camera matrix at 640 x 480:

```text
K_depth = [[386.658173,   0.000000, 324.116455],
           [  0.000000, 386.658173, 237.453629],
           [  0.000000,   0.000000,   1.000000]]
```

### Color intrinsics

The distortion model is inverse Brown-Conrady and all five reported coefficients are zero.

| Resolution | fx | fy | ppx | ppy | Horizontal FOV | Vertical FOV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 424 x 240 | 303.683624 | 303.626251 | 213.244156 | 122.863411 | 69.84 deg | 43.13 deg |
| **640 x 480** | **607.367310** | **607.252502** | **322.488281** | **245.726822** | **55.57 deg** | **43.13 deg** |
| 1280 x 720 | 911.050964 | 910.878784 | 643.732422 | 368.590240 | 70.17 deg | 43.13 deg |
| 1920 x 1080 | 1366.576416 | 1366.318115 | 965.598694 | 552.885376 | 70.17 deg | 43.13 deg |

Color camera matrix at 640 x 480:

```text
K_color = [[607.367310,   0.000000, 322.488281],
           [  0.000000, 607.252502, 245.726822],
           [  0.000000,   0.000000,   1.000000]]
```

## Factory cross-sensor extrinsics

Matrices below follow the `rs-enumerate-devices -c` printed convention and translations are
in meters. Python's flat `extrinsics.rotation` array is column-major; reshape it accordingly
rather than treating it as a row-major array.

Depth to color:

```text
R_depth_to_color =
[[ 0.999969000, -0.000391363, -0.007842550],
 [ 0.000362701,  0.999993000, -0.003655790],
 [ 0.007843920,  0.003652840,  0.999963000]]

t_depth_to_color =
[0.0147143574, -0.0000631260, 0.0004082282]
```

Color to depth:

```text
R_color_to_depth =
[[ 0.999969000,  0.000362701,  0.007843920],
 [-0.000391363,  0.999993000,  0.003652840],
 [-0.007842550, -0.003655790,  0.999963000]]

t_color_to_depth =
[-0.0147170834, 0.0000673930, -0.0002930457]
```

Other unique factory transforms:

| Source | Target | Rotation | Translation in meters |
| --- | --- | --- | --- |
| Depth | Infrared 1 | Identity | `[0, 0, 0]` |
| Depth | Infrared 2 | Identity | `[-0.0500081442, 0, 0]` |
| Depth | Gyroscope | Identity | `[-0.0055200001, 0.0051000002, 0.0117400000]` |
| Depth | Accelerometer | Identity | `[-0.0055200001, 0.0051000002, 0.0117400000]` |
| Color | Infrared 1 | `R_color_to_depth` | `[-0.0147170834, 0.0000673930, -0.0002930457]` |
| Color | Infrared 2 | `R_color_to_depth` | `[-0.0647252277, 0.0000673930, -0.0002930457]` |
| Color | Gyroscope | `R_color_to_depth` | `[-0.0202370845, 0.0051673930, 0.0114469547]` |
| Color | Accelerometer | `R_color_to_depth` | `[-0.0202370845, 0.0051673930, 0.0114469547]` |

Reverse transforms are the inverses of the corresponding entries above. The SDK reported
Gyroscope-to-Accelerometer and Accelerometer-to-Gyroscope as identity with zero translation.

## RGB-D server configuration observed during collection

The PC2 server was verified and started successfully for collection, then stopped on
2026-08-03 at the user's request after the measurements were complete.

| Setting | Observed value |
| --- | --- |
| Process during collection | PID 4042, `python3 img_server_rgbd.py` (now stopped) |
| Bind address | `tcp://*:5556` |
| Camera serial | `246422070545` |
| Requested depth stream | Z16, 640 x 480, 30 FPS |
| Requested color stream | BGR8, 640 x 480, 30 FPS |
| Published shape | 640 x 480 |
| Color encoding | JPEG, quality 80 |
| Depth encoding | lossless PNG preserving uint16 values |
| ZMQ message | two parts: `[color_jpg, depth_png]` |
| RGB/depth alignment during collection | Disabled (`align_to_depth=False`) |

The RGB-D server previously defaulted to port 5555, conflicting with the depth-only server
and the repository tutorial. Its default was corrected to 5556. The original PC2 file is
recoverable as:

```text
/home/unitree/osher/image_server/img_server_rgbd.py.bak-before-port-5556
```

Depth processing order on every server frame:

1. Decimation filter with magnitude 1.
2. Depth-to-disparity transform.
3. Spatial filter with SDK defaults.
4. Temporal filter with SDK defaults.
5. Disparity-to-depth transform.
6. Hole-filling filter with SDK defaults.
7. Nearest-neighbor resize to the publish shape (currently unchanged at 640 x 480).

Color is resized with `INTER_AREA` when needed. Because alignment is disabled, a color pixel
and depth pixel at the same `(u, v)` are not the same camera ray. Use both intrinsics plus the
depth-to-color extrinsic for geometric correspondence, or deliberately enable server-side
alignment and account for its changed output geometry.

## Live stream measurements

Measurements are scene- and time-dependent and are included as validation, not calibration.

### Direct SDK capture before server filtering

| Measurement | Observed value |
| --- | --- |
| Requested profile | Depth Z16 + color BGR8, 640 x 480 at 30 FPS |
| Warmed-up host rate | 29.65 FPS |
| Depth shape/type | 480 x 640, uint16 |
| Color shape/type | 480 x 640 x 3, uint8 |
| Nonzero raw depth minimum | 650 units / approximately 0.650 m |
| Nonzero raw depth maximum | 9375 units / approximately 9.375 m |
| Center-pixel depth | approximately 1.551 m |
| Invalid/zero depth fraction | 0.079609375 |

### ZMQ wire transport, 30-message sample

| Measurement | Observed value |
| --- | --- |
| Receive rate | 28.92 FPS |
| Parts per message | 2 |
| JPEG color payload, min / mean / max | 26,574 / 26,663.6 / 26,861 bytes |
| Decoded color | 480 x 640 x 3, uint8 |
| PNG depth payload, min / mean / max | 133,784 / 137,389.0 / 140,537 bytes |
| Decoded depth | 480 x 640, uint16 |

### Repository `SimpleDepthClient`, five-second sample

| Measurement | Observed value |
| --- | --- |
| Frames | 147 |
| Receive/decode rate | 29.40 FPS |
| Color output | 480 x 640 x 3, uint8 BGR |
| Mean B/G/R channel values | 120.807 / 119.637 / 120.116 |
| Depth output after center crop | 480 x 480, float64 meters |
| Depth minimum / mean / maximum | 0.000 / 1.6765 / 9.5210 m |
| Zero-depth fraction after filtering | 0.000108507 |

The client's `latest_depth` is center-cropped to 480 x 480 while `latest_color` remains 640 x
480. Even if the server were changed to align color to depth, these two client arrays would
not remain pixel-corresponding after this asymmetric crop.

### Combined read-only check

`check_streams.py` simultaneously received fresh Unitree `rt/lowstate` and 29-30 new depth
frames per one-second reporting interval. Tactile was deliberately disabled. RGB decoding
was validated separately through `SimpleDepthClient.latest_color` and the raw multipart
sample.

## Workstation software used for collection

The local environment is `g1_deploy/.venv` with Python 3.11, Unitree SDK 1.0.1,
CycloneDDS 0.10.2, OpenCV 5.0.0, NumPy 2.4.6, and PyZMQ 27.1.0.

The accompanying update to `tutorials/04_cameras.md` warns that older robot-side RGB-D
servers may still default to port 5555 even though the RGB-D client documentation uses port
5556. The separate local DDS-interface and collection-config changes are not required to use
this calibration record and are not part of this documentation update.

## Reproduction commands

Inspect the physical camera on PC2:

```bash
ssh g1
rs-enumerate-devices -c
```

Start the current RGB-D server on PC2:

```bash
ssh g1
cd ~/osher/image_server
nohup python3 img_server_rgbd.py > /tmp/g1_img_server_rgbd.log 2>&1 < /dev/null &
```

Run the interactive RGB-D viewer on a graphical workstation session:

```bash
cd /home/clhsieh/aria/third_party/tactile/g1_deploy
source .venv/bin/activate
PYTHONPATH=. python common/depth_client.py \
  --server 192.168.123.164 --port 5556 --height 480 --width 640 --debug
```

When started with `nohup` as shown, the camera server continues running after the SSH session
ends. To stop precisely that instance, first verify its PID and command with
`ssh g1 'ps -ef | grep [i]mg_server_rgbd.py'`, then terminate that confirmed PID. No RGB-D
server process was running and no process was listening on port 5556 at the conclusion of
this collection session.
