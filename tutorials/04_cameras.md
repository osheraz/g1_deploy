# 4. Cameras

One **Intel RealSense on the robot**, serial `246422070545`. Both RGB and depth come from
it. Neither goes through DDS — an image server on the robot's onboard PC
(`192.168.123.164`) grabs frames and publishes them over ZMQ, and your machine subscribes.

```
RealSense (on robot)
  └─ image server, ~/osher/image_server/ on the robot   ← robot side, not in this repo
       └─ ZMQ PUB
            └─ your machine subscribes
```

Nothing is plugged into your machine. Without the server running you get no frames and no
error — the subscriber just waits.

## The server

Run **`img_server_rgbd.py`** on the robot. It publishes colour and depth as a 2-part
message `[color_jpg, depth_png]` on port 5556, which covers both the deploy runtime and
data collection.

| Script | Port | Publishes | |
| --- | --- | --- | --- |
| `img_server_rgbd.py` | 5556 | color + depth, 2-part | use this |
| `img_server_osher.py` | 5555 | depth only, single-part | older, redundant |

Do not run both: they open the same camera, so the second gets `Device or resource busy`.
[`depth_client.py`](../core/common/depth_client.py) accepts either message shape, so nothing
downstream cares.

## Depth encoding

The RealSense filter chain runs on each frame (decimation → disparity → spatial → temporal
→ hole-filling), then the 16-bit `z16` depth is PNG-encoded. PNG because JPEG would destroy
the 16-bit values.

View depth alone with the built-in viewer in
[`core/common/depth_client.py`](../core/common/depth_client.py):

```bash
PYTHONPATH=. python core/common/depth_client.py --server 192.168.123.164 --port 5556 --debug
```

Grayscale window clipped to `0.1–2.0 m`. Press `q` to quit.

## RGB + depth

On the robot:

```bash
ssh g1
cd ~/osher/image_server && python3 img_server_rgbd.py
```

This one enables the color stream and sends a 2-part message: color as JPEG (quality 80),
depth as PNG so the `z16` values survive. At `480×640` that's roughly 18 KB color + 155 KB
depth per frame.

**On alignment.** Color and depth are published *unaligned* (`align_to_depth: False`), so
they are not pixel-corresponding — a pixel in the color image is not the same ray as that
pixel in the depth image.

Set `align_to_depth: True` in the config dict to reproject color into the depth frame. Be
aware of what that costs: on a D435 the **depth FOV (~87°) is wider than the color FOV
(~69°)**, so aligning color to depth leaves a **black border** around the color image where
there is simply no color data to fill the wider depth view. Anywhere depth is invalid goes
black too.

So: unaligned for looking at the streams, aligned only if you actually need
pixel-corresponding RGB-D and can live with the border.

View both from your machine:

```bash
python - <<'EOF'
import zmq, cv2, numpy as np
ctx = zmq.Context()
s = ctx.socket(zmq.SUB)
s.connect("tcp://192.168.123.164:5556")
s.setsockopt_string(zmq.SUBSCRIBE, "")
while True:
    color_msg, depth_msg = s.recv_multipart()
    color = cv2.imdecode(np.frombuffer(color_msg, np.uint8), cv2.IMREAD_COLOR)
    depth = cv2.imdecode(np.frombuffer(depth_msg, np.uint8), cv2.IMREAD_UNCHANGED)
    d = np.clip(depth * 0.001, 0.1, 2.0)
    vis = ((d - 0.1) / 1.9 * 255).astype(np.uint8)
    cv2.imshow("color", color)
    cv2.imshow("depth", vis)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
EOF
```

Depth arrives as raw uint16 counts — multiply by `0.001` for meters.

## Config knobs

In the server's `config` dict at the bottom of each file:

```python
"fps": 30,
"head_camera_image_shape": [480, 640],   # capture resolution [h, w]
"publish_shape": [480, 640],             # resize before publishing
"jpeg_quality": 80,                      # rgbd server only
```

`head_camera_id_numbers` defaults to `["246422070545"]` — change it if you swap cameras
(`rs-enumerate-devices` lists serials).

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| subscriber hangs, no frames | server not running on the robot |
| `Device or resource busy` | another image server already holds the RealSense |
| `No module named 'zmq'` | `uv pip install pyzmq` |
| depth all zeros | scene closer than the RealSense min range, or out past max |
| color black, depth fine | color stream not enabled — you're running the depth-only server |

---

Using depth as policy input: [3. Config](03_config.md#depth-as-policy-input).

Next: [5. Run](05_run.md)
