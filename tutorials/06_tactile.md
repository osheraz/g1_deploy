# 6. Tactile (e-skin)

Optional SONIC/sensor-viewer feature; the current diffusion planner does not consume tactile.
Use `--no-tactile` in the read-only viewer when it is disconnected.

Tactile arrives over ZMQ, exactly like depth — no ROS, no pixi. The robot runs a
publisher; [`sonic/common/tactile_client.py`](../sonic/common/tactile_client.py) subscribes.

## Robot

```bash
ssh g1
/usr/bin/python3 -m jq_tactile_driver.server --role left_hand right_hand
```

Runs on the system python with only `pyserial` + `pyzmq`. Auto-detects boards by
USB id if you omit `--role`. Prints per-sensor rates every 5 s:

```
[tactile_server] publishing on tcp://*:5557
[tactile_server] LH=187Hz RH=249Hz
```

## Remote

Set in the config:

```yaml
use_tactile: true
tactile_sensor_order: ["LH", "RH"]
tactile_server_ip: 192.168.123.164
tactile_server_port: 5557
```

Then `sonic/check_streams.py` shows tactile panels in viser alongside robot state, and
`deploy_vigor.py` logs `tactile_<sensor>_raw_u8` / `_imu_u8` per step.

For data collection, `deploy_eskin.py` records
tactile together with colour, depth and robot state.

Check the stream on its own:

```bash
python -m sonic.common.tactile_client --server 192.168.123.164
```

## Wire format

One multipart ZMQ message per frame:

| part | bytes | contents |
| --- | --- | --- |
| 0 | 2 | sensor id, ascii (`LH`, `RH`, `LF`, `RF`, `WB`) |
| 1 | 256 | raw taxel bytes |
| 2 | 16 | raw IMU bytes (float32 quaternion wxyz) |

Part 0 doubles as the ZMQ subscription prefix, so a client only receives the
sensors it asked for.

Decoding lives in
[`jq_tactile_driver`](https://github.com/osheraz/e_skin/tree/master/jq_tactile_driver)
— see [`docs/tactile.md`](https://github.com/osheraz/e_skin/blob/master/docs/tactile.md).
