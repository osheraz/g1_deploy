# 6. Tactile (e-skin)

Optional. Skip with `use_tactile: false`.

The only piece needing ROS 2, via `rclpy` in
[`common/tactile_client.py`](../common/tactile_client.py). A plain uv venv has no ROS, so
this is the one case where `pixi` is needed — `pixi.toml` pulls `ros-humble-desktop`:

```bash
pixi run bootstrap
source .envrc
uv pip install -e .
```

Topics are `/jq_tactile/left_hand/LH/...` and `/jq_tactile/right_hand/RH/...`, published by
the robot-side `ros_ws/`, which expects `/dev/jq_hand_left` and `/dev/jq_hand_right`.

Then set `use_tactile: true` and `tactile_sensor_order` in the config.

`check_streams.py` shows tactile panels in viser alongside robot state.

See the shared quickstart in
[`e_skin`](https://github.com/osheraz/e_skin/blob/master/docs/quickstart.md).
