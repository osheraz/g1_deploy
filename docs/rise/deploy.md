# Robot recovery

Prepare a bundle using the [deployment guide](https://github.com/osheraz/hv2/blob/master/docs/guides/evaluation/deployment.md)
and keep `HV2_RECOVERY_BUNDLE` set. Robot settings live in
[`rise/config/models.yaml`](../../rise/config/models.yaml).

Bundle preparation does not verify hardware performance. Before use, check sensor
geometry, depth processing and clock alignment against the model's saved settings.

```bash
python g1_deploy/rise/deploy.py \
  --bundle "$HV2_RECOVERY_BUNDLE" --model ours --check
```

This check opens no sensors or motor channels. `--bundle` overrides the configured
model bundle; `HV2_RECOVERY_BUNDLE` is the fallback for unbound models.
Keep each planner paired with its matching tracker.

## Terminal 1: depth

After updating the server, copy both standalone files once:

```bash
cd /home/osher/hv2
scp g1_deploy/rise/sensing/native_depth_server.py \
  g1_deploy/core/sensing/depth_wire.py g1_29:/home/unitree/osher/native_recovery/
```

Start the depth server and both SSH forwards; leave this terminal running:

```bash
ssh -t -o AddressFamily=any -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:15556:127.0.0.1:5556 \
  -L 127.0.0.1:15558:127.0.0.1:5558 g1_29 \
  '/home/unitree/osher/native_recovery/.venv/bin/python /home/unitree/osher/native_recovery/native_depth_server.py --serial 247122073927 --processing raw --encoding raw --max-mbps 0 --bind tcp://127.0.0.1:5556 --clock-bind tcp://127.0.0.1:5558'
```

Raw transport requires gigabit Ethernet. On a 100 Mb/s link, replace
`--encoding raw --max-mbps 0` with `--encoding png --max-mbps 40`.
Both preserve the same source depth values. [Depth troubleshooting](troubleshooting.md#depth).

## Sensor session and operator controls

After configuring the robot interface and camera, a sensor-only session is:

```bash
python g1_deploy/rise/deploy.py \
  --bundle "$HV2_RECOVERY_BUNDLE" --model ours --backend pytorch \
  --duration-s 30 --record-sensors --record-depth --record-model --output "$NEW_SESSION"
```

Motor output requires `--actuate`. Stop other motor publishers before using it.

| Control | Action |
| --- | --- |
| A after readiness | Start recovery from the measured pose |
| START (optional) | Move to default joints or `--start-pose` JSON |
| A again / SELECT / Ctrl+C | Stop the session |
| Y with `--standing-hold` | Request [standing mode](standing-hold.md) |

Wait for recording to finish saving before using its files.

Use [steering](steering.md) for explicit goals and [recording](recording.md) for review.
