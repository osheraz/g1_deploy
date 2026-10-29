# Robot recovery

Use the [verified release bundle](../../../docs/recovery/deployment.md) and the
robot/site configuration in `config/models.yaml`. The current pair is not
hardware-qualified. Check sensor geometry, depth processing and clock alignment
against its saved contract before commissioning it.

```bash
python g1_deploy/steerable-recovery/deploy.py --model ours --check --max-plan-age-ticks 10
```

The 10-tick reply budget in these examples fits the 50-tick horizon and is
an explicit commissioning setting, not measured hardware qualification. The
bundle retains its provisional 50-tick value; it cannot satisfy the asynchronous
horizon check unchanged. Choose a valid budget from measured latency before
actuation. This check opens no sensors or motor channels. `--bundle PATH` selects a verified
bundle for another published pair; the planner and tracker must stay paired.

## Terminal 1: depth

After updating the server, copy both standalone files once:

```bash
cd /home/osher/hv2
scp g1_deploy/steerable-recovery/steerable_recovery/sensing/native_depth_server.py \
  g1_deploy/core/common/depth_wire.py g1_29:/home/unitree/osher/native_recovery/
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
Both preserve the same source depth values. [Depth troubleshooting](status.md#depth).

## Sensor session and operator controls

After configuring the robot interface and camera, a sensor-only session is:

```bash
python g1_deploy/steerable-recovery/deploy.py --model ours --backend pytorch --max-plan-age-ticks 10 \
  --duration-s 30 --record-sensors --record-depth --record-model --output "$NEW_SESSION"
```

Motor output requires the explicit `--actuate` flag. Stop other motor publishers
before an actuated session. After readiness, A starts from the measured pose;
START optionally moves to the default joints or an explicit `--start-pose` JSON.
A again, SELECT or Ctrl+C stops the session. Wait for recording completion before
reusing its files. `--standing-hold` enables the [manual Y mode](standing-hold.md).

Use [steering](steering.md) for explicit goals and [recording](recording.md) for
review. Sensor settings and historical hardware evidence do not confer
qualification on a different checkpoint.
