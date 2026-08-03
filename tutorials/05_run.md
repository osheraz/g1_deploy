# 6. Run

## Check the streams first

`check_streams.py` never sends commands:

```bash
python check_streams.py --config g1_23dof.yaml
```

Want a `robot: ok` line with a rising `msgs=` and a fresh `age=`. Stuck on
`waiting for low_state...` means the robot isn't publishing — recheck
[1. Network](01_network.md) and that the robot isn't in a mode that suppresses
`rt/lowstate`.

`viser_display: true` opens a browser view of robot state.

## Run the policy

**The robot moves the moment this starts.** Clear the area, remote in hand.

```bash
python deploy_g1.py --net enp132s0 --config g1_23dof.yaml --log TEST.npz
```

Sequence:

1. Ramps every joint to the default pose over 2s —
   [deploy_g1.py:377](../deploy_g1.py#L377). **No button gate; this happens on launch.**
2. Holds default, printing `Waiting for the Button A signal...`
3. **A** → policy starts.
4. **select** → stop; sends damping on the way out.

Ctrl-C works during the policy loop (also ends in damping) but is *not* caught during the
initial 2s ramp.

Startup fails before any motion if the LocoClient can't connect or the camera won't open.

Any remote combo needed to get the G1 out of its built-in sport mode into a state accepting
`rt/lowcmd` is firmware-side, not in this repo. Check the Unitree manual for your firmware.

## Logs

`--log` writes an `.npz` per run:

```bash
PYTHONPATH=. python tools/replay_rollout.py --log TEST.npz
```

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `waiting for low_state...` | no DDS from robot — [1. Network](01_network.md) |
| `LocoClient connection failed` | robot unreachable or refusing the client |
| policy runs but behaves wrong | `obs_dim`/scales don't match the trained config — [3. Config](03_config.md) |
| joint reads outside its URDF limit | encoder zero is off — mechanical, not config |
