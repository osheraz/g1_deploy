# Joint calibration

Calibration measures the 23-DoF robot while suspended. It loads no policy.
Reusable protocols, fitting and the MuJoCo plant live in `sonic/calibration`;
commands live in `sonic/scripts/calibration`.

## Rehearse and record

```bash
bash g1_deploy/sonic/scripts/calibration/run_gated_calibration_mujoco.sh
bash g1_deploy/sonic/scripts/calibration/run_gated_calibration_robot.sh --protocol pilot
```

The robot command defaults to protocol/build checks without DDS. It requires the
separate `sonic_cpp` checkout beside `g1_deploy`. `G1_PYTHON` overrides the parent
checkout's Python. Outputs default to `logs/sonic/calibration/`.

Physical measurement requires `--arm-real-robot`, a foreground terminal and both
shoulder attachments, with limbs clear. Startup moves slowly to the initial pose.

| Control | Action |
| --- | --- |
| F1+F3 after `INITIAL POSE READY` | Capture a steady baseline |
| A | Run the displayed block |
| B | Skip or cancel, returning through neutral |
| SELECT / Ctrl+C | Enter damping |

Stale state, motor faults, excessive temperature or velocity also trigger damping.
Inspect the pilot before `--protocol screening`.

For manual simulation rehearsal:

```bash
bash g1_deploy/sonic/scripts/calibration/run_interactive_calibration_mujoco.sh
```

## Fit and interpret

```bash
python g1_deploy/sonic/scripts/calibration/fit_suspended_calibration.py state.csv \
  --coordinate-basis encoder_only --output provisional.json
```

An encoder-only fit measures gain, lag and hysteresis. It cannot establish joint
zero or contact compliance, and is insufficient for deployment.

Synchronize external measurements with `merge_external_calibration_truth.py`,
then fit with `--coordinate-basis external_truth`. Supply `time_s` and
`q_external_0` through `q_external_28` in the model frame.

Apply coordinate corrections in both directions: encoder angles to observations,
and model targets to motor commands. Correcting targets alone makes coordinates
inconsistent. Calibration does not install corrections automatically. Validate
external geometry and supported contacts before using one during locomotion.

Logs use 29-slot SONIC indexing; slots 13, 14, 20, 21, 27 and 28 are absent on the
23-DoF robot and are never commanded.
