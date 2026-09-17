# Robot recovery

From `/home/osher/hv2`, run **two terminals**. Setup once: [project README](../README.md).
These commands include recording and live pelvis XYZ; no separate recorder is needed.

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
  '/home/unitree/osher/native_recovery/.venv/bin/python /home/unitree/osher/native_recovery/native_depth_server.py --serial 247122073927 --processing vigor --encoding raw --max-mbps 0 --bind tcp://127.0.0.1:5556 --clock-bind tcp://127.0.0.1:5558'
```

Raw transport requires gigabit Ethernet. On a 100 Mb/s link, replace
`--encoding raw --max-mbps 0` with `--encoding png --max-mbps 40`.
Both send the same filtered depth values. [Depth troubleshooting](status.md#depth).

## Terminal 2: recovery

Place the robot on the floor in the intended orientation, or support it.
**An actuated launch enters zero torque before START/A.** Stop any other motor
publisher, including `zero_torque.py`, before launching.

Choose **one** command. Prone = face down; supine = face up:

```bash
cd /home/osher/hv2
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --preset prone --model r8_s25_diffusion --seed random --replan-hz 2.5
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --preset supine --model r8_s25_diffusion --seed random --replan-hz 2.5
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --preset prone --model r8_s50r_diffusion --seed random --replan-hz 5
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --preset supine --model r8_s50r_diffusion --seed random --replan-hz 5
```

All four use TensorRT, fresh noise each plan, no joint jitter, live XYZ,
raw sensors, full-resolution depth, model traces and the external camera.
They stop after 60 seconds; change `--seconds` to set another duration.
Each launch prints its new `logs/real_world/MODEL_POSTURE_TIMESTAMP` folder.

1. Wait for streams, the five-second warmup and **Live inference ready**.
2. **A** starts recovery from the measured pose and starts recording.
   **START is optional**: move to the selected joints, wait for readiness, then A.
   Use START when comparing the same joint targets across trials.
3. **A again / SELECT / Ctrl+C** stops and damps. Wait for
   **Saved and verified raw sensors** and the model-trace save status.

The preset controls joints, not floor position or body orientation.
The extra command limiter runs only during the first **0.5 s** of policy execution.
Normal depth/timing delays warn or use existing fallback behavior; this reorganization
adds no new stop conditions.

## Current pose, without a joint preset

A starts from the measured pose. Optional START in this command selects the
standard default joint targets instead of a lying preset.

```bash
.venv/bin/python g1_deploy/steerable-recovery/deploy.py \
  --model r8_s25_diffusion --backend tensorrt --seed random \
  --noise-policy fresh --replan-hz 2.5 --actuate --duration-s 60 \
  --live-odometry --record-sensors --record-depth --record-model \
  --external-camera --camera-size 1920x1080 --camera-fps 30
```

## Replay and zero torque

After the run, pass the printed folder name:

```bash
.venv/bin/python g1_deploy/steerable-recovery/replay.py RUN_NAME
```

Viser: **http://localhost:8081**. [Videos, depth and replay options](recording.md).
For steering use the [separate commands](steering.md).

Stop deployment before running zero torque:

```bash
.venv/bin/python g1_deploy/zero_torque.py
```

This makes the robot limp. Ctrl+C stops the publisher; it does not restore standing.
