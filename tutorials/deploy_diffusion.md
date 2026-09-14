# Run P11 recovery on G1

**Start here on the configured bluefin workstation.** Use two terminals: depth,
then policy. The policy command owns LiDAR, live odometry and recording.
The default is Stanley's **P11 diffusion20k + N17k depth tracker**, using
**PyTorch TF32, EMA DDIM8, fresh noise, K20/blend0**. `--model diffusion`
selects the older September 10 model, not P11.
For simulation use [MuJoCo](sim_diffusion.md); for a new machine use
[setup](GETTING_STARTED.md). [Current validation / remaining work](deployment_status.md).

## 0. Check the new models — no robot needed

The local P11 bundle is already installed on bluefin:

```bash
cd /home/osher/hv2
.venv/bin/python g1_deploy/deploy_diffusion.py --check
```

This checks hashes, contracts, original training PD parameters and CPU inference.
It opens no sensors. For another workstation, copy the bundle using the
[reference instructions](deployment_reference.md#model-bundle).

## 1. Depth — terminal 1

```bash
ssh -t -o AddressFamily=any -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:15556:127.0.0.1:5556 \
  -L 127.0.0.1:15558:127.0.0.1:5558 \
  g1_29 \
  '/home/unitree/osher/native_recovery/.venv/bin/python /home/unitree/osher/native_recovery/native_depth_server.py --serial 247122073927 --bind tcp://127.0.0.1:5556 --clock-bind tcp://127.0.0.1:5558'
```

Leave this running. If these ports already have a working depth stream/tunnel,
use it. Do not start another camera server.

## First P11 sensor check — terminal 2

With the robot stationary and supported on the hoist, run:

```bash
cd /home/osher/hv2
.venv/bin/python g1_deploy/deploy_diffusion.py \
  --model p11_diffusion --backend pytorch --duration-s 30 \
  --record-sensors --record-label p11_stationary_hoist
```

No A press or motor output. This checks the real depth stream, both networks and
recording together. Inspect camera ages, planner rejections and late control ticks
in the printed session's `COMPLETE.json` and `telemetry.jsonl.gz`. It cannot test
get-up behavior while the robot is stationary. Add `--live-odometry` to also check
XYZ monitoring; unguided P11 does not need it.

## 2. Recovery + live XYZ recording — terminal 2

For the next ground evaluation, start a fixed camera recording the pelvis and
measured floor marks. Place the robot in its starting ground posture and leave
it still during initialization. Keep the remote available; run one motor process.

```bash
cd /home/osher/hv2
SESSION="logs/native_recovery/ground_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python g1_deploy/deploy_diffusion.py \
  --model p11_diffusion --backend pytorch --actuate --release-controller \
  --live-odometry --duration-s 60 --record-label ground_recovery \
  --output "$SESSION"
```

Wait for policy readiness and **Live XYZ ready for monitoring (clock unqualified)**.
Then press **A**: policy and recording start from the current pose. No automatic
move to standing. First control observation anchors relative XYZ and orientation.

| Control | Result |
| --- | --- |
| A while ready | Start policy + recording |
| A again, SELECT or Ctrl+C | Stop policy, damp, then save recording |
| START before policy | Optional move to default; skip it for ground recovery |

Release buttons between presses. Retain about five seconds after recovery before
stopping. The duration limit starts with the policy. Wait for **Saved and verified
raw sensors** before closing the terminal. Stop the depth terminal with Ctrl+C
when finished. Restart the deployment command with a new session for another run.

**Timing delays do not end the running session.** A late inference result can
still publish when current feedback is fresh; delays above 40 ms are logged.
If feedback is stale, a plan runs out, or the writer sees a command gap above
60 ms, it holds a measured joint pose with position PD. Obsolete plans are
removed and policy control resumes with fresh feedback and a fresh plan. The
60 ms value selects a hold; it is not a shutdown timeout. There is no catch-up
burst of old commands. Episode time and steering origin are preserved.

Operator stops, the requested duration, invalid state and actual transport
failures retain their stop behavior. Position hold does not guarantee balance
during a moving recovery. Delay handling has mock-robot tests, not physical
validation; test it with support before relying on it on the ground.

**XYZ is monitoring only:** clock bounds and dynamic accuracy remain unqualified.
Depth stays the planner input. The fixed camera provides a gross motion check,
not full 3D ground truth. Save its video with the session.

## 3. Evaluate and view — after stopping, terminal 2

```bash
.venv/bin/python g1_deploy/evaluate_live_motion.py "$SESSION"
.venv/bin/python g1_deploy/replay_motion.py "$SESSION" --live-estimates --port 8081
```

In a new terminal, first `cd /home/osher/hv2` and set `SESSION` to the printed run
path. Forward port **8081** in VS Code, then open **http://localhost:8081**.
The second command starts Viser only when you run it.

Check relative XYZ, heading and orientation against the video. The camera stays
fixed in the world; the robot moves using the **estimates logged during control**.
Missing/invalid positions remain gaps. Inspect `live_evaluation/evaluation.json`
and `motion.png` for availability, estimated age, tilt/heading disagreement and
policy timing. Recording completeness is in `sensors/recording_result.json`.

## Other runs

- **Replanning comparison:** add `--replan-hz 5` or `--replan-hz 10` to the
  sensor-only or motor command. P11's default is 2.5 Hz; tracker stays at 50 Hz.
  The existing reply-age/horizon checks remain active.
- **Blind-tracker comparison:** use `--model p05_diffusion --backend pytorch`.
  P05 still uses depth in its planner and selects its own A24.5k blind tracker.
  Hardware uses 2.5 Hz by default to leave room for replies; MuJoCo defaults to
  its reference 2 Hz. This pair has not been physically commissioned.

- **Legacy models:** `--model diffusion --backend tensorrt` runs B20k/C10k;
  `--model flow --backend tensorrt` runs E20k/C10k. Neither selects P11.
- **Sensor-only hoist check:** omit `--actuate --release-controller`, use
  `--record-label stationary_hoist` and a new session. Inference and recording
  start automatically; no A press or motor output. This checks compute/streams,
  not recovery behavior against a stationary robot.
- **Policy + raw recording, without live XYZ:** omit `--live-odometry`.
  [Recording/replay options](../RECORDING.md).
- **Policy only:** omit `--live-odometry` and add `--no-record-sensors`.
- **Existing LiDAR driver:** add `--existing-lidar-driver` only when deliberately
  reusing a working driver. Otherwise deployment starts and stops its own.

No separate `record_motion.py`, LiDAR command or ROS terminal is needed.
For missing streams, inspect `sensors/capture.log` and `sensors/driver.log`;
recorder failure does not stop ordinary policy execution, but an incomplete
capture cannot validate XYZ. Do not press A for this evaluation until streams
and live XYZ are ready. A rejected policy preview is reported separately; inspect
its reason rather than bypassing it.

Configuration: [diffusion.yaml](../configs/diffusion.yaml).
Model/export details: [reference](deployment_reference.md).
Sensor installation: [permanent workspace](../config/lidar/README.md).
