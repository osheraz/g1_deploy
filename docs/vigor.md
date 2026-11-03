# VIGOR deployment

VIGOR runs its reactive visual policy without a diffusion planner.
Configure [`configs/g1_23dof.yaml`](../configs/g1_23dof.yaml), including policy
weights, joint mapping, depth server and network interface.

```bash
python g1_deploy/vigor/deploy.py --help
```

**This launcher moves the robot during startup.** It is not a sensor viewer.
Use [sensor checks](setup/sensors.md) to inspect depth and state without motion.

`vigor/policy_runner.py` assembles the trained observation/history layout;
`student_model.py` builds the policy. VIGOR's depth preprocessing differs from
RISE's calibrated metric-depth input. Preserve the selected checkpoint's contract.

Replay a saved VIGOR log with:

```bash
python g1_deploy/vigor/replay.py --log /path/to/recording.npz
```
