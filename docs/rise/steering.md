# Steering

Pass a `hv2.hardware_motion_goal.v1` JSON with `--goal-json`. Simulation and hardware
use the same goal checks and steering cost.

Targets use the initial pelvis origin and causal heading. Replanning does not
restart command time. `--guidance-scale` changes strength while preserving the
target, gains and checkpoint.

Prepare the selected pair with the [deployment guide](https://github.com/osheraz/hv2/blob/master/docs/guides/evaluation/deployment.md)
and keep its `HV2_RECOVERY_BUNDLE` setting.

```bash
python g1_deploy/rise/sim.py \
  --bundle "$HV2_RECOVERY_BUNDLE" --model ours --backend pytorch \
  --recorded-start "$START_JSON" --goal-json "$GOAL_JSON" --seed 42 --seconds 15
```

Guidance needs the PyTorch planner to compute gradients. Hardware position goals
also need live pelvis odometry. Qualified feedback is the default.

`--steering-feedback estimated` permits experimental clock bounds, but keeps
freshness and geometry checks. Invalid feedback suspends steering as specified
by the saved execution settings.

For the paper's starts, shared targets and replanning comparisons, use
[the evaluation workflow](https://github.com/osheraz/hv2/blob/master/docs/guides/evaluation/README.md). These generic
MuJoCo/robot commands do not substitute for native evaluation.
