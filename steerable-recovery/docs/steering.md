# Steering

Pass an explicit `hv2.hardware_motion_goal.v1` JSON with `--goal-json`. The same
goal validator and cost are used by simulation and hardware. The frame is the
initial pelvis origin with causal heading; physical time does not restart when
the planner replans. `--guidance-scale` overrides strength without modifying the
target, gains or checkpoint.

```bash
python g1_deploy/steerable-recovery/sim.py --model ours --backend pytorch \
  --recorded-start "$START_JSON" --goal-json "$GOAL_JSON" --seed 42 --seconds 15
```

Guidance requires the PyTorch planner for gradients. Hardware position goals
also require measured pelvis feedback, supplied by live odometry. Qualified
feedback is the default; `--steering-feedback estimated` explicitly permits
experimental clock bounds while retaining freshness and geometry checks.
Invalid feedback suspends steering according to the saved execution policy.

For the paper's starts, shared targets and replanning comparisons, use
[the evaluation workflow](../../../docs/recovery/evaluation.md). These generic
MuJoCo/robot commands do not substitute for native evaluation.
