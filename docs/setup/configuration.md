# Configuration

| Setting | File |
| --- | --- |
| Recovery models, camera and transport | [models.yaml](../../rise/config/models.yaml) |
| Shared 29-joint robot settings | [g1_29dof_sonic.yaml](../../configs/g1_29dof_sonic.yaml) |
| VIGOR | [g1_23dof.yaml](../../configs/g1_23dof.yaml) |
| SONIC adaptations | [Adaptation guide](../sonic/adaptations.md) |

Recovery gains, default joints, action scale and joint order come from the selected
model bundle. Keep the paired tracker/planner together. `--config PATH` selects
another hardware configuration; `--bundle PATH` selects model artifacts.

```bash
python g1_deploy/rise/deploy.py --bundle "$HV2_RECOVERY_BUNDLE" --model ours --check
```

This check opens no sensors or motor channels. See [deployment](../rise/deploy.md).
