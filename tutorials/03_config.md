# Model and hardware configuration

Recovery settings: [`steerable-recovery/config/models.yaml`](../steerable-recovery/config/models.yaml).
Shared robot geometry: [`configs/g1_29dof_sonic.yaml`](../configs/g1_29dof_sonic.yaml).
The verified model bundle supplies training PD, default joints and action scale.
Do not substitute SONIC gains or exchange paired trackers.

```bash
cd /home/osher/hv2
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --model r8_s25_diffusion --check
.venv/bin/python g1_deploy/steerable-recovery/deploy.py --model r8_s50r_diffusion --check
```

These validate the selected bundle without sensors or motor commands.
Use `--config PATH` for another configuration. Explicit model selection avoids
the full launcher's historical P11 default.

[Operator commands](../steerable-recovery/docs/deploy.md) ·
[Runtime layout](../steerable-recovery/docs/architecture.md) ·
[Export/benchmark tooling](../steerable-recovery/docs/performance.md).

The read-only `sonic/check_streams.py` uses shared camera/joint settings; it does
not select or load the recovery policy. VIGOR settings remain
[separate](../deprecated/tutorials/config_vigor.md).
