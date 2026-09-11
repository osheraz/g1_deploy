# Deprecated deployment code

Archived for reference, not the current diffusion/action-decoder implementation.

- `vigor2/`: former teacher-z_g planner/tracker, asynchronous planner/Viser
  processes, old depth worker and experiment YAMLs.
- `deploy_vigor2.py`: its historical launcher.
- `scripts/check_deploy_parity.py`: old teacher-z_g simulator parity diagnostic.
- `tutorials/`: legacy VIGOR run/config, filtered RGB-D and VIGOR2 LiDAR guides.

The older VIGOR reactive launcher is now explicitly named
[`../deploy_vigor.py`](../deploy_vigor.py). **It moves the robot during startup**;
it is not a read-only viewer or the current recovery planner.

Use [`../deploy_diffusion.py`](../deploy_diffusion.py) and
[the current guide](../tutorials/deploy_diffusion.md) for diffusion/flow.
That path has its own asynchronous planner process, timestamped sensing,
action-code queue and motor watchdog. It imports none of this archive.

Reusable DDS, sensor and logging helpers stay in `../core/`; SONIC stays in
`../sonic/`. Archived files remain recoverable here and in Git history.
