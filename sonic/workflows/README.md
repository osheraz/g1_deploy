# Deployment workflows

User-facing launchers are grouped by input operation rather than model or
embodiment. This keeps the three independent choices explicit.

- [`joystick/`](joystick/README.md): planner-only keyboard/Unitree-remote
  MuJoCo and physical workflows.
- `pico`: the stable three-terminal operator interface remains under
  `sonic_cpp/teleop/` because its flat `t1`/`t2`/`t3` names are operationally
  useful; see [`../../sonic_cpp/teleop/README.md`](../../sonic_cpp/teleop/README.md).

Within an operation, generic launchers accept a full model/embodiment
selection. Reviewed fixed combinations live in a `presets/` subdirectory.
Historical entry points under `g1_deploy/sonic/scripts/` are compatibility wrappers,
not alternate implementations.
