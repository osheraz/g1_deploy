"""Deprecated VIGOR2 parity script; not the current planner/action-decoder evaluator."""

"""Deploy stack vs the in-sim planner+tracker, on identical states.

Drives the env with the in-sim path, and every step feeds the deploy runner's pieces the same
robot state through the deploy code path (FK from encoders + IMU instead of sim body tensors).
Checks, in order: dof ordering, action_scale / default angles, actor_obs layout, FK body poses,
encoded planner state, and the tracker's actions.

  python g1_deploy/deprecated/scripts/check_deploy_parity.py \
    +checkpoint=<tracker>/model_last.pt +planner=<gen>/model.pt \
    +terrain=terrain_fall_steps +num_envs=4 +headless=true \
    ++env.config.ego_camera.ego_camera=true \
    ++env.config.reference_motion.file=<pack> +steps=40
"""

import logging
import os
import sys
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "g1_deploy"))

from humanoidverse2.utils.config_utils import *  # noqa: F403


@hydra.main(config_path=str(ROOT / "humanoidverse2/config"), config_name="base_eval",
            version_base="1.1")
def main(override_config: OmegaConf) -> None:
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logging.basicConfig(level=logging.ERROR)
    for n in ("omni", "carb", "asyncio", "trimesh", "yourdfpy", "jax", "absl",
              "isaaclab", "isaacsim", "warp", "matplotlib", "h5py", "pyroki"):
        logging.getLogger(n).setLevel(logging.ERROR)
    os.chdir(hydra.utils.get_original_cwd())

    from humanoidverse2.utils.wandb_checkpoint import (
        load_training_config_from_checkpoint, resolve_checkpoint_path,
    )

    if override_config.checkpoint is None or override_config.get("planner") is None:
        raise ValueError("Set +checkpoint=<tracker> and +planner=<generator model.pt>")
    train_config = load_training_config_from_checkpoint(
        str(override_config.checkpoint),
        Path(override_config.eval_log_dir) / ".wandb_downloads",
        default_entity=None, default_project=override_config.get("project_name"),
    )
    config = OmegaConf.merge(train_config, override_config)
    config.simulator.config.with_multi_viewport_camera = False
    config.simulator.config.init_viser = False
    config.simulator.config.record0 = False

    from isaaclab.app import AppLauncher
    import argparse

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    args_cli.num_envs = config.num_envs
    args_cli.seed = config.seed
    args_cli.env_spacing = config.env.config.env_spacing
    args_cli.headless = config.get("headless", True)
    args_cli.enable_cameras = True
    app = AppLauncher(args_cli).app

    import numpy as np
    import torch
    from hydra.utils import instantiate
    from hydra.core.hydra_config import HydraConfig
    from humanoidverse2.utils.helpers import pre_process_config

    pre_process_config(config)
    device = config.get("device") or "cuda:0"
    log_dir = Path(HydraConfig.get().runtime.output_dir)
    config.checkpoint = str(resolve_checkpoint_path(
        config.checkpoint, log_dir / ".wandb_downloads",
        default_project=config.get("project_name")))
    env = instantiate(config.env, device=device)

    from humanoidverse2.agents.base_algo.base_algo import BaseAlgo

    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=str(log_dir))
    algo.setup()
    algo.load(config.checkpoint)
    algo._eval_mode()
    env.set_is_evaluating()
    env.depth_transform = lambda d: d

    checks = run_checks(config, env, algo, str(override_config.planner), device)
    report(checks)
    os._exit(0 if all(c[1] for c in checks) else 1)


def run_checks(config, env, algo, planner_path, device):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from deprecated.vigor2.runner import FallRecRunner
    from deprecated.vigor2.tracker import ACTOR_OBS_KEYS, ZG_KEY

    checks = []
    runner = FallRecRunner(SimpleNamespace(
        tracker_checkpoint=config.checkpoint, planner_model=planner_path, planner_cfg=None,
        planner_sampler="ddim", planner_inference_steps=12, planner_action_horizon=2,
        planner_async=False, planner_executable_only=True, planner_blend_frames=0,
        depth_source="full", depth_min_d=0.0,
        depth_max_d=float(config.env.config.ego_camera.clipping_range.max_d),
        depth_crop_px=tuple(config.env.config.ego_camera.crop_px),
        yaw_command_deg=0.0, stand_takeover=False, stand_up_threshold=0.9,
    ), device=device)

    sim_dof = [str(n) for n in env.simulator.dof_names]
    checks.append(("dof ordering: deploy vs simulator", sim_dof == runner.dof_names,
                   f"{sim_dof[:3]}... vs {runner.dof_names[:3]}..."))

    sim_bodies = [str(n) for n in env.simulator._body_list]
    bundle_bodies = [str(n) for n in runner.planner.bundle.body_names]
    shared = bundle_bodies[:len(sim_bodies)]
    checks.append(("body ordering: planner bundle vs simulator",
                   shared == sim_bodies[:len(shared)],
                   f"{len(bundle_bodies)} bundle vs {len(sim_bodies)} sim bodies"))

    obs_keys = [str(k) for k in config.obs.obs_dict.actor_obs]
    checks.append(("actor_obs layout: deploy concat order == sorted(obs keys)",
                   sorted(obs_keys) == list(ACTOR_OBS_KEYS),
                   f"{sorted(obs_keys)} vs {list(ACTOR_OBS_KEYS)}"))

    head = set(str(k) for k in config.algo.config.module_dict.actor.input_dim)
    checks.append(("tracker head reads only proprio + z_g", head <= {"actor_obs", ZG_KEY},
                   f"input_dim={sorted(head)}"))

    robot_yaml = str(config.get("robot_yaml", "g1_29dof_native_fallrec.yaml"))
    try:
        runner.verify_robot_config(_deploy_robot_config(robot_yaml))
        ok, detail = True, (
            f"{robot_yaml} order/frame/defaults/scales/Kp/Kd match training"
        )
    except Exception as exc:
        ok, detail = False, f"{robot_yaml}: {exc}"
    checks.append(("deploy robot yaml vs training gains", ok, detail))

    obs = env.reset_all()
    for k in obs:
        obs[k] = obs[k].to(device)

    e, anchor = 0, int(runner.planner.bundle.anchor_body_id)
    fk_err, act_err, state_err, n = 0.0, 0.0, 0.0, 0
    with torch.inference_mode():
        for step in range(int(config.get("steps", 40))):
            actions = algo.actor.act_inference(obs).detach()
            zg_sim = algo.actor.actor_module.last_feats[ZG_KEY][e:e + 1]
            obs_sim = obs["actor_obs"][e:e + 1]

            dof_pos = env.simulator.dof_pos[e:e + 1]
            dof_vel = env.simulator.dof_vel[e:e + 1]
            body_rot = env.simulator._rigid_body_rot[e:e + 1]
            from motion_diffusion.features.rotation import quat_apply, quat_conjugate

            anchor_rot = body_rot[:, anchor]
            base_ang_vel_body = quat_apply(
                quat_conjugate(anchor_rot),
                env.simulator._rigid_body_ang_vel[e:e + 1, anchor],
            )
            runner.state.push(dof_pos, dof_vel, anchor_rot, base_ang_vel_body)

            if runner.state.ready:
                nb = len(runner.state.body_names)
                fk_err += float(_quat_angle(runner.state.body_rot[0, 0],
                                            body_rot[0, :nb]).max())
                # frame 0 only: _sim_window holds one instant, the deploy ring holds real history
                sim_win = _sim_window(env, runner, e)
                state_err += float(
                    (runner.planner._encode_state(runner.state.window())[:, 0]
                     - runner.planner._encode_state(sim_win)[:, 0]).abs().max()
                )
                proprio = _split_actor_obs(obs_sim, config, ACTOR_OBS_KEYS)
                act_dep = runner.tracker.act(proprio, zg_sim)
                act_err += float((act_dep - actions[e:e + 1]).abs().max())
                n += 1

            obs, _, _, _ = env.step({"actions": actions})
            for k in obs:
                obs[k] = obs[k].to(device)

    m = max(n, 1)
    checks.append((f"tracker actions: deploy path vs sim ({n} frames)", act_err / m < 1e-4,
                   f"max|delta| {act_err / m:.2e}"))
    checks.append((f"planner state encoding: FK vs sim tensors", state_err / m < 5e-2,
                   f"max|delta| {state_err / m:.4f}"))
    checks.append((f"FK body rotation vs simulator", fk_err / m < 0.05,
                   f"mean max angle error {np.degrees(fk_err / m):.3f} deg"))
    return checks


def _deploy_robot_config(name: str):
    """Load inherited native robot fields without constructing a DDS RobotInterface."""
    from core.config import Config

    return Config(name)


def _sim_window(env, runner, e):
    """The same window the deploy history holds, but filled from sim body tensors."""
    nb = len(runner.state.body_names)
    rep = lambda x: x[e:e + 1, :nb].unsqueeze(1).expand(-1, runner.state.H, -1, -1)  # noqa: E731
    sim = env.simulator
    return {
        "body_pos": rep(sim._rigid_body_pos), "body_rot": rep(sim._rigid_body_rot),
        "body_vel": rep(sim._rigid_body_vel), "body_ang_vel": rep(sim._rigid_body_ang_vel),
        "dof_pos": sim.dof_pos[e:e + 1].unsqueeze(1).expand(-1, runner.state.H, -1),
        "dof_vel": sim.dof_vel[e:e + 1].unsqueeze(1).expand(-1, runner.state.H, -1),
    }


def _split_actor_obs(actor_obs, config, keys):
    """Undo the env's actor_obs concat so the deploy tracker re-assembles it itself."""
    raw = config.obs.obs_dims
    items = raw.items() if hasattr(raw, "items") else ((k, v) for d in raw for k, v in d.items())
    dims = {str(k): int(v) for k, v in items}
    out, i = {}, 0
    for key in sorted(str(k) for k in config.obs.obs_dict.actor_obs):
        out[key] = actor_obs[:, i:i + dims[key]]
        i += dims[key]
    if i != actor_obs.shape[-1] or set(out) != set(keys):
        raise ValueError(f"actor_obs split mismatch: {i} vs {actor_obs.shape[-1]}, {sorted(out)}")
    return out


def _quat_angle(a, b):
    import torch

    dot = (a * b).sum(-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def report(checks) -> None:
    logger.info("=== deploy parity ===")
    for name, ok, detail in checks:
        logger.info("{} {:<52} {}", "PASS" if ok else "FAIL", name, detail)
    bad = [c[0] for c in checks if not c[1]]
    if bad:
        logger.error("FAILED: {}", ", ".join(bad))
    else:
        logger.info("all checks passed")


if __name__ == "__main__":
    main()
