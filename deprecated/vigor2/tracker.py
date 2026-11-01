"""RL tracker loaded outside the env; z_g arrives as an obs_dict override on its preprocessor."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

ZG_KEY = "shared_goal_terrain_feat"
ACTOR_OBS_KEYS = ("actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity")


class DeployTracker:
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        from humanoidverse2.agents.ppo.modules import PPOActor

        ckpt_path = Path(ckpt_path).expanduser().resolve()
        cfg = _training_config(ckpt_path)
        algo_cfg = cfg.algo.config
        obs_dims = _obs_dim_dict(cfg)
        self.device = device
        self.num_actions = int(cfg.robot.actions_dim)
        self.actor = PPOActor(
            obs_dim_dict=obs_dims,
            module_config_dict=algo_cfg.module_dict.actor,
            num_actions=self.num_actions,
            init_noise_std=float(algo_cfg.init_noise_std),
        ).to(device)
        loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "actor_model_state_dict" not in loaded:
            raise KeyError(f"{ckpt_path} has no actor_model_state_dict")
        self.actor.load_state_dict(loaded["actor_model_state_dict"])
        self.actor.eval()
        self.config = cfg

        head_inputs = [str(k) for k in algo_cfg.module_dict.actor.input_dim]
        if set(head_inputs) - {"actor_obs", ZG_KEY}:
            raise ValueError(
                f"Tracker head reads {head_inputs}; deploy can only supply proprio and the "
                f"planner's z_g. A checkpoint needing privileged obs is not deployable."
            )
        # terrain_feat/goal_feat only build z_g, which the planner replaces
        self._suppressed = {
            name: torch.zeros(1, 1, device=device)
            for name in (algo_cfg.module_dict.actor.get("preprocessors") or {})
            if name != ZG_KEY
        }

    @torch.no_grad()
    def act(self, proprio: dict, zg: torch.Tensor) -> torch.Tensor:
        actor_obs = torch.cat([proprio[k] for k in ACTOR_OBS_KEYS], dim=-1)
        obs = {"actor_obs": actor_obs, ZG_KEY: zg, **self._suppressed}
        return self.actor.act_inference(obs)


def _training_config(ckpt_path: Path):
    import humanoidverse2.config.resolvers  # noqa: F401  registers ${eval:}

    for name in ("config.yaml", ".hydra/config.yaml"):
        candidate = ckpt_path.parent / name
        if candidate.is_file():
            return OmegaConf.load(candidate)
    raise FileNotFoundError(f"No training config next to {ckpt_path}")


def _obs_dim_dict(cfg) -> dict:
    dims = {k: int(v) for d in cfg.obs.obs_dims for k, v in d.items()}
    out = {}
    for group, keys in cfg.obs.obs_dict.items():
        out[group] = sum(dims[str(k)] for k in keys)
    return out
