"""Load a VIGOR checkpoint with the shared PPO actor implementation."""

import torch
from humanoidverse2.agents.ppo.modules import PPOActor


class DeployStudentPolicy:
    def __init__(self, ckpt_path: str, deploy_config, device: str = "cpu"):
        self.device = device

        actor_obs_keys = deploy_config.actor_obs_keys
        obs_dim = deploy_config.obs_dim
        hist_cfg = deploy_config.history_actor_long_config
        T = next(iter(hist_cfg.values()))

        obs_aux = deploy_config.train_cfg.obs.obs_auxiliary
        if "depth_history" in obs_aux and "depth" in obs_aux.depth_history:
            depth_T = obs_aux.depth_history.depth
        else:
            depth_T = 0

        H = deploy_config.depth_img_height
        W = deploy_config.depth_img_width
        depth_flat = H * W

        self.algo_obs_dim_dict = {
            "actor_obs": sum(obs_dim[k] for k in actor_obs_keys if k in obs_dim),
            "student_hist_obs_long": sum(obs_dim[k] * T for k in hist_cfg if k in obs_dim),
        }
        if deploy_config.use_depth_obs:
            self.algo_obs_dim_dict["perception_obs"] = depth_flat
            if depth_T > 0:
                self.algo_obs_dim_dict["perception_obs_history_obs"] = depth_flat * depth_T

        student_actor_cfg = deploy_config.student_actor_cfg

        self.student = PPOActor(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=student_actor_cfg,
            num_actions=deploy_config.num_actions,
            init_noise_std=0.01,
        ).to(self.device)

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.student.load_state_dict(checkpoint["student_model_state_dict"], strict=True)
        self.student.eval()

    @torch.no_grad()
    def act(self, obs_dict):
        for k, v in obs_dict.items():
            obs_dict[k] = v.to(self.device)
        return self.student.act_inference(obs_dict)
