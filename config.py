# from legged_gym import LEGGED_GYM_ROOT_DIR
LEGGED_GYM_ROOT_DIR = "/home/osher/HumanoidVerse2/deploy/configs"
import numpy as np
import yaml
import os
from pathlib import Path
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", lambda x: eval(x))


class Config:
    def __init__(self, file_path) -> None:
        with open(os.path.join(LEGGED_GYM_ROOT_DIR, file_path), "r") as f:

            config = yaml.load(f, Loader=yaml.FullLoader)

            self.viser_display = config.get("viser_display", False)
            # General settings
            self.control_dt = config["control_dt"]
            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.policy_path = config["policy_path"].replace(
                "{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR
            )

            # Joint-to-motor mapping (active joints)
            self.action_joint2motor_idx = config["action_joint2motor_idx"]
            self.kps = np.array(config["kps"])  # * 0.70
            self.kds = np.array(config["kds"])  # * 0.7
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            self.default_angles_23 = np.array(
                config["default_angles_23"], dtype=np.float32
            )

            # Fixed joints and motor indices
            self.fixed_joint2motor_idx = config["fixed_joint2motor_idx"]
            self.fixed_kps = np.array(config["fixed_kps"])
            self.fixed_kds = np.array(config["fixed_kds"])
            self.fixed_target = np.array(config["fixed_target"], dtype=np.float32)

            # Scaling factors
            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)

            # Number of actions and observations
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            self.obs_dim = config["obs_dim"]

            ckpt_path = Path(self.policy_path)
            self.train_cfg_path = ckpt_path.parent / "config.yaml"
            self.train_cfg = OmegaConf.load(self.train_cfg_path)

            self.actor_obs_keys = list(self.train_cfg.obs.obs_dict.actor_obs)
            self.history_actor_long_config = dict(
                self.train_cfg.obs.obs_auxiliary.history_actor_long
            )
            self.student_actor_cfg = (
                self.train_cfg.algo.config.student.module_dict.actor
            )

            # Motion reference configuration
            self.reference_motion = config["reference_motion"]
            self.reference_keyframes = config["reference_keyframes"]

            # Clipping configurations for actions/observations
            self.clip_obs = config["clip_obs"]
            self.clip_actions = config["clip_actions"]

            # Depth observation related parameters
            self.use_depth_obs = config["use_depth_obs"]

            # Tactile ROS parameters
            self.use_tactile = config.get("use_tactile", False)
            self.tactile_max_age_s = config.get("tactile_max_age_s", 0.05)
            self.tactile_sensor_order = config.get(
                "tactile_sensor_order",
                ["LF", "RF", "WB", "LH", "RH"],
            )

            # Depth clipping range
            self.min_d = config.get("min_d", 0.1)
            self.max_d = config.get("max_d", 2.0)
            self.depth_scale = config.get("depth_scale", 0.001)

            # Depth image resolution and server info
            self.depth_img_width = (
                self.train_cfg.env.config.ego_camera.width
            )  # config.get("depth_img_width", 96)
            self.depth_img_height = (
                self.train_cfg.env.config.ego_camera.height
            )  # config.get("depth_img_height", 64)

            self.depth_obs_dim = int(self.depth_img_width * self.depth_img_height)
            self.depth_server_ip = config.get("depth_server_ip", "192.168.123.164")
            self.depth_server_port = config.get("depth_server_port", 5555)

            # RealSense raw depth resolution
            self.depth_raw_height = config.get("depth_raw_height", 480)
            self.depth_raw_width = config.get("depth_raw_width", 640)

            print("Loaded configuration:")
            if self.use_depth_obs:
                print(f"Use depth observations: {self.use_depth_obs}")
                print(
                    f"Depth range: {self.min_d} to {self.max_d}, Scale: {self.depth_scale}"
                )
                print(
                    f"Depth image resolution: {self.depth_img_width}x{self.depth_img_height}"
                )
                print(
                    f"Depth server IP: {self.depth_server_ip}:{self.depth_server_port}"
                )
                print(
                    f"Raw depth resolution: {self.depth_raw_width}x{self.depth_raw_height}"
                )
            print(f"Use tactile ROS: {self.use_tactile}")
            if self.use_tactile:
                print(
                    f"Tactile freshness window: {self.tactile_max_age_s:.3f}s "
                    f"for sensors={self.tactile_sensor_order}"
                )
