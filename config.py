from pathlib import Path

import numpy as np
import yaml
from omegaconf import OmegaConf


REMOTE_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = REMOTE_ROOT / "configs"

# Home-relative so it resolves for any user; override with `urdf_path:` in the config yaml.
DEFAULT_URDF_PATH = (
    "~/HumanoidVerse2/humanoidverse/data/robots/g1/g1_23dof_rev_1_0.urdf"
)

OmegaConf.register_new_resolver("eval", lambda x: eval(x))


def _resolve_config_path(file_path: str | Path) -> Path:
    path = Path(file_path).expanduser()
    if path.is_absolute():
        return path
    return (CONFIGS_DIR / path).resolve()


def _resolve_path(value: str, *, config_dir: Path) -> str:
    expanded = (
        value.replace("{REMOTE_ROOT}", str(REMOTE_ROOT))
        .replace("{CONFIG_DIR}", str(config_dir))
    )
    path = Path(expanded).expanduser()
    if path.is_absolute():
        return str(path)
    return str((config_dir / path).resolve())


class Config:
    def __init__(self, file_path) -> None:
        config_path = _resolve_config_path(file_path)
        config_dir = config_path.parent

        with config_path.open("r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.config_path = config_path
        self.config_dir = config_dir
        self.remote_root = REMOTE_ROOT

        self.viser_display = config.get("viser_display", False)
        self.control_dt = config["control_dt"]
        self.msg_type = config["msg_type"]
        self.imu_type = config["imu_type"]

        self.weak_motor = config.get("weak_motor", [])

        self.lowcmd_topic = config["lowcmd_topic"]
        self.lowstate_topic = config["lowstate_topic"]

        self.policy_path = _resolve_path(config["policy_path"], config_dir=config_dir)
        self.urdf_path = _resolve_path(
            config.get("urdf_path", DEFAULT_URDF_PATH), config_dir=config_dir
        )

        self.action_joint2motor_idx = config["action_joint2motor_idx"]
        self.kps = np.array(config["kps"])
        self.kds = np.array(config["kds"])
        self.default_angles = np.array(config["default_angles"], dtype=np.float32)
        self.default_angles_23 = np.array(config["default_angles_23"], dtype=np.float32)

        self.fixed_joint2motor_idx = config["fixed_joint2motor_idx"]
        self.fixed_kps = np.array(config["fixed_kps"])
        self.fixed_kds = np.array(config["fixed_kds"])
        self.fixed_target = np.array(config["fixed_target"], dtype=np.float32)

        self.ang_vel_scale = config["ang_vel_scale"]
        self.dof_pos_scale = config["dof_pos_scale"]
        self.dof_vel_scale = config["dof_vel_scale"]
        self.action_scale = config["action_scale"]
        self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)

        self.num_actions = config["num_actions"]
        self.num_obs = config["num_obs"]
        self.obs_dim = config["obs_dim"]

        ckpt_path = Path(self.policy_path)
        self.train_cfg_path = ckpt_path.parent / "config.yaml"
        self.train_cfg = OmegaConf.load(self.train_cfg_path)

        self.actor_obs_keys = list(self.train_cfg.obs.obs_dict.actor_obs)
        obs_aux = self.train_cfg.obs.get("obs_auxiliary", {})
        self.history_actor_long_config = dict(
            obs_aux.get("history_actor_long", {})
        )
        self.student_actor_cfg = OmegaConf.select(
            self.train_cfg, "algo.config.student.module_dict.actor"
        )

        # reference-tracking: replay clip (.npz on control-dt grid) -> proto obs
        self.reference_clip = config.get("reference_clip", None)
        self.future_steps = list(
            config.get(
                "future_steps",
                OmegaConf.select(
                    self.train_cfg, "env.config.reference_motion.future_steps"
                )
                or [1, 2, 4, 8],
            )
        )
        self.loop_mode = config.get("loop_mode", "clamp")
        self.action_ema = config.get("action_ema", 1.0)  # deploy action smoothing

        # live PICO 3-point teleop (sparse keypoint goal) instead of a clip
        self.pico_teleop = config.get("pico_teleop", False)
        self.pico_tracked_order = list(
            config.get("pico_tracked_order", ["head", "left_wrist", "right_wrist"])
        )
        self.pico_emit = list(
            config.get(
                "pico_emit",
                ["motion_anchor_pos_b", "motion_anchor_ori_b",
                 "ref_body_pos_b", "ref_body_ori_b"],
            )
        )

        self.reference_motion = config["reference_motion"]
        self.reference_keyframes = config["reference_keyframes"]

        self.clip_obs = config["clip_obs"]
        self.clip_actions = config["clip_actions"]

        self.use_depth_obs = config["use_depth_obs"]

        self.use_tactile = config.get("use_tactile", False)
        self.tactile_sensor_order = config.get(
            "tactile_sensor_order",
            ["LF", "RF", "WB", "LH", "RH"],
        )

        self.tactile_server_ip = config.get(
            "tactile_server_ip", config.get("depth_server_ip", "192.168.123.164")
        )
        self.tactile_server_port = config.get("tactile_server_port", 5557)

        self.min_d = config.get("min_d", 0.1)
        self.max_d = config.get("max_d", 2.0)
        self.depth_scale = config.get("depth_scale", 0.001)

        if self.use_depth_obs:
            self.depth_img_width = self.train_cfg.env.config.ego_camera.width
            self.depth_img_height = self.train_cfg.env.config.ego_camera.height
        else:
            self.depth_img_width = config.get("depth_img_width", 0)
            self.depth_img_height = config.get("depth_img_height", 0)

        self.depth_obs_dim = int(self.depth_img_width * self.depth_img_height)
        self.depth_server_ip = config.get("depth_server_ip", "192.168.123.164")
        self.depth_server_port = config.get("depth_server_port", 5555)

        self.depth_raw_height = config.get("depth_raw_height", 480)
        self.depth_raw_width = config.get("depth_raw_width", 640)

        print(f"Loaded config: {self.config_path}")
        print(f"Policy path: {self.policy_path}")
        if self.use_depth_obs:
            print(f"Depth: {self.depth_img_width}x{self.depth_img_height}")
            print(f"Depth server: {self.depth_server_ip}:{self.depth_server_port}")
        print(f"Use tactile: {self.use_tactile}")
        if self.use_tactile:
            print(f"Tactile sensors: {self.tactile_sensor_order}")
            print(
                f"Tactile server: {self.tactile_server_ip}:{self.tactile_server_port}"
            )
