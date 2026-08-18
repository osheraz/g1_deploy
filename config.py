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


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(path: Path, seen=()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"cyclic config inheritance: {path}")
    with path.open("r") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader) or {}
    parent = config.pop("extends", None)
    if parent is None:
        return config
    parent_path = Path(parent).expanduser()
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_merge(_load_config(parent_path, seen + (path,)), config)


class Config:
    def __init__(self, file_path) -> None:
        config_path = _resolve_config_path(file_path)
        config_dir = config_path.parent

        config = _load_config(config_path)

        self.config_path = config_path
        self.config_dir = config_dir
        self.remote_root = REMOTE_ROOT

        self.viser_display = config.get("viser_display", False)
        self.control_dt = config["control_dt"]
        self.max_low_state_age_s = config.get("max_low_state_age_s", 0.1)
        self.max_default_hold_low_state_age_s = config.get(
            "max_default_hold_low_state_age_s", 0.5
        )
        if self.max_default_hold_low_state_age_s < self.max_low_state_age_s:
            raise ValueError(
                "max_default_hold_low_state_age_s must be greater than or "
                "equal to max_low_state_age_s"
            )
        self.max_joint_velocity = config.get("max_joint_velocity", 35.0)
        self.max_motor_temperature_c = config.get("max_motor_temperature_c", 75.0)
        self.max_pelvis_tilt_rad = config.get("max_pelvis_tilt_rad", 0.65)
        self.max_tracking_error_rad = config.get("max_tracking_error_rad", 0.45)
        self.max_command_rate_rad_s = config.get("max_command_rate_rad_s", 12.0)
        self.max_command_age_s = config.get("max_command_age_s", 0.10)
        self.validate_low_state_crc = config.get("validate_low_state_crc", True)
        self.msg_type = config["msg_type"]
        self.imu_type = config["imu_type"]

        self.weak_motor = config.get("weak_motor", [])

        self.lowcmd_topic = config["lowcmd_topic"]
        self.lowstate_topic = config["lowstate_topic"]

        # Relative to the repo root, not configs/: "policy/x.onnx" means
        # g1_deploy/policy/x.onnx. Absolute and ~ paths pass through.
        self.policy_path = _resolve_path(
            config["policy_path"], config_dir=REMOTE_ROOT
        )
        self.urdf_path = _resolve_path(
            config.get("urdf_path", DEFAULT_URDF_PATH), config_dir=config_dir
        )

        self.action_joint2motor_idx = config["action_joint2motor_idx"]
        self.kps = np.array(config["kps"])
        self.kds = np.array(config["kds"])
        self.default_angles = np.array(config["default_angles"], dtype=np.float32)
        physical_default = config.get("physical_default_angles", config.get("default_angles_23"))
        if physical_default is None:
            raise ValueError("config requires physical_default_angles (legacy: default_angles_23)")
        self.physical_default_angles = np.array(physical_default, dtype=np.float32)
        # Compatibility for the fall-recovery code. New SONIC code uses the
        # embodiment-neutral physical_default_angles name.
        self.default_angles_23 = self.physical_default_angles
        self.dof_lower = np.array(config["dof_lower"], dtype=np.float32)
        self.dof_upper = np.array(config["dof_upper"], dtype=np.float32)

        self.fixed_joint2motor_idx = config["fixed_joint2motor_idx"]
        self.fixed_kps = np.array(config["fixed_kps"])
        self.fixed_kds = np.array(config["fixed_kds"])
        self.fixed_target = np.array(config["fixed_target"], dtype=np.float32)

        self.ang_vel_scale = config["ang_vel_scale"]
        self.dof_pos_scale = config["dof_pos_scale"]
        self.dof_vel_scale = config["dof_vel_scale"]
        # scalar for the fall-recovery policy; per-joint (23) for SONIC
        _as = config["action_scale"]
        self.action_scale = (
            np.array(_as, dtype=np.float32) if isinstance(_as, list) else _as
        )
        self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)

        self.num_actions = config["num_actions"]
        self.num_obs = config["num_obs"]
        self.obs_dim = config["obs_dim"]

        # The training config sits next to the checkpoint. It is only needed by
        # PolicyRunner; SONIC and record mode never touch it, so a missing file
        # is not an error here -- loading the policy is what will complain.
        ckpt_path = Path(self.policy_path)
        self.train_cfg_path = ckpt_path.parent / "config.yaml"
        if self.train_cfg_path.exists():
            self.train_cfg = OmegaConf.load(self.train_cfg_path)
            self.actor_obs_keys = list(self.train_cfg.obs.obs_dict.actor_obs)
            obs_aux = self.train_cfg.obs.get("obs_auxiliary", {})
            self.history_actor_long_config = dict(obs_aux.get("history_actor_long", {}))
            self.student_actor_cfg = OmegaConf.select(
                self.train_cfg, "algo.config.student.module_dict.actor"
            )
        else:
            self.train_cfg = OmegaConf.create({})
            self.actor_obs_keys = []
            self.history_actor_long_config = {}
            self.student_actor_cfg = None

        self.action_ema = config.get("action_ema", 1.0)  # deploy action smoothing

        self.reference_motion = config["reference_motion"]
        self.reference_keyframes = config["reference_keyframes"]

        self.clip_obs = config["clip_obs"]
        self.clip_actions = config["clip_actions"]

        self.use_depth_obs = config["use_depth_obs"]

        self.policy_type = config.get("policy_type", "actor")
        self.sonic_encoder = (
            _resolve_path(config["sonic_encoder"], config_dir=REMOTE_ROOT)
            if config.get("sonic_encoder") else None
        )
        self.sonic_decoder = (
            _resolve_path(config["sonic_decoder"], config_dir=REMOTE_ROOT)
            if config.get("sonic_decoder") else None
        )
        self.sonic_observation_config = (
            _resolve_path(config["sonic_observation_config"], config_dir=REMOTE_ROOT)
            if config.get("sonic_observation_config") else None
        )
        self.sonic_model_variant = config.get("sonic_model_variant", "default")
        self.sonic_robot_profile = config.get(
            "sonic_robot_profile", f"g1_{int(config['num_actions'])}dof"
        )
        self.sonic_inference_backend = config.get("sonic_inference_backend", "auto")
        self.sonic_planner_replan_interval_s = float(
            config.get("sonic_planner_replan_interval_s", 0.1)
        )
        if not 0.1 <= self.sonic_planner_replan_interval_s <= 10.0:
            raise ValueError(
                "sonic_planner_replan_interval_s must be in [0.1, 10.0]"
            )
        self.sonic_planner_backend = config.get(
            "sonic_planner_backend", self.sonic_inference_backend
        )
        self.sonic_tensorrt_fp16 = config.get("sonic_tensorrt_fp16", False)
        self.sonic_tensorrt_tf32 = config.get("sonic_tensorrt_tf32", False)
        self.sonic_engine_cache_dir = _resolve_path(
            config.get("sonic_engine_cache_dir", "~/.cache/e_skin/sonic"),
            config_dir=REMOTE_ROOT,
        )
        self.sonic_mode = config.get("sonic_mode", "g1")
        self.sonic_planner_onnx = (
            _resolve_path(config["sonic_planner_onnx"], config_dir=REMOTE_ROOT)
            if config.get("sonic_planner_onnx") else None
        )
        self.sonic_planner_default_height = config.get(
            "sonic_planner_default_height", 0.788740
        )
        self.sonic_planner_version = config.get("sonic_planner_version", None)
        self.sonic_planner_random_seed = int(
            config.get("sonic_planner_random_seed", 1234)
        )
        if self.sonic_planner_random_seed < 0:
            raise ValueError("sonic_planner_random_seed must be non-negative")
        self.pico_server_ip = config.get("pico_server_ip", "127.0.0.1")
        self.pico_server_port = config.get("pico_server_port", 5556)
        self.pico_feedback_port = config.get("pico_feedback_port", 5558)
        self.sonic_motion_clip = _resolve_path(
            config.get("sonic_motion_clip", "policy/sonic/motion_clip.npz"),
            config_dir=REMOTE_ROOT,
        )
        self.sonic_motion_loop = config.get("sonic_motion_loop", True)
        self.sonic_mask_missing_reference = config.get(
            "sonic_mask_missing_reference",
            config.get("phantom_hold_ref_dof", False),
        )
        self.sonic_mask_missing_action_history = config.get(
            "sonic_mask_missing_action_history",
            False,
        )
        self.sonic_missing_adaptation_scope = config.get(
            "sonic_missing_adaptation_scope", "all"
        )
        self.sonic_adaptation_profile = config.get(
            "sonic_adaptation_profile", "official-parity"
        )
        self.sonic_reduced_proprioception = config.get(
            "sonic_reduced_proprioception", "zero"
        )
        self.sonic_shadow_settling_time_s = float(
            config.get("sonic_shadow_settling_time_s", 0.12)
        )
        # Neutral by default.  These physical-order vectors are populated by
        # research profiles or MuJoCo CLI overrides; official parity is direct.
        physical_dof = int(self.num_actions)
        self.sonic_motor_action_scale = np.asarray(
            config.get("sonic_motor_action_scale", [1.0] * physical_dof),
            dtype=np.float64,
        )
        self.sonic_motor_action_abs_limit = np.asarray(
            config.get("sonic_motor_action_abs_limit", [float("inf")] * physical_dof),
            dtype=np.float64,
        )
        self.sonic_motor_action_ema = np.asarray(
            config.get("sonic_motor_action_ema", [1.0] * physical_dof),
            dtype=np.float64,
        )
        self.sonic_motor_target_rate_limit = np.asarray(
            config.get("sonic_motor_target_rate_limit", [float("inf")] * physical_dof),
            dtype=np.float64,
        )
        self.sonic_action_history_source = str(
            config.get("sonic_action_history_source", "raw")
        ).strip().lower()
        # Encoder-facing gait amplitude. Neutral by default. Unlike motor
        # shaping, this asks SONIC to track a smaller reference while retaining
        # the tracker's full recovery authority at the output.
        self.sonic_reference_leg_scale = float(
            config.get("sonic_reference_leg_scale", 1.0)
        )
        self.sonic_reference_root_height_scale = float(
            config.get("sonic_reference_root_height_scale", 1.0)
        )
        self.sonic_reference_joint_scale = np.asarray(
            config.get("sonic_reference_joint_scale", [1.0] * physical_dof),
            dtype=np.float64,
        )
        # Compatibility only; embodiment selection is now explicit above.
        self.phantom_23dof = self.sonic_robot_profile == "g1_23dof"
        self.phantom_hold_ref_dof = self.sonic_mask_missing_reference

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
            self.depth_img_width = OmegaConf.select(
                self.train_cfg, "env.config.ego_camera.width")
            self.depth_img_height = OmegaConf.select(
                self.train_cfg, "env.config.ego_camera.height")
        else:
            self.depth_img_width = config.get("depth_img_width", 0)
            self.depth_img_height = config.get("depth_img_height", 0)

        self.depth_obs_dim = int(self.depth_img_width * self.depth_img_height)
        self.depth_server_ip = config.get("depth_server_ip", "192.168.123.164")
        self.depth_server_port = config.get("depth_server_port", 5555)

        self.depth_raw_height = config.get("depth_raw_height", 480)
        self.depth_raw_width = config.get("depth_raw_width", 640)

        print(f"Loaded config: {self.config_path}")
        print(
            "Policy path (base config; an explicit SONIC model profile may "
            f"replace it): {self.policy_path}"
        )
        if self.use_depth_obs:
            print(f"Depth: {self.depth_img_width}x{self.depth_img_height}")
            print(f"Depth server: {self.depth_server_ip}:{self.depth_server_port}")
        print(f"Use tactile: {self.use_tactile}")
        if self.use_tactile:
            print(f"Tactile sensors: {self.tactile_sensor_order}")
            print(
                f"Tactile server: {self.tactile_server_ip}:{self.tactile_server_port}"
            )

    def select_sonic_model_profile(self, profile: str) -> None:
        """Select one complete released encoder/decoder tensor contract.

        Model files and ``sonic_model_variant`` are intentionally changed as
        one unit. Mixing the 1247-D low-latency encoder with the original
        1762-D release contract is rejected later by ``SonicRunner`` as well.
        """
        if self.policy_type != "sonic":
            raise ValueError("SONIC model profiles require policy_type=sonic")
        from policy.sonic.deployment_profile import resolve_python_model_profile

        resolved = resolve_python_model_profile(profile)
        key = resolved["name"]
        variant = resolved["variant"]
        self.sonic_encoder = str(resolved["encoder"])
        self.sonic_decoder = str(resolved["decoder"])
        self.sonic_observation_config = str(resolved["observation_config"])
        self.policy_path = self.sonic_decoder
        self.sonic_model_variant = variant
        self.sonic_release_profile = key
        print(
            f"SONIC model profile: {key} "
            f"(variant={variant}, encoder={self.sonic_encoder}, "
            f"decoder={self.sonic_decoder}, "
            f"observation={self.sonic_observation_config})"
        )
