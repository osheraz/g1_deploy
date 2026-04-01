from __future__ import annotations

from typing import Optional
from pathlib import Path

import numpy as np
import math
import torch
import onnxruntime as ort

from common.depth_client import depth_to_obs_vector_match_sim
from common.history import HistoryHandler
from config import Config


class PolicyRunner:
    def __init__(
        self,
        config: Config,
    ):
        self.config = config

        self.act_count = 0
        self.policy = None
        self.policy_type = None  # "jit" or "onnx" or "torch"
        self.ort_sess = None
        self.onnx_input_names = []

        self.action = np.zeros(config.num_actions, dtype=np.float32)

        self.history_config_long = config.history_actor_long_config
        self.history_handler = HistoryHandler(
            history_config=self.history_config_long,
            obs_dims=config.obs_dim,
        )

        obs_aux = self.config.train_cfg.obs.obs_auxiliary

        if "depth_history" in obs_aux and "depth" in obs_aux.depth_history:
            self.depth_T = obs_aux.depth_history.depth
        else:
            self.depth_T = 0

        H = self.config.depth_img_height
        W = self.config.depth_img_width
        self.depth_buf = np.zeros((1, self.depth_T, H * W), np.float32)

        self._load_policy()

    # -------- policy loading --------

    def _load_policy(self):
        path = Path(self.config.policy_path)

        if not path.exists():
            print(f"Policy path does not exist: {path}")
            return

        suffix = path.suffix.lower()

        if suffix == ".onnx":
            self.ort_sess = ort.InferenceSession(str(path))
            self.policy_type = "onnx"
            self.onnx_input_names = [i.name for i in self.ort_sess.get_inputs()]
            print(f"\nLoaded ONNX policy from {path}\n")
            print(f"\nonnx_input_names {self.onnx_input_names }\n")

            for inp in self.ort_sess.get_inputs():
                print(inp.name, inp.shape)
            return

        # --- Try JIT first ---
        try:
            self.policy = torch.jit.load(str(path), map_location="cpu")
            self.policy.eval()
            self.policy_type = "jit"
            print(f"\nLoaded JIT policy from {path}\n")
            return
        except Exception:
            pass

        try:
            from policy.student_model import DeployStudentPolicy

            self.device = "cuda"
            self.policy = DeployStudentPolicy(
                path, self.config, device=self.device
            ).student
            self.policy_type = "torch"
            return

        except Exception as e:
            print(f"Failed to load policy: {e}")
            self.policy = None
            self.policy_type = None

        print(self.policy_type)

    # -------- depth + history helpers --------

    def _build_depth_vec_from_raw(self, depth_raw) -> Optional[np.ndarray]:
        if not self.config.use_depth_obs:
            return None
        assert depth_raw is not None, "Depth is required but depth_raw is None"
        return depth_to_obs_vector_match_sim(
            depth_raw=depth_raw,
            out_h=self.config.depth_img_height,
            out_w=self.config.depth_img_width,
            min_d=self.config.min_d,
            max_d=self.config.max_d,
        ).astype(np.float32)

    def _update_history_long(
        self,
        base_ang_vel: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ):
        hc = self.history_config_long

        values = {
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "actions": self.action,
        }

        for key, arr in values.items():
            if key in hc:
                # features already scaled in RobotInterface.get_base_features_scaled
                v = arr.reshape(1, -1).astype(np.float32)
                self.history_handler.add(key, v)

    def _build_student_hist_obs_long(self) -> np.ndarray:
        history_config = self.history_config_long
        per_key_hist = []
        T = None

        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            hist_full = self.history_handler.query(key)
            hist = hist_full[:, :history_length]
            if T is None:
                T = hist.shape[1]
            else:
                assert (
                    hist.shape[1] == T
                ), f"History length mismatch for '{key}': {hist.shape[1]} vs {T}"
            per_key_hist.append(hist)

        hist_all = np.concatenate(per_key_hist, axis=-1)
        hist_flat = hist_all.reshape(1, -1).astype(np.float32)
        return hist_flat

    # -------- obs building --------
    def _build_onnx_inputs(
        self,
        actor_feat: np.ndarray,
        hist_flat: np.ndarray,
        depth_vec: Optional[np.ndarray],
    ) -> dict:
        inputs = {}
        names = self.onnx_input_names

        if "actor_obs" in names:
            inputs["actor_obs"] = actor_feat[np.newaxis, :].astype(np.float32)

        if "student_hist_obs_long" in names:
            inputs["student_hist_obs_long"] = hist_flat.astype(np.float32)

        if "perception_obs" in names:
            assert depth_vec is not None
            inputs["perception_obs"] = depth_vec[np.newaxis, :].astype(np.float32)

        if self.depth_T and "perception_obs_history_obs" in names:
            inputs["perception_obs_history_obs"] = self.depth_buf.reshape(1, -1).astype(
                np.float32
            )

        return inputs

    def compute_action(
        self,
        base_ang_vel: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        depth_raw: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        if self.policy_type is None:
            return np.zeros_like(self.action)

        values = {
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "actions": self.action,
        }

        keys = sorted(values.keys())
        actor_obs = np.concatenate([values[k] for k in keys], axis=-1).astype(
            np.float32
        )

        self._update_history_long(
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
        )

        hist_flat = self._build_student_hist_obs_long()

        depth_vec = (
            self._build_depth_vec_from_raw(depth_raw)
            if self.config.use_depth_obs
            else None
        )
        use_depth_hist = bool(self.depth_T) and (depth_vec is not None)

        # if use_depth_hist:
        #     self.depth_buf[:, 1:] = self.depth_buf[:, :-1]
        #     self.depth_buf[:, 0] = depth_vec.reshape(1, -1)

        if self.policy_type == "jit":
            with torch.no_grad():
                act = self.policy(
                    torch.from_numpy(actor_obs).unsqueeze(0).float(),
                    torch.from_numpy(depth_vec).unsqueeze(0).float(),
                    torch.from_numpy(hist_flat).float(),
                )
                act = act.cpu().numpy().squeeze()

        elif self.policy_type == "torch":
            if not self.depth_T:
                obs_dict = {
                    "actor_obs": torch.from_numpy(actor_obs)
                    .unsqueeze(0)
                    .float()
                    .to(self.device),
                    "perception_obs": (
                        torch.from_numpy(depth_vec).unsqueeze(0).float().to(self.device)
                        if depth_vec is not None
                        else torch.zeros(
                            1,
                            self.config.depth_img_height * self.config.depth_img_width,
                        )
                        .float()
                        .to(self.device)
                    ),
                    "student_hist_obs_long": torch.from_numpy(hist_flat)
                    .float()
                    .to(self.device),
                }
            else:
                obs_dict = {
                    "actor_obs": torch.from_numpy(actor_obs)
                    .unsqueeze(0)
                    .float()
                    .to(self.device),
                    "perception_obs": torch.from_numpy(self.depth_buf[:, 0])
                    .float()
                    .to(self.device),
                    "perception_obs_history_obs": torch.from_numpy(
                        self.depth_buf.reshape(1, -1)
                    )
                    .float()
                    .to(self.device),
                    "student_hist_obs_long": torch.from_numpy(hist_flat)
                    .float()
                    .to(self.device),
                }

            with torch.no_grad():
                act = self.policy.act_inference(obs_dict)
                act = act.cpu().numpy().squeeze()

        elif self.policy_type == "onnx":
            inputs = self._build_onnx_inputs(actor_obs, hist_flat, depth_vec)
            outputs = self.ort_sess.run(None, inputs)
            act = outputs[0].squeeze()

        else:
            return np.zeros_like(self.action)

        # update depth history *after* using it, so history is lagged (no current frame)
        if use_depth_hist:
            self.depth_buf[:, 1:] = self.depth_buf[:, :-1]
            self.depth_buf[:, 0] = depth_vec.reshape(1, -1)

        act = np.clip(act, -self.config.clip_actions, self.config.clip_actions)
        # print(act)
        # Smooth
        # num_warmup_steps = 50
        # if self.act_count < num_warmup_steps:
        #     e = min(self.act_count / num_warmup_steps, 1)
        #     self.action = (e) * act.astype(np.float32)
        #     # self.action = (1- e) * self.action + (e) * act.astype(np.float32)
        # else:
        self.action = act.astype(np.float32)

        self.act_count += 1

        return self.action
