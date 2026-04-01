from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np
import viser
from viser.extras import ViserUrdf

fixed_joint2motor_idx = []

JOINT_NAMES_BY_MOTOR_INDEX = [
    "left_hip_pitch_joint",  # 0
    "left_hip_roll_joint",  # 1
    "left_hip_yaw_joint",  # 2
    "left_knee_joint",  # 3
    "left_ankle_pitch_joint",  # 4
    "left_ankle_roll_joint",  # 5
    "right_hip_pitch_joint",  # 6
    "right_hip_roll_joint",  # 7
    "right_hip_yaw_joint",  # 8
    "right_knee_joint",  # 9
    "right_ankle_pitch_joint",  # 10
    "right_ankle_roll_joint",  # 11
    "waist_yaw_joint",  # 12
    "left_shoulder_pitch_joint",  # 13
    "left_shoulder_roll_joint",  # 14
    "left_shoulder_yaw_joint",  # 15
    "left_elbow_joint",  # 16
    "left_wrist_roll_joint",  # 17
    "right_shoulder_pitch_joint",  # 18
    "right_shoulder_roll_joint",  # 19
    "right_shoulder_yaw_joint",  # 20
    "right_elbow_joint",  # 21
    "right_wrist_roll_joint",  # 22
]


class DeployViser:
    def __init__(
        self,
        urdf_path: str,
        with_depth: bool,
        depth_min: float,
        depth_max: float,
        depth_scale: float,
    ) -> None:
        self.server = viser.ViserServer()
        self.joint_names_by_motor_index = list(JOINT_NAMES_BY_MOTOR_INDEX)
        self.with_depth = with_depth
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.depth_scale = float(depth_scale)

        self.base_frame = self.server.scene.add_frame(
            "/base",
            position=(0.0, 0.0, 0.0),
            show_axes=False,
        )

        self.urdf_vis = ViserUrdf(
            self.server,
            urdf_or_path=urdf_path,
            root_node_name="/base",
            load_meshes=True,
            load_collision_meshes=False,
        )

        self.server.scene.add_grid(
            "/grid",
            width=2.0,
            height=2.0,
            position=(0.0, 0.0, -0.5),
        )

        state_folder = self.server.gui.add_folder("Robot State")
        with state_folder:
            self.tick_handle = self.server.gui.add_number("Tick", 0, disabled=True)
            self.mode_pr_handle = self.server.gui.add_number(
                "mode_pr", 0, disabled=True
            )
            self.mode_machine_handle = self.server.gui.add_number(
                "mode_machine", 0, disabled=True
            )
            self.imu_w_handle = self.server.gui.add_number("imu_w", 1.0, disabled=True)
            self.imu_x_handle = self.server.gui.add_number("imu_x", 0.0, disabled=True)
            self.imu_y_handle = self.server.gui.add_number("imu_y", 0.0, disabled=True)
            self.imu_z_handle = self.server.gui.add_number("imu_z", 0.0, disabled=True)

        self.depth_image_handle = None
        if self.with_depth:
            depth_folder = self.server.gui.add_folder("Depth")
            with depth_folder:
                dummy = np.zeros((120, 160, 3), dtype=np.uint8)
                self.depth_image_handle = self.server.gui.add_image(
                    dummy, label="Depth view"
                )
                self.depth_min_gui = self.server.gui.add_number(
                    "Depth min (m)", self.depth_min
                )
                self.depth_max_gui = self.server.gui.add_number(
                    "Depth max (m)", self.depth_max
                )

        self.rgb_image_handle = None
        rgb_folder = self.server.gui.add_folder("External RGB")
        with rgb_folder:
            dummy = np.zeros((120, 160, 3), dtype=np.uint8)
            self.rgb_image_handle = self.server.gui.add_image(dummy, label="RGB")

        vis_folder = self.server.gui.add_folder("Visibility")
        with vis_folder:
            show_meshes_cb = self.server.gui.add_checkbox(
                "Show meshes", self.urdf_vis.show_visual
            )
            show_collision_meshes_cb = self.server.gui.add_checkbox(
                "Show collision meshes", self.urdf_vis.show_collision
            )

        @show_meshes_cb.on_update
        def _(_evt) -> None:
            self.urdf_vis.show_visual = show_meshes_cb.value

        @show_collision_meshes_cb.on_update
        def _(_evt) -> None:
            self.urdf_vis.show_collision = show_collision_meshes_cb.value

    def update(
        self,
        low_state,
        qj: np.ndarray,
        depth_raw: Optional[np.ndarray] = None,
        rgb_bgr: Optional[np.ndarray] = None,
    ) -> None:
        # Tick & modes
        self.tick_handle.value = int(low_state.tick)
        self.mode_pr_handle.value = int(low_state.mode_pr)
        self.mode_machine_handle.value = int(low_state.mode_machine)

        # IMU
        imu = low_state.imu_state
        wxyz = tuple(imu.quaternion)
        self.base_frame.wxyz = wxyz
        self.imu_w_handle.value = float(wxyz[0])
        self.imu_x_handle.value = float(wxyz[1])
        self.imu_y_handle.value = float(wxyz[2])
        self.imu_z_handle.value = float(wxyz[3])

        joint_cfg = {
            name: float(qj[i])
            for i, name in enumerate(JOINT_NAMES_BY_MOTOR_INDEX)
            if i not in fixed_joint2motor_idx
        }

        self.urdf_vis.update_cfg(joint_cfg)

        # Depth update (if enabled)
        if (
            self.with_depth
            and self.depth_image_handle is not None
            and depth_raw is not None
        ):
            self._update_depth_image(depth_raw)

        if self.rgb_image_handle is not None and rgb_bgr is not None:
            self.rgb_image_handle.image = rgb_bgr[..., ::-1]

    def _update_depth_image(self, depth_raw: np.ndarray) -> None:
        d = np.asarray(depth_raw)
        if d.dtype == np.uint16 or d.dtype == np.int16:
            d = d.astype(np.float32) * self.depth_scale
        else:
            d = d.astype(np.float32)

        finite = np.isfinite(d)
        if not finite.any():
            H, W = d.shape[:2]
            img = np.zeros((H, W, 3), dtype=np.uint8)
            self.depth_image_handle.image = img
            return

        vmin = float(self.depth_min_gui.value)
        vmax = float(self.depth_max_gui.value)
        if vmax <= vmin:
            vmax = vmin + 1e-3

        t = np.clip((d - vmin) / (vmax - vmin), 0.0, 1.0)
        g = (t * 255.0).astype(np.uint8)
        img = np.stack([g, g, g], axis=-1)
        self.depth_image_handle.image = img
