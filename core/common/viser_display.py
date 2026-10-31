from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np

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
        tactile_sensors: Optional[list[str]] = None,
        joint_names: Optional[list[str]] = None,
        port: int = 8080,
        replay_layout: bool = False,
        robot_visible: bool = True,
    ) -> None:
        import viser
        from viser.extras import ViserUrdf

        self.server = viser.ViserServer(port=port)
        self.joint_names_by_motor_index = list(
            JOINT_NAMES_BY_MOTOR_INDEX if joint_names is None else joint_names
        )
        self.with_depth = with_depth
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.depth_scale = float(depth_scale)
        self.tactile_sensors = set(tactile_sensors or [])

        self.base_frame = self.server.scene.add_frame(
            "/base",
            position=(0.0, 0.0, 0.0),
            show_axes=False,
            visible=robot_visible,
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

        state_folder = self.server.gui.add_folder("Robot State",expand_by_default=not replay_layout)
        self.state_folder=state_folder
        state_folder.visible=robot_visible
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
            depth_folder = self.server.gui.add_folder("Depth",order=-10 if replay_layout else None)
            self.depth_folder=depth_folder
            with depth_folder:
                dummy = np.zeros((120, 160, 3), dtype=np.uint8)
                self.depth_image_handle = self.server.gui.add_image(
                    dummy, label="Depth view"
                )
                self.depth_min_gui = self.server.gui.add_number(
                    "Depth min (m)", self.depth_min
                )
                self.depth_max_gui = self.server.gui.add_number(
                    "Depth display max (m)", self.depth_max
                )

        self.rgb_image_handle = None
        rgb_folder = self.server.gui.add_folder("External RGB",order=-20 if replay_layout else None)
        with rgb_folder:
            dummy = np.zeros((120, 160, 3), dtype=np.uint8)
            self.rgb_image_handle = self.server.gui.add_image(dummy, label="RGB")

        tactile_folder = self.server.gui.add_folder("Tactile",expand_by_default=not replay_layout)
        if replay_layout and not self.tactile_sensors:tactile_folder.visible=False
        with tactile_folder:
            self.tactile_overall_handle = self.server.gui.add_text(
                "Status", initial_value="disabled", disabled=True
            )
            self.tactile_lf_handle = None
            self.tactile_rf_handle = None
            self.tactile_wb_handle = None
            self.tactile_lh_handle = None
            self.tactile_rh_handle = None
            self.tactile_lf_image = None
            self.tactile_rf_image = None
            self.tactile_lh_image = None
            self.tactile_rh_image = None
            self.tactile_wb_front_image = None
            self.tactile_wb_back_image = None
            self.tactile_wb_left_shoulder_image = None
            self.tactile_wb_right_shoulder_image = None
            self.tactile_wb_left_arm_image = None
            self.tactile_wb_right_arm_image = None
            dummy_foot = np.zeros((96, 160, 3), dtype=np.uint8)
            dummy_hand = np.zeros((156, 180, 3), dtype=np.uint8)
            dummy_arm = np.zeros((96, 96, 3), dtype=np.uint8)
            dummy_shoulder = np.zeros((32, 96, 3), dtype=np.uint8)
            if "LF" in self.tactile_sensors:
                self.tactile_lf_handle = self.server.gui.add_text(
                    "LF", initial_value="-", disabled=True
                )
                self.tactile_lf_image = self.server.gui.add_image(
                    dummy_foot, label="LF map"
                )
            if "RF" in self.tactile_sensors:
                self.tactile_rf_handle = self.server.gui.add_text(
                    "RF", initial_value="-", disabled=True
                )
                self.tactile_rf_image = self.server.gui.add_image(
                    dummy_foot, label="RF map"
                )
            if "LH" in self.tactile_sensors:
                self.tactile_lh_handle = self.server.gui.add_text(
                    "LH", initial_value="-", disabled=True
                )
                self.tactile_lh_image = self.server.gui.add_image(
                    dummy_hand, label="LH glove"
                )
            if "RH" in self.tactile_sensors:
                self.tactile_rh_handle = self.server.gui.add_text(
                    "RH", initial_value="-", disabled=True
                )
                self.tactile_rh_image = self.server.gui.add_image(
                    dummy_hand, label="RH glove"
                )
            if "WB" in self.tactile_sensors:
                self.tactile_wb_handle = self.server.gui.add_text(
                    "WB", initial_value="-", disabled=True
                )
                self.tactile_wb_front_image = self.server.gui.add_image(
                    dummy_foot, label="WB front"
                )
                self.tactile_wb_back_image = self.server.gui.add_image(
                    dummy_foot, label="WB back"
                )
                self.tactile_wb_left_shoulder_image = self.server.gui.add_image(
                    dummy_shoulder, label="WB left shoulder"
                )
                self.tactile_wb_right_shoulder_image = self.server.gui.add_image(
                    dummy_shoulder, label="WB right shoulder"
                )
                self.tactile_wb_left_arm_image = self.server.gui.add_image(
                    dummy_arm, label="WB left arm"
                )
                self.tactile_wb_right_arm_image = self.server.gui.add_image(
                    dummy_arm, label="WB right arm"
                )

        vis_folder = self.server.gui.add_folder("Visibility",expand_by_default=not replay_layout)
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
            for i, name in enumerate(self.joint_names_by_motor_index)
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
            img = np.full((H, W, 3),(100,0,130),dtype=np.uint8)
            self.depth_image_handle.image = img
            return

        vmin = float(self.depth_min_gui.value)
        vmax = float(self.depth_max_gui.value)
        if vmax <= vmin:
            vmax = vmin + 1e-3

        t = np.clip((d - vmin) / (vmax - vmin), 0.0, 1.0)
        g = (np.nan_to_num(t) * 255.0).astype(np.uint8)
        img = np.stack([g, g, g], axis=-1)
        img[~finite]=(100,0,130)
        self.depth_image_handle.image = img

    def update_tactile(self, overall: str, sensor_status: dict[str, str]) -> None:
        self.tactile_overall_handle.value = overall
        if self.tactile_lf_handle is not None:
            self.tactile_lf_handle.value = sensor_status.get("LF", "-")
        if self.tactile_rf_handle is not None:
            self.tactile_rf_handle.value = sensor_status.get("RF", "-")
        if self.tactile_wb_handle is not None:
            self.tactile_wb_handle.value = sensor_status.get("WB", "-")
        if self.tactile_lh_handle is not None:
            self.tactile_lh_handle.value = sensor_status.get("LH", "-")
        if self.tactile_rh_handle is not None:
            self.tactile_rh_handle.value = sensor_status.get("RH", "-")

    def update_tactile_images(self, panels: dict[str, np.ndarray]) -> None:
        if self.tactile_lf_image is not None and "LF" in panels:
            self.tactile_lf_image.image = panels["LF"]
        if self.tactile_rf_image is not None and "RF" in panels:
            self.tactile_rf_image.image = panels["RF"]
        if self.tactile_lh_image is not None and "LH" in panels:
            self.tactile_lh_image.image = panels["LH"]
        if self.tactile_rh_image is not None and "RH" in panels:
            self.tactile_rh_image.image = panels["RH"]
        if self.tactile_wb_front_image is not None and "WB_front" in panels:
            self.tactile_wb_front_image.image = panels["WB_front"]
        if self.tactile_wb_back_image is not None and "WB_back" in panels:
            self.tactile_wb_back_image.image = panels["WB_back"]
        if self.tactile_wb_left_shoulder_image is not None and "WB_left_shoulder" in panels:
            self.tactile_wb_left_shoulder_image.image = panels["WB_left_shoulder"]
        if self.tactile_wb_right_shoulder_image is not None and "WB_right_shoulder" in panels:
            self.tactile_wb_right_shoulder_image.image = panels["WB_right_shoulder"]
        if self.tactile_wb_left_arm_image is not None and "WB_left_arm" in panels:
            self.tactile_wb_left_arm_image.image = panels["WB_left_arm"]
        if self.tactile_wb_right_arm_image is not None and "WB_right_arm" in panels:
            self.tactile_wb_right_arm_image.image = panels["WB_right_arm"]
