"""Viser drawing for deploy_vigor2 (runs inside vigor2/viser_process.py, never in the control process).
The sim planner viz (motion_diffusion/planner/viz.py) on the robot: the measured robot, one trail per
body over the plan horizon (green), the commanded heading (red) and the heading reached (grey) as arrows
from the pelvis, the depth image, and the yaw slider. Trails are anchored on the measured bodies: each
body's trail is its real position now plus the plan's predicted displacement (FK of the predicted joints
at each frame minus FK at the plan's first frame). The planner's absolute first frame can sit decimetres
off the robot when the pose is outside its data (the mount), which would otherwise hide the motion."""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import torch
from core.common.viser_display import JOINT_NAMES_BY_MOTOR_INDEX, DeployViser

_FOLLOW, _GOAL, _DIM = (60, 255, 120), (255, 60, 60), (150, 150, 150)  # sim viz palette


def yaw_of_wxyz(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wxyz_from_rpy(roll: float, pitch: float, yaw: float):
    cr, sr, cp, sp, cy, sy = (math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2),
                              math.sin(pitch / 2), math.cos(yaw / 2), math.sin(yaw / 2))
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def _rot_from_wxyz(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=np.float32)


def ghost_pose(row: np.ndarray, slices: dict, yaw_now: float):
    """World pose of one predicted state: offsets are in the current heading frame (ref frame = newest history frame)."""
    h = float(row[slices["anchor_height"]][0])
    x, y = (float(v) for v in row[slices["anchor_xy_rel"]])
    yaw_rel = 0.0
    if "anchor_yaw_rel" in slices:
        c, s = (float(v) for v in row[slices["anchor_yaw_rel"]])
        yaw_rel = math.atan2(s, c)
    gx, gy, gz = (float(v) for v in row[slices["projected_gravity"]])
    pitch = math.asin(max(-1.0, min(1.0, gx)))
    roll = math.atan2(-gy, -gz)
    cy, sy = math.cos(yaw_now), math.sin(yaw_now)
    pos = (cy * x - sy * y, sy * x + cy * y, h)
    return pos, _wxyz_from_rpy(roll, pitch, yaw_now + yaw_rel)


class PlanDrawer:
    def __init__(self, urdf_path: str, depth_min: float, depth_max: float, depth_scale: float,
                 feature_layout: list, dof_names: list, fk_urdf_path: str, yaw_cmd_deg: float, on_cmd) -> None:
        from motion_diffusion.terrain.robot_mesh import build_pyroki_fk_model, load_fk_urdf_for_mesh
        self.base = DeployViser(urdf_path=Path(urdf_path), with_depth=True, depth_min=depth_min,
                                depth_max=depth_max, depth_scale=depth_scale)
        self.server = self.base.server
        self.slices = {r["name"]: slice(int(r["start"]), int(r["end"])) for r in feature_layout}
        self.dof_names = list(dof_names)
        urdf = load_fk_urdf_for_mesh(Path(fk_urdf_path))
        self.fk = build_pyroki_fk_model(urdf, self.dof_names)
        names = set(self.fk.link_names)
        bodies = ["pelvis"] + [j.child for j in urdf.joint_map.values() if j.type == "revolute" and j.child in names]
        self.body_idx = np.array([self.fk.link_index(b) for b in bodies], dtype=np.int64)
        self.motor_idx = np.array([JOINT_NAMES_BY_MOTOR_INDEX.index(n) for n in self.dof_names], dtype=np.int64)  # qj (motor order) -> dof order
        with self.server.gui.add_folder("Planner"):
            self.cmd_h = self.server.gui.add_slider("yaw command (deg)", min=-180.0, max=180.0, step=5.0, initial_value=round(yaw_cmd_deg))
            self.cmd_h.on_update(lambda _: on_cmd(math.radians(float(self.cmd_h.value))))
            self.prog_h = self.server.gui.add_number("yaw progress (deg)", 0.0, disabled=True)
            self.age_h = self.server.gui.add_number("plan age (ticks)", 0, disabled=True)
            self.lat_h = self.server.gui.add_number("plan latency (ms)", 0.0, disabled=True)
            self.held_h = self.server.gui.add_checkbox("stand held", False, disabled=True)

    def _body_trajectories(self, states: np.ndarray, yaw_now: float) -> np.ndarray:
        """(F, n_bodies, 3) world positions over the plan: FK of the predicted joints at each frame's predicted root pose."""
        dof = torch.as_tensor(states[:, self.slices["dof_pos"]], dtype=torch.float32)
        link_pos = np.asarray(self.fk.forward_link_pos_rot_xyzw(dof)[0])[:, self.body_idx]  # (F, n_bodies, 3) pelvis frame
        out = np.empty_like(link_pos)
        for k, row in enumerate(states):
            pos, q = ghost_pose(row, self.slices, yaw_now)
            out[k] = link_pos[k] @ _rot_from_wxyz(q).T + np.asarray(pos, dtype=np.float32)
        return out

    def update_robot(self, low_state, qj: np.ndarray, depth_raw, yaw_cmd_rad: float, start_yaw: float,
                     plan_age: int, latency_ms: float, held: bool) -> float:
        self.base.update(low_state, qj, depth_raw)
        yaw_now = yaw_of_wxyz(low_state.imu_state.quaternion)
        cmd_deg = math.degrees(yaw_cmd_rad)
        if abs(cmd_deg - float(self.cmd_h.value)) > 0.5:  # changed from the remote: reflect without re-triggering
            self.cmd_h.value = round(cmd_deg)
        self.prog_h.value = 0.0 if math.isnan(start_yaw) else round(math.degrees((yaw_now - start_yaw + math.pi) % (2 * math.pi) - math.pi), 1)
        self.age_h.value = int(plan_age)
        self.lat_h.value = round(float(latency_ms), 1)
        self.held_h.value = bool(held)
        if not math.isnan(start_yaw):  # arrows from the pelvis: commanded heading (red), heading reached (grey)
            for name, yaw, color in (("goal_command", start_yaw + yaw_cmd_rad, _GOAL), ("goal_progress", yaw_now, _DIM)):
                tip = np.array([[[0.0, 0.0, 0.0], [math.cos(yaw), math.sin(yaw), 0.0]]], dtype=np.float32)
                self.server.scene.add_line_segments(f"/plan/{name}", points=tip, colors=np.full((1, 2, 3), color, dtype=np.uint8), line_width=4.0)
        return yaw_now

    def update_plan(self, states: np.ndarray, yaw_now: float, qj: np.ndarray, imu_wxyz) -> None:
        """One trail per body across the plan's frames, anchored on the measured bodies."""
        pos = self._body_trajectories(states, yaw_now)  # (F, n_bodies, 3) predicted, absolute
        meas = torch.as_tensor(np.asarray(qj)[self.motor_idx], dtype=torch.float32).reshape(1, -1)
        now = np.asarray(self.fk.forward_link_pos_rot_xyzw(meas)[0])[0][self.body_idx] @ _rot_from_wxyz(imu_wxyz).T
        pos = pos - pos[0:1] + now[None]
        seg = np.stack([pos[:-1], pos[1:]], axis=2).reshape(-1, 2, 3)
        self.server.scene.add_line_segments("/plan/trails", points=seg, colors=np.full((len(seg), 2, 3), _FOLLOW, dtype=np.uint8), line_width=2.5)
