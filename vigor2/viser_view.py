"""Viser drawing for deploy_vigor2 (runs inside vigor2/viser_process.py, never in the control process):
the measured robot (IMU + joints), the depth image, the planner's predicted motion as stick figures from
the observed robot (FK of the predicted joints at several horizon frames, green now -> red +0.5 s), the
predicted pelvis path, and the yaw-command slider."""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import torch
from core.common.viser_display import DeployViser

SKELETON_FRAMES = (0, 6, 12, 18, 24)  # ticks ahead (20 ms each)


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
        self.edges = np.array([(self.fk.link_index(j.parent), self.fk.link_index(j.child))
                               for j in urdf.joint_map.values()
                               if j.type == "revolute" and j.parent in names and j.child in names], dtype=np.int64)
        with self.server.gui.add_folder("Planner"):
            self.cmd_h = self.server.gui.add_slider("yaw command (deg)", min=-180.0, max=180.0, step=5.0, initial_value=round(yaw_cmd_deg))
            self.cmd_h.on_update(lambda _: on_cmd(math.radians(float(self.cmd_h.value))))
            self.prog_h = self.server.gui.add_number("yaw progress (deg)", 0.0, disabled=True)
            self.age_h = self.server.gui.add_number("plan age (ticks)", 0, disabled=True)
            self.lat_h = self.server.gui.add_number("plan latency (ms)", 0.0, disabled=True)
            self.held_h = self.server.gui.add_checkbox("stand held", False, disabled=True)

    def _skeleton(self, row: np.ndarray, yaw_now: float) -> np.ndarray:
        """(n_edges, 2, 3) world segments of one predicted frame: FK of the predicted joints at the predicted root pose."""
        pos, q = ghost_pose(row, self.slices, yaw_now)
        dof = torch.as_tensor(row[self.slices["dof_pos"]], dtype=torch.float32).reshape(1, -1)
        link_pos = np.asarray(self.fk.forward_link_pos_rot_xyzw(dof)[0])[0]  # (n_links, 3) in the pelvis frame
        world = link_pos @ _rot_from_wxyz(q).T + np.asarray(pos, dtype=np.float32)
        return world[self.edges]

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
        return yaw_now

    def update_plan(self, states: np.ndarray, yaw_now: float) -> None:
        segs, cols = [], []
        for i, k in enumerate(SKELETON_FRAMES):
            s = self._skeleton(states[min(k, len(states) - 1)], yaw_now)
            segs.append(s)
            t = i / max(1, len(SKELETON_FRAMES) - 1)
            cols.append(np.tile(np.array([255 * t, 200 * (1 - t), 0], dtype=np.uint8), (len(s), 2, 1)))
        self.server.scene.add_line_segments("/plan/skeleton", points=np.concatenate(segs), colors=np.concatenate(cols), line_width=3.0)
        path = np.array([ghost_pose(r, self.slices, yaw_now)[0] for r in states], dtype=np.float32)
        self.server.scene.add_line_segments("/plan/path", points=np.stack([path[:-1], path[1:]], axis=1),
                                            colors=np.full((len(path) - 1, 2, 3), (255, 255, 255), dtype=np.uint8), line_width=2.0)
