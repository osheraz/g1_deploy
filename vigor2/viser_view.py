"""Viser view for deploy_vigor2: the measured robot (IMU + joints), the depth image, and the planner's
predicted motion drawn from the observed robot as stick figures (URDF joint tree through FK) at several
horizon frames, green (now) -> red (+0.5 s), with the predicted pelvis path."""
from __future__ import annotations
import math
import threading
import time
from pathlib import Path

import numpy as np
import torch
from core.common.viser_display import DeployViser

SKELETON_FRAMES = (0, 6, 12, 18, 24)  # ticks ahead (20 ms each)


def _yaw_of_wxyz(q) -> float:
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


class Vigor2Viser:
    def __init__(self, urdf_path: str, depth_min: float, depth_max: float, depth_scale: float, runner) -> None:
        from motion_diffusion.terrain.robot_mesh import load_fk_urdf_for_mesh
        self.runner = runner
        self.base = DeployViser(urdf_path=Path(urdf_path), with_depth=True, depth_min=depth_min,
                                depth_max=depth_max, depth_scale=depth_scale)
        self.server = self.base.server
        bundle = runner.planner.bundle
        self.slices = {r["name"]: slice(int(r["start"]), int(r["end"])) for r in bundle.feature_layout}
        self.fk = runner.state.fk
        urdf = load_fk_urdf_for_mesh(Path(runner.fk_urdf_path))
        names = set(self.fk.link_names)
        self.edges = np.array([(self.fk.link_index(j.parent), self.fk.link_index(j.child))
                               for j in urdf.joint_map.values()
                               if j.type == "revolute" and j.parent in names and j.child in names], dtype=np.int64)
        with self.server.gui.add_folder("Planner"):
            self.cmd_h = self.server.gui.add_slider("yaw command (deg)", min=-180.0, max=180.0, step=5.0,
                                                    initial_value=round(math.degrees(runner.yaw_command_rad)))
            self.cmd_h.on_update(lambda _: setattr(runner, "yaw_command_rad", math.radians(float(self.cmd_h.value))))
            self.prog_h = self.server.gui.add_number("yaw progress (deg)", 0.0, disabled=True)
            self.age_h = self.server.gui.add_number("plan age (ticks)", 0, disabled=True)
            self.lat_h = self.server.gui.add_number("plan latency (ms)", 0.0, disabled=True)
            self.held_h = self.server.gui.add_checkbox("stand held", False, disabled=True)

    def _skeleton(self, row: np.ndarray, yaw_now: float) -> np.ndarray:
        """(n_edges, 2, 3) world segments of one predicted frame: FK of the predicted joints, placed at the predicted root pose."""
        pos, q = ghost_pose(row, self.slices, yaw_now)
        dof = torch.as_tensor(row[self.slices["dof_pos"]], dtype=torch.float32).reshape(1, -1)
        link_pos = np.asarray(self.fk.forward_link_pos_rot_xyzw(dof)[0])[0]  # (n_links, 3) in the pelvis frame
        world = link_pos @ _rot_from_wxyz(q).T + np.asarray(pos, dtype=np.float32)
        return world[self.edges]

    def update(self, low_state, qj: np.ndarray, depth_raw, runner) -> None:
        self.base.update(low_state, qj, depth_raw)
        yaw_now = _yaw_of_wxyz(low_state.imu_state.quaternion)
        start = runner.planner.start_yaw
        cmd_deg = math.degrees(runner.yaw_command_rad)
        if abs(cmd_deg - float(self.cmd_h.value)) > 0.5:  # the remote changed it: reflect without re-triggering
            self.cmd_h.value = round(cmd_deg)
        self.prog_h.value = 0.0 if start is None else round(math.degrees((yaw_now - start + math.pi) % (2 * math.pi) - math.pi), 1)
        self.age_h.value = int(runner.plan_age)
        self.lat_h.value = round(float(runner.async_planner.last_latency_ms), 1) if runner.async_planner is not None else 0.0
        self.held_h.value = bool(runner.stand_held)
        states = runner.async_planner.latest_states if runner.async_planner is not None else runner.planner.latest_states
        if states is None:
            return
        st = states[0].numpy()
        segs, cols = [], []
        for i, k in enumerate(SKELETON_FRAMES):
            s = self._skeleton(st[min(k, len(st) - 1)], yaw_now)
            segs.append(s)
            t = i / max(1, len(SKELETON_FRAMES) - 1)
            cols.append(np.tile(np.array([255 * t, 200 * (1 - t), 0], dtype=np.uint8), (len(s), 2, 1)))
        self.server.scene.add_line_segments("/plan/skeleton", points=np.concatenate(segs), colors=np.concatenate(cols), line_width=3.0)
        path = np.array([ghost_pose(r, self.slices, yaw_now)[0] for r in st], dtype=np.float32)
        self.server.scene.add_line_segments("/plan/path", points=np.stack([path[:-1], path[1:]], axis=1),
                                            colors=np.full((len(path) - 1, 2, 3), (255, 255, 255), dtype=np.uint8), line_width=2.0)


class ViserThread(threading.Thread):
    def __init__(self, view: Vigor2Viser, robot, depth_client, runner, hz: float = 20.0) -> None:
        super().__init__(daemon=True, name="viser")
        self.view, self.robot, self.depth_client, self.runner, self.dt = view, robot, depth_client, runner, 1.0 / hz
        self.running = True

    def run(self) -> None:
        while self.running:
            state, _ = self.robot.low_state_snapshot_with_age()
            if state is not None and self.runner.planner.start_yaw is not None:
                self.view.update(state, self.robot.qj, self.depth_client.latest_depth, self.runner)
            time.sleep(self.dt)
