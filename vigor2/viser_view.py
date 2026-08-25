"""Viser view for deploy_vigor2: the measured robot (IMU + joints), the depth image, the planner's
predicted pelvis path over its horizon (line, green -> red = now -> +0.5 s) and a ghost robot at the
end of the horizon."""
from __future__ import annotations
import math
import threading
import time
from pathlib import Path

import numpy as np
from core.common.viser_display import DeployViser
from viser.extras import ViserUrdf

GHOST_FRAME = 24  # ticks ahead (20 ms each) of the ghost robot; the path line covers all frames


def _yaw_of_wxyz(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wxyz_from_rpy(roll: float, pitch: float, yaw: float):
    cr, sr, cp, sp, cy, sy = (math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2),
                              math.sin(pitch / 2), math.cos(yaw / 2), math.sin(yaw / 2))
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


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
    def __init__(self, urdf_path: str, depth_min: float, depth_max: float, depth_scale: float, bundle, runner) -> None:
        self.runner = runner
        urdf_path = Path(urdf_path)
        self.base = DeployViser(urdf_path=urdf_path, with_depth=True, depth_min=depth_min,
                                depth_max=depth_max, depth_scale=depth_scale)
        server = self.base.server
        self.slices = {r["name"]: slice(int(r["start"]), int(r["end"])) for r in bundle.feature_layout}
        self.dof_names = list(bundle.dof_names)
        self.server = server
        self.ghost_frame = server.scene.add_frame("/plan/ghost", show_axes=False)
        self.ghost = ViserUrdf(server, urdf_or_path=urdf_path, root_node_name="/plan/ghost",
                               mesh_color_override=(0.2, 0.55, 1.0, 0.35), load_collision_meshes=False)
        self.ghost_joints = set(self.ghost.get_actuated_joint_names())
        self.path = None
        with server.gui.add_folder("Planner"):
            self.cmd_h = server.gui.add_slider("yaw command (deg)", min=-180.0, max=180.0, step=5.0,
                                               initial_value=round(math.degrees(runner.yaw_command_rad)))
            self.cmd_h.on_update(lambda _: setattr(runner, "yaw_command_rad", math.radians(float(self.cmd_h.value))))
            self.prog_h = server.gui.add_number("yaw progress (deg)", 0.0, disabled=True)
            self.age_h = server.gui.add_number("plan age (ticks)", 0, disabled=True)
            self.lat_h = server.gui.add_number("plan latency (ms)", 0.0, disabled=True)
            self.held_h = server.gui.add_checkbox("stand held", False, disabled=True)

    def update(self, low_state, qj: np.ndarray, depth_raw, runner) -> None:
        self.base.update(low_state, qj, depth_raw)
        yaw_now = _yaw_of_wxyz(low_state.imu_state.quaternion)
        start = runner.planner.start_yaw
        cmd_deg = math.degrees(runner.yaw_command_rad)
        if abs(cmd_deg - float(self.cmd_h.value)) > 0.5:  # D-pad changed it: reflect without re-triggering
            self.cmd_h.value = round(cmd_deg)
        self.prog_h.value = 0.0 if start is None else round(math.degrees((yaw_now - start + math.pi) % (2 * math.pi) - math.pi), 1)
        self.age_h.value = int(runner.plan_age)
        self.lat_h.value = round(float(runner.async_planner.last_latency_ms), 1) if runner.async_planner is not None else 0.0
        self.held_h.value = bool(runner.stand_held)
        states = runner.async_planner.latest_states if runner.async_planner is not None else runner.planner.latest_states
        if states is None:
            return
        st = states[0].numpy()
        pts = np.array([ghost_pose(row, self.slices, yaw_now)[0] for row in st], dtype=np.float32)
        seg = np.stack([pts[:-1], pts[1:]], axis=1)  # (n-1, 2, 3)
        n = len(seg)
        col = np.stack([np.linspace(0, 255, n), np.linspace(200, 0, n), np.zeros(n)], axis=1).astype(np.uint8)
        self.path = self.server.scene.add_line_segments("/plan/path", points=seg, colors=np.repeat(col[:, None, :], 2, axis=1), line_width=4.0)
        row = st[min(GHOST_FRAME, n - 1)]
        self.ghost_frame.position, self.ghost_frame.wxyz = ghost_pose(row, self.slices, yaw_now)
        self.ghost.update_cfg({n_: float(v) for n_, v in zip(self.dof_names, row[self.slices["dof_pos"]]) if n_ in self.ghost_joints})


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
