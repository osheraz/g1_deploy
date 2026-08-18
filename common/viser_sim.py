"""Browser viewer for the MuJoCo sim2sim loop: URDF mirror + drive buttons.

SSH replacement for the GLFW window (which needs a local X display). Serves
http://<host>:8080 with the robot model, base pose, and the same command set
as the numeric keypad: axes, gait mode, IDLE stop, stand capture.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf

from common.viser_display import JOINT_NAMES_BY_MOTOR_INDEX


class ViserSimView:
    def __init__(self, urdf_path, teleop=None, on_capture=None, port: int = 8080):
        self.server = viser.ViserServer(port=port)
        self.base = self.server.scene.add_frame("/base", show_axes=False)
        self.urdf = ViserUrdf(self.server, urdf_or_path=Path(urdf_path),
                              root_node_name="/base", load_meshes=True,
                              load_collision_meshes=False)
        self.server.scene.add_grid("/grid", width=4.0, height=4.0,
                                   position=(0.0, 0.0, 0.0))
        self.status = self.server.gui.add_text("status", "waiting", disabled=True)

        if teleop is not None:
            self._add_controls(teleop, on_capture)

    def _add_controls(self, teleop, on_capture) -> None:
        def bump(dlx=0.0, dly=0.0, drx=0.0):
            teleop.set_axes(float(np.clip(teleop.lx + dlx, -1, 1)),
                            float(np.clip(teleop.ly + dly, -1, 1)),
                            float(np.clip(teleop.rx + drx, -1, 1)))

        with self.server.gui.add_folder("drive"):
            rows = [
                ("turn L", lambda: bump(drx=-0.5)),
                ("forward", lambda: bump(dly=+0.5)),
                ("turn R", lambda: bump(drx=+0.5)),
                ("left", lambda: bump(dlx=-0.5)),
                ("STOP (idle)", lambda: teleop.set_axes(0, 0, 0)),
                ("right", lambda: bump(dlx=+0.5)),
                ("back", lambda: bump(dly=-0.5)),
            ]
            for label, fn in rows:
                btn = self.server.gui.add_button(label)
                btn.on_click(lambda _e, f=fn: f())

        from common.pico_client import LocomotionMode

        with self.server.gui.add_folder("gait"):
            for label, mode in (("slow walk", LocomotionMode.SLOW_WALK),
                                ("walk", LocomotionMode.WALK),
                                ("run", LocomotionMode.RUN)):
                btn = self.server.gui.add_button(label)
                btn.on_click(lambda _e, m=mode: teleop.set_locomotion_mode(m))
            if on_capture is not None:
                cap = self.server.gui.add_button("capture stand (Numpad Enter)")
                cap.on_click(lambda _e: on_capture())

        self._teleop = teleop

    def update(self, base_pos, base_quat_wxyz, qj23) -> None:
        self.base.position = tuple(float(x) for x in base_pos)
        self.base.wxyz = tuple(float(x) for x in base_quat_wxyz)
        self.urdf.update_cfg({name: float(qj23[i])
                              for i, name in enumerate(JOINT_NAMES_BY_MOTOR_INDEX)})
        t = getattr(self, "_teleop", None)
        if t is not None:
            self.status.value = (f"mode={t.locomotion_mode.name} "
                                 f"lx={t.lx:+.1f} ly={t.ly:+.1f} rx={t.rx:+.1f}")
