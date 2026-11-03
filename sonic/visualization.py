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

from core.review.display import JOINT_NAMES_BY_MOTOR_INDEX


class ViserSimView:
    def __init__(self, urdf_path, teleop=None, on_capture=None, port: int = 8080):
        self.server = viser.ViserServer(port=port)
        self.port = self.server.get_port()
        if self.port != port:
            print(f"[viser] port {port} busy -> serving on :{self.port} "
                  f"(open THIS port, old tabs are a dead server)")
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
        from sonic.inputs.pico_client import LocomotionMode

        gui = self.server.gui
        with gui.add_folder("drive (sliders = stick)"):
            fwd = gui.add_slider("forward", min=-1.0, max=1.0, step=0.05,
                                 initial_value=0.0)
            strafe = gui.add_slider("strafe", min=-1.0, max=1.0, step=0.05,
                                    initial_value=0.0)
            turn = gui.add_slider("turn", min=-1.0, max=1.0, step=0.05,
                                  initial_value=0.0)
            stop = gui.add_button("STOP -> idle", color="red")

        def push(_e=None):
            teleop.set_axes(strafe.value, fwd.value, turn.value)
            print(f"[viser] axes lx={strafe.value:+.2f} ly={fwd.value:+.2f} "
                  f"rx={turn.value:+.2f}", flush=True)

        for s in (fwd, strafe, turn):
            s.on_update(push)

        def do_stop(_e=None):
            fwd.value = strafe.value = turn.value = 0.0
            teleop.set_axes(0, 0, 0)

        stop.on_click(do_stop)

        with gui.add_folder("gait"):
            gaits = gui.add_button_group("mode", ("SLOW", "WALK", "RUN"))
            gaits.on_click(lambda _e: teleop.set_locomotion_mode(
                {"SLOW": LocomotionMode.SLOW_WALK,
                 "WALK": LocomotionMode.WALK,
                 "RUN": LocomotionMode.RUN}[gaits.value]))
            if on_capture is not None:
                cap = gui.add_button("capture stand pose")
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
