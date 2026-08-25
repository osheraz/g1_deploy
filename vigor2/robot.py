"""RobotInterface + a 500 Hz command re-writer: the last command is re-sent every 2 ms; if the
control loop goes quiet for max_command_age_s the re-writer latches and publishes damping."""
from __future__ import annotations
import copy
import threading
import time
from core.common.command_helper import create_damping_cmd
from core.robot.robot_interface import CRC, RobotInterface


class WatchdogRobot(RobotInterface):
    def __init__(self, config):
        super().__init__(config)
        self.max_command_age_s = float(config.max_command_age_s)
        self._snap_lock = threading.Lock()
        self._snap = None
        self._snap_monotonic = 0.0
        self.watchdog_tripped = False
        self.rewriter_publish_count = 0
        self._rewriter_running = True
        self._rewriter_thread = threading.Thread(target=self._rewriter, name="lowcmd-rewriter", daemon=True)
        self._rewriter_thread.start()

    def send_cmd(self) -> None:
        with self._snap_lock:
            if self.watchdog_tripped:
                raise RuntimeError("motor-command watchdog is latched")
            super().send_cmd()
            self._snap = copy.deepcopy(self.low_cmd)
            self._snap_monotonic = time.monotonic()

    def _rewriter(self) -> None:
        while self._rewriter_running:
            with self._snap_lock:
                if self._snap is not None:
                    if not self.watchdog_tripped and time.monotonic() - self._snap_monotonic > self.max_command_age_s:
                        self.watchdog_tripped = True
                        create_damping_cmd(self._snap)
                        self._snap.crc = CRC().Crc(self._snap)
                    self.lowcmd_publisher_.Write(self._snap)
                    self.rewriter_publish_count += 1
            time.sleep(0.002)

    def send_damping_now(self) -> None:
        """Publish damping even after the latch; the one command allowed past it."""
        with self._snap_lock:
            create_damping_cmd(self.low_cmd)
            self.low_cmd.crc = CRC().Crc(self.low_cmd)
            self._snap = copy.deepcopy(self.low_cmd)
            self._snap_monotonic = time.monotonic()
            self.watchdog_tripped = True
            self.lowcmd_publisher_.Write(self._snap)

    def stop_rewriter(self) -> None:
        self._rewriter_running = False
        self._rewriter_thread.join(timeout=1.0)
