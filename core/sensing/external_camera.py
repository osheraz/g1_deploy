from __future__ import annotations

import time
import threading
from typing import Optional

import numpy as np
import cv2


class ExtCamRGB(threading.Thread):
    def __init__(
        self,
        cam_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        warmup: int = 30,
    ):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup = warmup

        self.running = True
        self.latest_bgr: Optional[np.ndarray] = None
        self.frame_count = 0
        self.open_ok = False

        self.cap: Optional[cv2.VideoCapture] = None

    def run(self):
        cap = cv2.VideoCapture(self.cam_id)
        self.cap = cap
        if not cap.isOpened():
            self.open_ok = False
            self.running = False
            return


        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        cap.set(cv2.CAP_PROP_FPS, float(self.fps))

        self.open_ok = True

        for _ in range(self.warmup):
            ok, frame = cap.read()
            if ok and frame is not None:
                self.latest_bgr = frame
                self.frame_count += 1

        while self.running:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            
            self.latest_bgr = frame
            self.frame_count += 1

        cap.release()

    def stop(self):
        self.running = False
        self.join(timeout=2.0)


def main():
    cam = ExtCamRGB(
        cam_id=0,
        width=1920,
        height=1080,
        fps=30,
        warmup=10,
    )
    cam.start()

    t0 = time.time()
    while cam.running and not cam.open_ok and (time.time() - t0) < 2.0:
        time.sleep(0.01)

    if not cam.open_ok:
        cam.stop()
        print("camera open failed")
        return

    last = time.time()
    last_count = 0

    while True:
        frame = cam.latest_bgr
        if frame is not None:
            cv2.imshow("ExtCam RGB", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        now = time.time()
        if now - last >= 1.0:
            fps = (cam.frame_count - last_count) / (now - last)
            print(f"{fps:.1f}")
            last = now
            last_count = cam.frame_count

        time.sleep(0.001)

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
