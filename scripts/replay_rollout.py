from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))

from common.viser_display import DeployViser
from config import DEFAULT_URDF_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log", type=str, default="./balance_board/TEST_1770074557.npz"
    )
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--urdf", type=str, default=DEFAULT_URDF_PATH)
    args = ap.parse_args()

    log_path = (DEPLOY_ROOT / "logs/real_rollouts" / args.log).resolve()

    z = np.load(str(log_path), allow_pickle=True)

    qj = z["qj"].astype(np.float32)
    imu_quat_wxyz = z["imu_quat_wxyz"].astype(np.float32)
    depth_vis_u8 = z["depth_vis_u8"].astype(np.uint8)
    depth_obs_u8 = z["depth_obs_u8"].astype(np.uint8)
    ext_rgb = (
        z["ext_cam_bgr_u8"].astype(np.uint8) if "ext_cam_bgr_u8" in z.files else None
    )

    n_frames = int(qj.shape[0])

    urdf_path = Path(args.urdf).expanduser()

    dv = DeployViser(
        urdf_path=urdf_path,
        with_depth=True,
        depth_min=0.1,
        depth_max=3.0,
        depth_scale=1.0,
    )

    replay_folder = dv.server.gui.add_folder("Replay")
    with replay_folder:
        idx = dv.server.gui.add_slider(
            "Index", min=0, max=n_frames - 1, step=1, initial_value=0
        )
        play = dv.server.gui.add_checkbox("Play", False)
        fps = dv.server.gui.add_number("FPS", float(args.fps))

    depth_folder = dv.server.gui.add_folder("Replay Depth")
    with depth_folder:
        height = int(depth_vis_u8.shape[1])
        width = int(depth_vis_u8.shape[2])
        dummy_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth_vis_handle = dv.server.gui.add_image(dummy_rgb, label="depth_vis_u8")
        depth_obs_handle = dv.server.gui.add_image(dummy_rgb, label="depth_obs_u8")

    def render(k: int):
        dummy = type("LS", (), {})()
        dummy.tick = int(k)
        dummy.mode_pr = 0
        dummy.mode_machine = 0
        imu = type("IMU", (), {})()
        imu.quaternion = tuple(float(x) for x in imu_quat_wxyz[k])
        dummy.imu_state = imu

        rgb_bgr = ext_rgb[k] if ext_rgb is not None else None

        dv.update(
            dummy,
            qj[k],
            depth_raw=None,
            rgb_bgr=rgb_bgr,
        )

        g0 = depth_vis_u8[k]
        g1 = depth_obs_u8[k]
        depth_vis_handle.image = np.stack([g0, g0, g0], axis=-1)
        depth_obs_handle.image = np.stack([g1, g1, g1], axis=-1)

    render(0)

    k = 0
    last = time.perf_counter()
    while True:
        kk = int(idx.value)
        if kk != k:
            k = kk
            render(k)

        if play.value:
            now = time.perf_counter()
            target_dt = 1.0 / max(1e-6, float(fps.value))
            if (now - last) >= target_dt:
                last = now
                k = min(k + 1, n_frames - 1)
                idx.value = k
                render(k)
                if k == n_frames - 1:
                    play.value = False

        time.sleep(0.002)


if __name__ == "__main__":
    main()
