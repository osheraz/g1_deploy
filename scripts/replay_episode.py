#!/usr/bin/env python3
"""Replay a recorded episode in viser: robot pose, camera, tactile.

Counterpart to replay_rollout.py, which reads deploy_g1's flat .npz. This one
reads the episode directories deploy_eskin.py writes:

    episodes/episode_001/
        state.npz     one row per control step
        frames.bin    colour+depth, still encoded
        meta.json

    python scripts/replay_episode.py --episode ~/episodes/episode_001

Opens http://localhost:8080 -- scrub the Index slider or tick Play.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))

from common.episode import load_episode  # noqa: E402
from common.tactile_render import render_sensor_panels  # noqa: E402
from common.viser_display import DeployViser  # noqa: E402
from config import DEFAULT_URDF_PATH  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True, help="episode directory")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--urdf", default=DEFAULT_URDF_PATH)
    ap.add_argument("--deadzone", type=int, default=12,
                    help="tactile: suppress values within this of the resting baseline")
    args = ap.parse_args()

    ep = Path(args.episode).expanduser()
    state, get_frame = load_episode(ep)
    meta = json.loads((ep / "meta.json").read_text()) if (ep / "meta.json").exists() else {}

    n = len(state["t"])
    sensors = list(meta.get("sensors", []))
    has_cam = "frame_offset" in state

    print(f"{ep.name}: {n} steps, {meta.get('duration_s', 0):.1f}s, "
          f"tactile={sensors or 'none'}, camera={'yes' if has_cam else 'no'}")
    print("fields:", ", ".join(sorted(k for k in state if k != "frame_offset")))

    qj = state["qj"].astype(np.float32) if "qj" in state else None
    imu = (state["imu_quat_wxyz"].astype(np.float32)
           if "imu_quat_wxyz" in state else None)

    # Resting baseline per sensor, so light contact is visible rather than
    # swamped by each taxel's offset. Same idea as the live viewer.
    baselines = {}
    for s in sensors:
        key = f"tactile_{s.lower()}_raw_u8"
        if key in state and len(state[key]) >= 30:
            baselines[s] = np.round(state[key][:30].mean(axis=0)).astype(int).tolist()

    dv = DeployViser(
        urdf_path=Path(args.urdf).expanduser(),
        with_depth=has_cam,
        depth_min=0.1,
        depth_max=3.0,
        depth_scale=1.0,
        tactile_sensors=sensors,
    )

    folder = dv.server.gui.add_folder("Replay")
    with folder:
        idx = dv.server.gui.add_slider("Index", min=0, max=max(n - 1, 1), step=1,
                                       initial_value=0)
        play = dv.server.gui.add_checkbox("Play", False)
        fps = dv.server.gui.add_number("FPS", float(args.fps))
        label = dv.server.gui.add_text("t", "0.000 s")

    def render(k: int):
        ls = type("LS", (), {})()
        ls.tick = int(k)
        ls.mode_pr = 0
        ls.mode_machine = 0
        q = type("IMU", (), {})()
        q.quaternion = tuple(float(x) for x in (imu[k] if imu is not None
                                                else (1.0, 0.0, 0.0, 0.0)))
        ls.imu_state = q

        color = depth = None
        if has_cam:
            color, depth_mm = get_frame(k)
            if depth_mm is not None:
                depth = depth_mm.astype(np.float32) * 0.001   # mm -> m

        dv.update(ls, qj[k] if qj is not None else np.zeros(23, np.float32),
                  depth_raw=depth, rgb_bgr=color)

        panels = {}
        for s in sensors:
            raw = state.get(f"tactile_{s.lower()}_raw_u8")
            if raw is not None:
                panels.update(render_sensor_panels(
                    s, raw[k].tolist(), baseline_values=baselines.get(s),
                    deadzone=args.deadzone))
        if panels:
            dv.update_tactile_images(panels)

        label.value = f"{float(state['t'][k]):.3f} s   step {k}/{n - 1}"

    render(0)
    print("\nviser: http://localhost:8080")

    k, last = 0, time.perf_counter()
    while True:
        kk = int(idx.value)
        if kk != k:
            k = kk
            render(k)
        if play.value:
            now = time.perf_counter()
            if (now - last) >= 1.0 / max(1e-6, float(fps.value)):
                last = now
                k = min(k + 1, n - 1)
                idx.value = k
                render(k)
                if k == n - 1:
                    play.value = False
        time.sleep(0.002)


if __name__ == "__main__":
    main()
