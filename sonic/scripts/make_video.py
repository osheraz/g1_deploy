from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
from humanoidverse.simulator.base_simulator.utlis.video_sink import VideoSink


DEPLOY_ROOT = Path(__file__).resolve().parents[2]
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))


def _depth_to_rgb_u8(depth_u8: np.ndarray) -> np.ndarray:
    return np.repeat(depth_u8[:, :, None], 3, axis=2)


def render_one_separate(
    npz_path: Path,
    out_ext_mp4: Path,
    out_depth_mp4: Path,
    fps: int,
    use_realtime: bool,
    max_sec: float,
    resize_depth_to_ext: bool = True,
):
    z = np.load(str(npz_path), allow_pickle=True)
    ext = z["ext_cam_bgr_u8"].astype(np.uint8)
    dep = z["depth_vis_u8"].astype(np.uint8)
    t = z["t"].astype(np.float32) if use_realtime and ("t" in z.files) else None

    if dep.ndim == 4:
        dep = dep[..., 0]

    n_frames = ext.shape[0]
    height, width = ext.shape[1], ext.shape[2]

    t0 = float(t[0]) if t is not None and len(t) else 0.0
    max_k = n_frames if t is not None else min(n_frames, int(max_sec * fps))

    vs_ext = VideoSink(str(out_ext_mp4), fps=fps)
    vs_dep = VideoSink(str(out_depth_mp4), fps=fps)

    for k in range(max_k):
        if t is not None and (float(t[k]) - t0) > max_sec:
            break

        frame_ext = ext[k][:, :, ::-1].copy()

        depth = dep[k]
        if resize_depth_to_ext and (depth.shape[0] != height or depth.shape[1] != width):
            depth_rgb = _depth_to_rgb_u8(depth)
            frame_dep = np.array(
                Image.fromarray(depth_rgb).resize((width, height), resample=Image.NEAREST),
                dtype=np.uint8,
            )
        else:
            frame_dep = _depth_to_rgb_u8(depth)

        if t is None or k >= n_frames - 1:
            nrep = 1
        else:
            dt = float(t[k + 1] - t[k])
            nrep = max(1, int(round(dt * fps)))

        for _ in range(nrep):
            vs_ext.append(frame_ext)
            vs_dep.append(frame_dep)

    vs_ext.close()
    vs_dep.close()


def make_grid(vids, out_mp4: Path, fps: int, gap: int, max_sec: float):
    if not vids:
        return None

    sample = iio.imread(str(vids[0]), index=0)
    height, width = sample.shape[0], sample.shape[1]

    n_tiles = len(vids)
    cols = int(math.ceil(math.sqrt(n_tiles)))
    rows = int(math.ceil(n_tiles / cols))

    out_width = cols * width + (cols - 1) * gap
    out_height = rows * height + (rows - 1) * gap

    readers = [iio.imiter(str(p)) for p in vids]
    frames = [None] * n_tiles
    done = [False] * n_tiles

    vs = VideoSink(str(out_mp4), fps=fps)
    max_frames = int(max_sec * fps)

    for _ in range(max_frames):
        all_done = True
        for i, it in enumerate(readers):
            if done[i]:
                continue
            try:
                frames[i] = next(it)
            except StopIteration:
                done[i] = True
                frames[i] = None
            if not done[i]:
                all_done = False

        if all_done:
            break

        canvas = np.zeros((out_height, out_width, 3), dtype=np.uint8)

        for i in range(n_tiles):
            row = i // cols
            col = i % cols
            y0 = row * (height + gap)
            x0 = col * (width + gap)
            frame = frames[i]
            if frame is None:
                continue
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            canvas[y0 : y0 + height, x0 : x0 + width] = frame

        vs.append(canvas)

    vs.close()
    return out_mp4


def main():
    use_realtime = True
    fps = 50
    gap = 8
    max_tiles = 25
    max_sec = 20.0
    resize_depth_to_ext = True

    root = (DEPLOY_ROOT / "logs/real_rollouts/balance_board").resolve()

    ext_vids = []
    dep_vids = []

    for npz_path in sorted(root.rglob("*.npz")):
        videos_dir = npz_path.parent / "videos"
        videos_dir.mkdir(exist_ok=True)

        out_ext = videos_dir / f"{npz_path.stem}_ext.mp4"
        out_dep = videos_dir / f"{npz_path.stem}_depth.mp4"

        if (not out_ext.exists()) or (not out_dep.exists()):
            render_one_separate(
                npz_path,
                out_ext,
                out_dep,
                fps,
                use_realtime,
                max_sec,
                resize_depth_to_ext=resize_depth_to_ext,
            )

        ext_vids.append(out_ext)
        dep_vids.append(out_dep)

    random.seed(0)
    idx = list(range(len(ext_vids)))
    random.shuffle(idx)
    idx = idx[:max_tiles]

    ext_sample = [ext_vids[i] for i in idx]
    dep_sample = [dep_vids[i] for i in idx]

    combined_dir = root / "combined"
    combined_dir.mkdir(exist_ok=True)

    out_ext_grid = combined_dir / "combined_grid_shuffle_ext.mp4"

    # if not out_ext_grid.exists():
    #     p = make_grid(ext_sample, out_ext_grid, fps=fps, gap=gap, max_sec=max_sec)
    #     if p is not None:
    #         print(p)


if __name__ == "__main__":
    main()
