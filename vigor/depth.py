"""VIGOR depth preprocessing and inspection of the policy image."""

import numpy as np
from core.sensing.depth_client import SimpleDepthClient, _require_cv2


def pack_depth_images(
    depth_raw: np.ndarray | None,
    out_h: int,
    out_w: int,
    min_d: float,
    max_d: float,
    crop_px,
):
    if depth_raw is None:
        vis = np.zeros((out_h, out_w), dtype=np.uint8)
        ov = np.zeros((out_h, out_w), dtype=np.uint8)
        return vis, ov

    d = np.clip(depth_raw.astype(np.float32), min_d, max_d)
    vis = ((d - min_d) / (max_d - min_d) * 255.0).astype(np.uint8)

    o = depth_to_obs_vector_match_sim(depth_raw, out_h, out_w, min_d, max_d, crop_px)
    ov = (o.reshape(out_h, out_w) * 255.0).astype(np.uint8)
    return vis, ov


def depth_to_obs_vector_match_sim(
    depth_raw: np.ndarray | None,
    out_h: int,
    out_w: int,
    min_d: float,
    max_d: float,
    crop_px,
) -> np.ndarray:
    """crop_px = (x, y) cut from each side of the raw frame before the resize; the sim renders the same crop (ego_camera.crop_px)."""
    if depth_raw is None:
        return np.zeros((out_h * out_w,), dtype=np.float32)

    cv2 = _require_cv2()

    cx, cy = (int(v) for v in crop_px)
    depth_m = depth_raw.astype(np.float32)[
        cy : depth_raw.shape[0] - cy, cx : depth_raw.shape[1] - cx
    ]
    depth_m[depth_m <= 0.0] = max_d  # RealSense "no depth" = 0; the sim renders a miss as max_d

    H, W = depth_m.shape
    if (H, W) != (out_h, out_w):
        depth_m = cv2.resize(depth_m, (out_w, out_h), interpolation=cv2.INTER_AREA)

    d = np.nan_to_num(
        depth_m,
        nan=max_d,
        posinf=max_d,
        neginf=min_d,
    )
    d = np.clip(d, min_d, max_d)

    depth_norm = (d - min_d) / (max_d - min_d)

    depth_norm -= 0.5

    return depth_norm.astype(np.float32).reshape(-1)


def main():
    """Viewer: left = raw depth from the robot (0.1-2 m grey, no-depth black), right = the obs the
    policy sees (64x48 after resize, 0 -> max_d, clip, normalise), upscaled. q quits."""
    import argparse
    import time
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="192.168.123.164")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--obs-h", type=int, default=48)
    parser.add_argument("--obs-w", type=int, default=64)
    parser.add_argument("--min-d", type=float, default=0.1)  # must equal training clipping_range
    parser.add_argument("--max-d", type=float, default=2.0)
    parser.add_argument(
        "--crop",
        type=int,
        nargs=2,
        default=(32, 8),
        metavar=("X", "Y"),
        help="px cut per side before the resize; = fallrec.yaml depth_crop_px (old stack: 0 0)",
    )
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    client = SimpleDepthClient(
        server_address=args.server,
        port=args.port,
        height=args.height,
        width=args.width,
        show_debug=False,
    )
    client.start()
    print(
        "Depth client started."
        + (
            ""
            if args.no_window
            else "  Window: left raw 0.1-2 m (no-depth black), right the 64x48 policy obs of the boxed crop. q quits."
        )
    )

    def grey(depth_m, lo, hi):
        g = np.clip((depth_m - lo) / (hi - lo), 0.0, 1.0)
        g = (g * 255).astype(np.uint8)
        g[depth_m <= 0.0] = 0
        return g

    last_seq, last_time, fps = -1, time.time(), 0.0
    try:
        while True:
            if client.latest_seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq = client.latest_seq
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last_time, 1e-6)
            last_time = now
            full = client.latest_depth_full
            if full is None:
                continue
            if args.no_window:
                if last_seq % 30 == 0:
                    print(
                        f"[Depth] {fps:.1f} Hz seq={last_seq} valid={np.mean(full > 0) * 100:.0f}%",
                        flush=True,
                    )
                continue
            obs = depth_to_obs_vector_match_sim(
                full, args.obs_h, args.obs_w, args.min_d, args.max_d, args.crop
            )
            obs_m = (obs.reshape(args.obs_h, args.obs_w) + 0.5) * (
                args.max_d - args.min_d
            ) + args.min_d
            raw = grey(full.astype(np.float32), args.min_d, args.max_d)
            cv2.putText(
                raw,
                f"{full.shape[1]}x{full.shape[0]}  {fps:.1f} Hz  valid {np.mean(full > 0) * 100:.0f}%",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                255,
                1,
            )
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(
                raw,
                (args.crop[0], args.crop[1]),
                (full.shape[1] - args.crop[0] - 1, full.shape[0] - args.crop[1] - 1),
                (0, 255, 0),
                2,
            )
            cv2.imshow(
                f"robot depth {full.shape[1]}x{full.shape[0]} (0.1-2 m, no-depth black, box = crop)",
                raw,
            )
            obs_img = grey(
                obs_m, args.min_d, args.max_d
            )  # 64x48 as the policy sees it, shown 6x nearest
            cv2.imshow(
                f"policy obs {args.obs_w}x{args.obs_h} (0->max, clip {args.min_d}-{args.max_d} m)",
                cv2.resize(
                    obs_img, (args.obs_w * 6, args.obs_h * 6), interpolation=cv2.INTER_NEAREST
                ),
            )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        client.running = False
        client.join()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
