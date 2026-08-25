import threading
import zmq
import numpy as np

try:
    import cv2
except ModuleNotFoundError:  # A headless --no-camera launch does not need it.
    cv2 = None


def _require_cv2():
    if cv2 is None:
        raise RuntimeError("camera/depth support requires opencv-python")
    return cv2


def pack_depth_images(
    depth_raw: np.ndarray | None,
    out_h: int,
    out_w: int,
    min_d: float,
    max_d: float,
):
    if depth_raw is None:
        vis = np.zeros((out_h, out_w), dtype=np.uint8)
        ov = np.zeros((out_h, out_w), dtype=np.uint8)
        return vis, ov

    d = np.clip(depth_raw.astype(np.float32), min_d, max_d)
    vis = ((d - min_d) / (max_d - min_d) * 255.0).astype(np.uint8)

    o = depth_to_obs_vector_match_sim(depth_raw, out_h, out_w, min_d, max_d)
    ov = (o.reshape(out_h, out_w) * 255.0).astype(np.uint8)
    return vis, ov


DEPTH_SCALE = 0.0010000000474974513


class SimpleDepthClient(threading.Thread):
    def __init__(
        self,
        server_address: str,
        port: int,
        height: int,
        width: int,
        show_debug: bool = False,
    ):
        super().__init__(daemon=True)
        _require_cv2()
        self.server_address = server_address
        self.port = port
        self.H = height
        self.W = width
        self.show_debug = show_debug

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{self.server_address}:{self.port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # latest_depth      : centre-cropped square, metres -- what the policy sees
        #                     (matches the sim's square ego camera).
        # latest_depth_full  : uncropped HxW, metres -- for data collection.
        # latest_color       : uncropped HxW BGR uint8.
        self.latest_depth = None
        self.latest_depth_full = None
        self.latest_color = None
        # The encoded bytes exactly as received. Recorders should store these
        # rather than the decoded arrays: colour is already JPEG so the decoded
        # array recovers no quality, and depth PNG is lossless. ~5x smaller.
        self.latest_color_jpg = None
        self.latest_depth_png = None
        self.latest_seq = 0
        self.running = True
        self.frame_count = 0

    def run(self):
        print("Depth client thread is running")

        try:
            while self.running:
                parts = self.socket.recv_multipart()
                if len(parts) == 2:
                    color_msg, depth_msg = parts
                    self.latest_color = cv2.imdecode(
                        np.frombuffer(color_msg, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    self.latest_color_jpg = color_msg
                else:
                    depth_msg = parts[0]
                    self.latest_color_jpg = None
                self.latest_depth_png = depth_msg

                depth = cv2.imdecode(
                    np.frombuffer(depth_msg, dtype=np.uint8), cv2.IMREAD_UNCHANGED
                )

                if depth is None:
                    continue

                if depth.size != self.H * self.W:
                    print(f"Unexpected depth shape: {depth.shape}")

                depth_m = depth * DEPTH_SCALE
                self.latest_depth_full = depth_m

                # Centre crop to a square for the policy; keep the full frame
                # above so logging retains the camera's native resolution.
                H, W = depth_m.shape
                crop = min(H, W)
                y0 = (H - crop) // 2
                x0 = (W - crop) // 2
                self.latest_depth = depth_m[y0:y0 + crop, x0:x0 + crop]
                self.frame_count += 1
                self.latest_seq += 1

                if self.show_debug:

                    d = np.clip(depth, 0.1, 2.0) 

                    vis = ((d - 0.1) / (2.0 - 0.1) * 255).astype(np.uint8)
                    cv2.imshow("Depth", vis)
                    if self.latest_color is not None:
                        cv2.imshow("Color", self.latest_color)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.running = False
                        break

        except zmq.ZMQError as zmq_err:
            print(f"ZeroMQ error occurred: {zmq_err}")
            self.running = False
        finally:
            self.socket.close()
            self.context.term()
            if self.show_debug:
                cv2.destroyAllWindows()


def depth_to_obs_vector_match_sim(
    depth_raw: np.ndarray | None,
    out_h: int,
    out_w: int,
    min_d: float,
    max_d: float,
) -> np.ndarray:
    if depth_raw is None:
        return np.zeros((out_h * out_w,), dtype=np.float32)

    _require_cv2()

    depth_m = depth_raw.astype(np.float32)
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
    """Viewer (viser, browser): left = raw depth from the robot (0.1-2 m grey, no-depth black), right = the
    obs the policy sees (64x48 after resize, 0 -> max_d, clip, normalise), upscaled. Ctrl-C stops."""
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
    parser.add_argument("--min-d", type=float, default=0.1)   # must equal training clipping_range
    parser.add_argument("--max-d", type=float, default=2.0)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--viser-port", type=int, default=8081)  # the venv's cv2 is headless: the panel is served by viser
    args = parser.parse_args()

    client = SimpleDepthClient(
        server_address=args.server, port=args.port, height=args.height, width=args.width,
        show_debug=False,
    )
    client.start()
    print("Depth client started.")
    viewer = None
    if not args.no_window:
        import viser
        server = viser.ViserServer(port=args.viser_port)
        viewer = server.gui.add_image(np.zeros((args.height, 2 * args.width + 8, 3), np.uint8), label="robot depth | policy obs")
        print(f"Viewer: http://localhost:{args.viser_port}  (left: raw 0.1-2 m, no-depth black; right: the 64x48 policy obs upscaled). Ctrl-C to stop.")

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
                    print(f"[Depth] {fps:.1f} Hz seq={last_seq} valid={np.mean(full > 0) * 100:.0f}%", flush=True)
                continue
            obs = depth_to_obs_vector_match_sim(full, args.obs_h, args.obs_w, args.min_d, args.max_d)
            obs_m = (obs.reshape(args.obs_h, args.obs_w) + 0.5) * (args.max_d - args.min_d) + args.min_d
            raw = grey(full.astype(np.float32), args.min_d, args.max_d)
            small = grey(obs_m, args.min_d, args.max_d)
            small = cv2.resize(small, (full.shape[1], full.shape[0]), interpolation=cv2.INTER_NEAREST)
            panel = np.concatenate([raw, np.full((full.shape[0], 8), 128, np.uint8), small], axis=1)
            cv2.putText(panel, f"raw {full.shape[1]}x{full.shape[0]}  {fps:.1f} Hz  valid {np.mean(full > 0) * 100:.0f}%",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
            cv2.putText(panel, f"policy obs {args.obs_w}x{args.obs_h}  clip {args.min_d}-{args.max_d} m  0->max",
                        (full.shape[1] + 18, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
            viewer.image = np.repeat(panel[..., None], 3, axis=-1)
    except KeyboardInterrupt:
        pass
    finally:
        client.running = False
        client.join()


if __name__ == "__main__":
    main()
