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
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="192.168.123.164")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    client = SimpleDepthClient(
        server_address=args.server,
        port=args.port,
        height=args.height,
        width=args.width,
        show_debug=True,
    )
    client.start()
    print("Depth client started.")

    last_count = 0
    last_time = time.time()

    try:
        while True:
            time.sleep(1.0)
            now = time.time()
            frames = client.frame_count - last_count
            dt = now - last_time
            fps = frames / dt if dt > 0 else 0.0
            print(f"[Depth FPS] {fps:.1f} Hz  (frames={client.frame_count})")
            last_count = client.frame_count
            last_time = now
    except KeyboardInterrupt:
        print("Stopping depth client...")
    finally:
        client.running = False
        client.join()


if __name__ == "__main__":
    main()
