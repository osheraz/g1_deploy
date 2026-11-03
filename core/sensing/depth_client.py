"""Receive camera frames; expose the full image and square policy crop in metres."""

import json
import threading
import zmq
import numpy as np

if __package__:
    from core.sensing.depth_wire import decode_depth, PNG
else:
    from depth_wire import decode_depth, PNG

try:
    import cv2
except ModuleNotFoundError:  # A headless --no-camera launch does not need it.
    cv2 = None


def _require_cv2():
    if cv2 is None:
        raise RuntimeError("camera/depth support requires opencv-python")
    return cv2


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
                if not self.socket.poll(timeout=100):
                    continue
                parts = self.socket.recv_multipart()
                depth_scale = DEPTH_SCALE
                metadata = None
                if len(parts) == 2:
                    color_msg, depth_msg = parts
                    if color_msg.startswith(b"{"):
                        metadata = json.loads(color_msg)
                        if metadata["schema"] != "hv2.native_depth_packet.v1":
                            raise ValueError("Unsupported native depth packet schema")
                        depth_scale = float(metadata["depth_scale_m"])
                        self.latest_color = None
                        self.latest_color_jpg = None
                    else:
                        self.latest_color = cv2.imdecode(
                            np.frombuffer(color_msg, dtype=np.uint8), cv2.IMREAD_COLOR
                        )
                        self.latest_color_jpg = color_msg
                else:
                    depth_msg = parts[0]
                    self.latest_color_jpg = None
                self.latest_depth_png = (
                    depth_msg if metadata is None or metadata.get("encoding", PNG) == PNG else None
                )
                depth = (
                    decode_depth(metadata, depth_msg)
                    if metadata is not None
                    else cv2.imdecode(
                        np.frombuffer(depth_msg, dtype=np.uint8), cv2.IMREAD_UNCHANGED
                    )
                )

                if depth is None:
                    continue

                if depth.size != self.H * self.W:
                    print(f"Unexpected depth shape: {depth.shape}")

                depth_m = depth * depth_scale
                self.latest_depth_full = depth_m

                # Centre crop to a square for the policy; keep the full frame
                # above so logging retains the camera's native resolution.
                H, W = depth_m.shape
                crop = min(H, W)
                y0 = (H - crop) // 2
                x0 = (W - crop) // 2
                self.latest_depth = depth_m[y0 : y0 + crop, x0 : x0 + crop]
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
