#!/usr/bin/env python3
"""Publish scripted teleop on the SONIC `planner` topic, without a headset.

Same wire format the real PICO publisher uses, so the whole path downstream
(PicoClient -> PicoReferenceWindow -> planner -> SONIC) is the real one.

    python scripts/fake_pico.py --test nav      # base motion only
    python scripts/fake_pico.py --test vr3      # arm targets only, no base
    python scripts/fake_pico.py --test both

Leave it running; start deploy_eskin --mode pico (or sim2sim_pico) alongside.
"""
from __future__ import annotations

import argparse
import json
import struct
import time

import numpy as np
import zmq

HEADER_SIZE = 1280
TOPIC = b"planner"

IDLE, SLOW_WALK = 0, 1

# Official endpoint order is [left wrist, right wrist, virtual head], all in
# the pelvis frame. These are the same neutral values used by scripted_teleop.
LEFT_WRIST, RIGHT_WRIST, HEAD = 0, 1, 2
REST_POS = np.array([[0.0903, 0.1615, -0.2411],
                     [0.1280, -0.1522, -0.2461],
                     [0.0241, -0.0081, 0.4028]], np.float32)
REST_QUAT = np.array([[0.7295, 0.3145, 0.5533, -0.2506],
                      [0.7320, -0.2639, 0.5395, 0.3217],
                      [0.9991, 0.0110, 0.0402, -0.0002]], np.float32)


def build(mode, movement, facing, speed, height, vr_pos=None, vr_quat=None):
    fields = [
        {"name": "mode", "dtype": "i32", "shape": [1]},
        {"name": "movement", "dtype": "f32", "shape": [3]},
        {"name": "facing", "dtype": "f32", "shape": [3]},
        {"name": "speed", "dtype": "f32", "shape": [1]},
        {"name": "height", "dtype": "f32", "shape": [1]},
    ]
    payload = (struct.pack("<i", int(mode))
               + struct.pack("<fff", *map(float, movement))
               + struct.pack("<fff", *map(float, facing))
               + struct.pack("<f", float(speed))
               + struct.pack("<f", float(height)))
    if vr_pos is not None:
        flat = np.asarray(vr_pos, np.float32).reshape(-1)
        fields.append({"name": "vr_position", "dtype": "f32", "shape": [flat.size]})
        payload += flat.tobytes()
        flatq = np.asarray(vr_quat, np.float32).reshape(-1)
        fields.append({"name": "vr_orientation", "dtype": "f32", "shape": [flatq.size]})
        payload += flatq.tobytes()
    header = json.dumps({"v": 1, "endian": "le", "count": 1, "fields": fields},
                        separators=(",", ":")).encode()
    return TOPIC + header.ljust(HEADER_SIZE, b"\x00") + payload


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", choices=["nav", "vr3", "both", "idle"], default="idle")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--speed", type=float, default=0.2, help="nav: m/s")
    args = ap.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.port}")
    print(f"publishing '{args.test}' on tcp://*:{args.port} at {args.rate:g} Hz")
    time.sleep(0.5)                      # let subscribers connect

    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            mode, speed = IDLE, -1.0
            movement = np.zeros(3, np.float32)
            facing = np.array([1.0, 0.0, 0.0], np.float32)
            vr_pos = vr_quat = None

            if args.test in ("nav", "both"):
                mode, speed = SLOW_WALK, args.speed
                movement = np.array([1.0, 0.0, 0.0], np.float32)

            if args.test in ("vr3", "both"):
                vr_pos = REST_POS.copy()
                vr_quat = REST_QUAT.copy()
                # raise and lower the right wrist, 0.2 m, 0.25 Hz
                vr_pos[RIGHT_WRIST, 2] += 0.20 * (
                    0.5 - 0.5 * np.cos(2 * np.pi * 0.25 * t)
                )

            sock.send(build(mode, movement, facing, speed, -1.0, vr_pos, vr_quat))
            print(f"\r t={t:5.1f}s mode={mode} speed={speed:+.2f} "
                  f"move={movement.round(2)} vr={'yes' if vr_pos is not None else 'no'}   ",
                  end="", flush=True)
            time.sleep(1.0 / args.rate)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
