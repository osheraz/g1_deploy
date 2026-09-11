"""Raw timestamped RealSense depth for the shared native recovery runtime.

Run on the computer physically connected to the camera. No robot SDK, motor
publisher, image filtering or color alignment is involved. The consumer owns
the checkpoint's crop, sampling, range mask and capture history.
"""
import argparse
import json
import threading
import time

import cv2
import numpy as np


class ClockServer:
    """Independent clock replies; never waits for camera capture/PNG encoding."""
    def __init__(self, context, endpoint):
        self.context=context;self.endpoint=endpoint
        self.stop=threading.Event();self.ready=threading.Event();self.error=None
        self.thread=threading.Thread(target=self.run,daemon=True,name='depth-clock')
        self.thread.start()
        if not self.ready.wait(3):raise RuntimeError('Camera clock server failed to start')
        if self.error:raise RuntimeError(self.error)

    def run(self):
        import zmq
        socket=self.context.socket(zmq.REP);socket.setsockopt(zmq.LINGER,0)
        try:
            socket.bind(self.endpoint);self.ready.set()
            while not self.stop.is_set():
                if not socket.poll(100):continue
                request=socket.recv_json()
                socket.send_json(dict(clock_probe=request.get('clock_probe'),unix_s=time.time()))
        except Exception as exc:self.error=str(exc);self.ready.set()
        finally:socket.close()

    def close(self):
        self.stop.set();self.thread.join(timeout=1.)


def packet(depth, *, serial, scale, intrinsics, stamp, domain, sequence):
    """Lossless wire packet; hardware/global timestamps are never receipt time."""
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError('Expected raw uint16 optical-axis depth')
    header = dict(schema='hv2.native_depth_packet.v1', serial=serial,
        processing='raw_depth_no_filters_no_alignment', capture_s=stamp,
        clock_domain=domain, sequence=sequence, depth_scale_m=scale,
        intrinsics=[intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy],
        distortion_coefficients=list(intrinsics.coeffs))
    ok, encoded = cv2.imencode('.png', depth, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError('Depth PNG encoding failed')
    return [json.dumps(header, allow_nan=False).encode(), encoded.tobytes()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', required=True)
    parser.add_argument('--bind', default='tcp://*:5556')
    parser.add_argument('--clock-bind', default='tcp://127.0.0.1:5558')
    args = parser.parse_args()
    import pyrealsense2 as rs
    import zmq
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.setsockopt(zmq.LINGER, 0)
    pipeline = rs.pipeline()
    started = False
    clock_server=None
    try:
        socket.bind(args.bind)
        clock_server=ClockServer(context,args.clock_bind)
        config = rs.config()
        config.enable_device(args.serial)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = pipeline.start(config)
        started = True
        sensor = profile.get_device().first_depth_sensor()
        if sensor.supports(rs.option.global_time_enabled):
            sensor.set_option(rs.option.global_time_enabled, 1)
        intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        if any(abs(value) > 1.e-8 for value in intrinsics.coeffs):
            raise ValueError('Nonzero distortion requires calibrated rectification')
        print(f'Raw depth on {args.bind}: serial={args.serial}, 640x480 @ 30 Hz. No motors.', flush=True)
        while True:
            if clock_server.error:raise RuntimeError(clock_server.error)
            frame = pipeline.wait_for_frames().get_depth_frame()
            if not frame:
                continue
            clock = frame.get_frame_timestamp_domain()
            if clock in (rs.timestamp_domain.system_time, rs.timestamp_domain.global_time):
                domain = 'unix'
            elif clock == rs.timestamp_domain.hardware_clock:
                domain = 'hardware'
            else:
                raise RuntimeError('Unknown camera clock domain')
            socket.send_multipart(packet(np.asanyarray(frame.get_data()),
                serial=args.serial, scale=float(sensor.get_depth_scale()), intrinsics=intrinsics,
                stamp=float(frame.get_timestamp()) * .001, domain=domain,
                sequence=int(frame.get_frame_number())))
    except KeyboardInterrupt:
        pass
    finally:
        if started:
            pipeline.stop()
        if clock_server:clock_server.close()
        socket.close()
        context.term()


if __name__ == '__main__':
    main()
