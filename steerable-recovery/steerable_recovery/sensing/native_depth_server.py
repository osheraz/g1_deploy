"""Timestamped RealSense depth with original VIGOR hole filling by default.

Run on the computer physically connected to the camera. No robot SDK, motor
publisher or color alignment is involved. --processing raw bypasses filters;
--processing vigor retains source missing/near/far categories after smoothing.
The consumer owns the checkpoint's crop, sampling, range mask and history.
"""
import argparse
import json
import math
import queue
import threading
import time

import numpy as np

if __package__:
    from core.common.depth_wire import encode_depth
elif __name__ == '__main__':  # Standalone copy on the camera computer.
    from depth_wire import encode_depth
else:  # Direct file loading in deployment contract checks.
    from g1_deploy.core.common.depth_wire import encode_depth


PROCESSING = {'raw': 'raw_depth_no_filters_no_alignment',
              'vigor': 'vigor_policy_depth_no_alignment_v2',
              'vigor_legacy': 'vigor_osher_depth_no_alignment_v1'}
VIGOR_PROFILE = dict(
    source_sha256='687b49e04ce6c045c2d6a9bf2d5f361ea2225b6ecad9206575c3d19a5561fc93',
    order=['decimation', 'depth_to_disparity', 'spatial', 'temporal',
           'disparity_to_depth', 'hole_filling'],
    decimation=1, spatial=dict(magnitude=2, alpha=.5, delta=20, holes_fill=0),
    temporal=dict(alpha=.4, delta=20, persistence=3, history='new_filter_per_frame'),
    hole_filling=1, threshold=False, alignment=False)
POLICY_PROFILE = dict(VIGOR_PROFILE,
    preserve_source_categories=True, minimum_m=.1, far_boundaries_m=[2.,3.],
    category_changes='restore original sensor value; do not fill missing/near returns')


def preserve_source_categories(raw, filtered, scale):
    """Keep VIGOR smoothing without relabeling missing/near/far observations.

    The shared model contracts use a 0.1 m minimum and 2 or 3 m maximum.
    Range clipping remains in the model consumer; no extra mask/image is sent
    or recorded. Float32 metres match that consumer's boundary arithmetic.
    """
    if raw.dtype!=np.uint16 or filtered.dtype!=np.uint16 or raw.shape!=filtered.shape:
        raise ValueError('Matching uint16 source and filtered depth required')
    source=raw.astype(np.float32)*np.float32(scale)
    changed=filtered.astype(np.float32)*np.float32(scale)
    restore=source<POLICY_PROFILE['minimum_m']
    restore |= (source>=POLICY_PROFILE['minimum_m'])!=(changed>=POLICY_PROFILE['minimum_m'])
    for boundary in POLICY_PROFILE['far_boundaries_m']:
        restore |= (source>boundary)!=(changed>boundary)
    return np.where(restore,raw,filtered).astype(np.uint16)


def process_depth(frame, rs, mode):
    """Match img_server_osher.py's active path, including temporal resets.

    Pin the defaults read from the old robot's SDK. Its custom self.rs_filters
    and threshold filter were unused. Reusing a temporal filter across frames
    would change the old behavior, so each call constructs fresh filters.
    """
    if mode == 'raw': return frame
    if mode not in ('vigor','vigor_legacy'): raise ValueError('Unknown depth processing mode')
    frame = rs.decimation_filter(1).process(frame)
    frame = rs.disparity_transform(True).process(frame)
    spatial = rs.spatial_filter()
    for option, value in ((rs.option.filter_magnitude, 2),
                          (rs.option.filter_smooth_alpha, .5),
                          (rs.option.filter_smooth_delta, 20), (rs.option.holes_fill, 0)):
        spatial.set_option(option, value)
    frame = spatial.process(frame)
    temporal = rs.temporal_filter()
    for option, value in ((rs.option.filter_smooth_alpha, .4),
                          (rs.option.filter_smooth_delta, 20), (rs.option.holes_fill, 3)):
        temporal.set_option(option, value)
    frame = temporal.process(frame)
    frame = rs.disparity_transform(False).process(frame)
    return rs.hole_filling_filter(1).process(frame)


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


def packet(depth, *, serial, scale, intrinsics, stamp, domain, sequence,
           processing='raw', processing_config=None, processing_ms=0., encoding='png',sensor_provenance=None):
    """Lossless wire packet; hardware/global timestamps are never receipt time."""
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError('Expected uint16 optical-axis depth')
    header = dict(schema='hv2.native_depth_packet.v1', serial=serial,
        processing=PROCESSING[processing], capture_s=stamp,
        clock_domain=domain, sequence=sequence, depth_scale_m=scale,
        intrinsics=[intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy],
        distortion_coefficients=list(intrinsics.coeffs))
    if sensor_provenance is not None:
        header.update(sensor_provenance=sensor_provenance)
        if processing=='raw':header['depth_protocol']='hv2.planner_metric_depth_condition.v3'
    if processing != 'raw':
        header.update(processing_config=processing_config or (POLICY_PROFILE if processing=='vigor' else VIGOR_PROFILE),
                      processing_ms=float(processing_ms))
    wire, payload = encode_depth(depth, encoding)
    header.update(wire)
    return [json.dumps(header, allow_nan=False).encode(), payload]


class DepthPublisher:
    """Encode/publish off the capture thread with at most one waiting image.

    A busy encoder drops pending work instead of accumulating camera latency.
    Original exposure IDs and timestamps are retained, so loss remains visible
    to the consumer. The publishing thread owns its ZMQ socket.
    """
    def __init__(self,context,endpoint,*,max_mbps=0.,encoding='png'):
        if not math.isfinite(max_mbps) or max_mbps<0:
            raise ValueError('max_mbps must be finite and nonnegative')
        self.context=context;self.endpoint=endpoint
        self.max_mbps=float(max_mbps)
        if encoding not in ('png','raw'):raise ValueError('Unknown depth encoding')
        self.encoding=encoding
        self.pending=queue.Queue(maxsize=1)
        self.stop=threading.Event();self.ready=threading.Event();self.error=None
        self.dropped=0
        self.thread=threading.Thread(target=self.run,daemon=True,name='depth-publisher')
        self.thread.start()
        if not self.ready.wait(3):raise RuntimeError('Depth publisher failed to start')
        self.check()

    def check(self):
        if self.error:raise RuntimeError(f'Depth publisher failed: {self.error}')

    def offer(self,depth,**metadata):
        self.check()
        # Detach the array from the camera frame's lifetime before returning.
        item=(depth.copy(),metadata)
        try:self.pending.put_nowait(item)
        except queue.Full:
            try:self.pending.get_nowait();self.dropped+=1
            except queue.Empty:pass
            self.pending.put_nowait(item)

    def run(self):
        import zmq
        socket=self.context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM,2);socket.setsockopt(zmq.LINGER,0)
        try:
            socket.bind(self.endpoint);self.ready.set()
            next_encode_s=0.
            while not self.stop.is_set():
                # Limit bytes before choosing the newest pending exposure.
                # SNDHWM alone cannot bound the downstream TCP/SSH queues.
                # Do not encode and then sleep on an already aging frame, or
                # catch up with a burst after an encoder/network delay.
                if self.stop.wait(max(0.,next_encode_s-time.monotonic())):break
                try:depth,metadata=self.pending.get(timeout=.1)
                except queue.Empty:continue
                began=time.monotonic()
                parts=packet(depth,encoding=self.encoding,**metadata)
                header=json.loads(parts[0])
                header.update(encoding_ms=(time.monotonic()-began)*1000,
                    publish_unix_s=time.time(),publish_max_mbps=self.max_mbps,
                    publisher_dropped_frames=self.dropped)
                parts[0]=json.dumps(header,allow_nan=False).encode()
                socket.send_multipart(parts)
                if self.max_mbps:
                    # Charge the encode interval to the byte budget too. Never
                    # accumulate credit during an encoder stall for a burst.
                    next_encode_s=max(time.monotonic(),
                        began+sum(map(len,parts))*8/(self.max_mbps*1e6))
        except Exception as exc:self.error=str(exc);self.ready.set()
        finally:socket.close()

    def close(self):
        self.stop.set();self.thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', required=True)
    parser.add_argument('--bind', default='tcp://*:5556')
    parser.add_argument('--clock-bind', default='tcp://127.0.0.1:5558')
    parser.add_argument('--max-mbps',type=float,default=0.,
                        help='Limit depth payload bandwidth, dropping pending exposures (0: unlimited). '
                             'Use --encoding png --max-mbps 40 on a 100 Mb/s link.')
    parser.add_argument('--encoding',choices=('png','raw'),default='png',
                        help='png: lossless compression; raw: identical uint16 pixels without camera-side '
                             'compression, requires gigabit (148 Mb/s depth payload at 30 Hz). '
                             'Independent of --processing; raw encoding still uses the selected filters.')
    parser.add_argument('--processing', choices=tuple(PROCESSING), default='vigor_legacy',
                        help='vigor_legacy: original VIGOR smoothing and hole filling (default); '
                             'vigor: smoothing with source missing/near/far categories preserved; '
                             'raw: no filtering')
    args = parser.parse_args()
    if not math.isfinite(args.max_mbps) or args.max_mbps<0:
        parser.error('--max-mbps must be finite and nonnegative')
    import pyrealsense2 as rs
    from importlib.metadata import version
    import zmq
    context = zmq.Context()
    pipeline = rs.pipeline()
    started = False
    clock_server=publisher=None
    try:
        publisher=DepthPublisher(context,args.bind,max_mbps=args.max_mbps,encoding=args.encoding)
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
        processing_config=dict(POLICY_PROFILE if args.processing=='vigor' else VIGOR_PROFILE,
                               sdk_version=version('pyrealsense2'))
        scale=float(sensor.get_depth_scale())
        device=profile.get_device()
        def option(name):
            key=getattr(rs.option,name)
            return float(sensor.get_option(key)) if sensor.supports(key) else None
        sensor_provenance=dict(serial=args.serial,
            firmware=device.get_info(rs.camera_info.firmware_version),
            visual_preset=option('visual_preset'),emitter_enabled=option('emitter_enabled'),
            exposure=option('exposure'),enable_auto_exposure=option('enable_auto_exposure'),
            depth_scale_m=scale,intrinsics=[intrinsics.fx,intrinsics.fy,intrinsics.ppx,intrinsics.ppy],
            processing=PROCESSING[args.processing],resolution=[640,480],format='z16',rate_hz=30,
            sdk_version=version('pyrealsense2'))
        print(f'Depth on {args.bind}: serial={args.serial}, 640x480 @ 30 Hz, '
              f'processing={args.processing}, async {args.encoding} publisher, '
              f'payload cap={args.max_mbps:g} Mb/s (0: unlimited). No motors.', flush=True)
        while True:
            publisher.check()
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
            began=time.perf_counter()
            processed=process_depth(frame,rs,args.processing)
            pixels=np.asanyarray(processed.get_data())
            if args.processing=='vigor':
                pixels=preserve_source_categories(np.asanyarray(frame.get_data()),pixels,scale)
            processing_ms=(time.perf_counter()-began)*1000
            exposure=(float(frame.get_frame_metadata(rs.frame_metadata_value.actual_exposure))
                if frame.supports_frame_metadata(rs.frame_metadata_value.actual_exposure) else option('exposure'))
            # Exposure time and sequence belong to the original camera frame,
            # never the completion time of filtering or PNG encoding.
            publisher.offer(pixels,
                serial=args.serial, scale=scale, intrinsics=intrinsics,
                stamp=float(frame.get_timestamp()) * .001, domain=domain,
                sequence=int(frame.get_frame_number()),processing=args.processing,
                processing_config=processing_config,processing_ms=processing_ms,
                sensor_provenance=dict(sensor_provenance,exposure=exposure))
    except KeyboardInterrupt:
        pass
    finally:
        if started:
            pipeline.stop()
        if publisher:publisher.close()
        if clock_server:clock_server.close()
        context.term()


if __name__ == '__main__':
    main()
