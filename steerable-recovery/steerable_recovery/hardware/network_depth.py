"""Timestamped depth over ZMQ, with explicit source processing provenance.

Wire: [JSON metadata, lossless uint16 PNG or raw pixels]. Untimestamped legacy packets are
rejected. Raw and verified VIGOR-filtered pixels use the same calibrated sampler
and model range/mask transform. The consumer never filters them a second time.
"""
from __future__ import annotations

import json
import math
import threading
import time

import numpy as np

from motion_diffusion.inference.sensors import calibrated_depth_image
from steerable_recovery.hardware.camera import DepthUnavailable
from g1_deploy.core.common.depth_wire import decode_depth, PNG, RAW

SCHEMA = 'hv2.native_depth_packet.v1'
CLOCK_REFRESH_INTERVAL_S=10.
CLOCK_RETRY_INTERVAL_S=1.
PROCESSING_PROFILES = {
    'raw': 'raw_depth_no_filters_no_alignment',
    'vigor': 'vigor_policy_depth_no_alignment_v2',
    'vigor_legacy': 'vigor_osher_depth_no_alignment_v1',
}


def decode_packet(parts, capture_config, *, serial, return_source=False, processing=None):
    if len(parts) != 2:
        raise ValueError('Native depth needs timestamp metadata plus pixels; legacy stream is not compatible')
    try:
        metadata = json.loads(parts[0])
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError('Depth stream has no native timestamp header') from exc
    if metadata.get('schema') != SCHEMA or metadata.get('serial') != serial:
        raise ValueError('Depth packet schema/camera identity differs')
    if processing is not None:
        if processing not in PROCESSING_PROFILES:
            raise ValueError('Unknown requested camera processing')
        if metadata.get('processing') != PROCESSING_PROFILES[processing]:
            raise ValueError(f'Camera processing differs from requested {processing}')
    from hv2_common.sensors.depth import is_v3,CONDITION_SCHEMA
    if is_v3(capture_config):
        # An explicit deployment filter is separate from the checkpoint's raw
        # training recipe. Preserve its real profile and all acquired provenance.
        filtered_override = processing in ('vigor', 'vigor_legacy')
        if not filtered_override and (metadata.get('depth_protocol')!=CONDITION_SCHEMA or metadata.get('processing')!='raw_depth_no_filters_no_alignment'):
            raise ValueError('Depth v3 requires an explicitly versioned raw, unfiltered camera stream')
        required=('firmware','visual_preset','emitter_enabled','exposure','serial','depth_scale_m','intrinsics','processing')
        if not isinstance(metadata.get('sensor_provenance'),dict) or any(k not in metadata['sensor_provenance'] for k in required):
            raise ValueError('Depth v3 packet omitted acquired sensor provenance')
        if any(metadata['sensor_provenance'][k]!=metadata[k] for k in ('serial','depth_scale_m','intrinsics','processing')):
            raise ValueError('Depth v3 sensor provenance conflicts with the actual packet')
    if metadata.get('processing') not in ('raw_depth_no_filters_no_alignment',
                                         'vigor_policy_depth_no_alignment_v2',
                                         'vigor_osher_depth_no_alignment_v1'):
        raise ValueError('Unknown camera filtering or alignment profile')
    if metadata['processing']!='raw_depth_no_filters_no_alignment' and not isinstance(metadata.get('processing_config'),dict):
        raise ValueError('Filtered depth needs its processing configuration')
    if metadata.get('clock_domain') not in ('unix', 'hardware'):
        raise ValueError('Camera packet has no qualified capture clock')
    stamp, scale = float(metadata['capture_s']), float(metadata['depth_scale_m'])
    if not math.isfinite(stamp) or not 0 < scale < .1:
        raise ValueError('Invalid capture timestamp or depth units')
    if type(metadata.get('sequence')) is not int or metadata['sequence'] < 0:
        raise ValueError('Camera sequence must be a nonnegative integer')
    image = decode_depth(metadata, parts[1])
    if metadata.get('distortion_coefficients') != [0.] * 5:
        raise ValueError('Nonzero camera distortion requires calibrated rectification')
    depth, valid = calibrated_depth_image(image.astype(np.float32) * scale,
        intrinsics=metadata['intrinsics'], capture_config=capture_config)
    if return_source:return metadata, depth, valid, image
    return metadata, depth, valid


class NetworkDepth:
    def __init__(self, *, endpoint, serial, buffer, robot, calibration_receipt=None, hardware_clock=None,
                 capture_config=None,clock_endpoint=None,mount_confirmed=False,frame_recorder=None,processing=None):
        import zmq
        if processing is not None and processing not in PROCESSING_PROFILES:
            raise ValueError('Unknown requested camera processing')
        self.processing = processing
        from steerable_recovery.hardware.camera import validate_calibration, wall_monotonic_offset
        self.capture_config = buffer.capture_config if capture_config is None else capture_config
        self.clock_endpoint=clock_endpoint;self.clock_report={}
        self.frame_recorder=frame_recorder
        if clock_endpoint:
            if mount_confirmed is not True:raise ValueError('Confirm the configured camera mount before deployment')
            from steerable_recovery.hardware.clocks import measure_camera_clock
            hardware_clock,self.clock_report=measure_camera_clock(clock_endpoint)
            self.uncertainty=hardware_clock.uncertainty_s
        else:self.uncertainty = validate_calibration(calibration_receipt, self.capture_config, serial)
        self.wall_offset = wall_monotonic_offset()
        self.endpoint, self.serial, self.buffer, self.robot = endpoint, serial, buffer, robot
        self.hardware_clock = hardware_clock
        self.last_capture_timing = {}
        self.processing_report = None
        self.rejected_frames=0;self.last_rejection=None
        self.paired_frames=0;self.last_packet_s=time.monotonic();self.last_clock_check_s=self.last_packet_s
        self.clock_refresh_failures=0;self.clock_warning=None
        self._clock_lock=threading.Lock();self._last_clock_warning_s=-math.inf
        self.clock_thread=None
        self.fault = None
        self._stop = threading.Event()
        self.zmq = zmq
        self.thread = threading.Thread(target=self._read, daemon=True, name='native-network-depth')
        self.thread.start()
        if self.clock_endpoint:
            self.clock_thread=threading.Thread(target=self._refresh_clock,daemon=True,name='native-depth-clock')
            self.clock_thread.start()

    def _refresh_clock_once(self):
        from steerable_recovery.hardware.clocks import measure_camera_clock, CameraClockProbeError
        try:
            mapping,report=measure_camera_clock(self.clock_endpoint,samples=8)
        except CameraClockProbeError as exc:
            # A congested exchange says nothing about whether the clocks moved.
            # Do not replace a valid alignment with this uncertain measurement.
            now=time.monotonic()
            with self._clock_lock:
                self.clock_refresh_failures+=1;self.clock_warning=str(exc)
            if now-self._last_clock_warning_s>=10.:
                print(f'Camera clock refresh delayed: {exc}. Keeping the last accepted alignment; retrying.',flush=True)
                self._last_clock_warning_s=now
            return False
        with self._clock_lock:
            if abs(mapping.offset_s-self.hardware_clock.offset_s)>.005:
                raise RuntimeError('Camera host clock jumped; restart deployment')
            self.hardware_clock=mapping;self.clock_report=report
            self.last_clock_check_s=time.monotonic();self.clock_warning=None
        return True

    def _refresh_clock(self):
        # Probing can block on network replies. Never do it on the depth reader
        # or the 50 Hz preview/control thread, including during the START hold.
        delay=CLOCK_REFRESH_INTERVAL_S
        while not self._stop.wait(delay):
            try:delay=CLOCK_REFRESH_INTERVAL_S if self._refresh_clock_once() else CLOCK_RETRY_INTERVAL_S
            except Exception as exc:
                if not self._stop.is_set():self.fault=str(exc)
                return

    def _read(self):
        context = self.zmq.Context()
        socket = context.socket(self.zmq.SUB)
        socket.setsockopt(self.zmq.SUBSCRIBE, b'')
        socket.setsockopt(self.zmq.RCVHWM, 2)
        socket.setsockopt(self.zmq.RCVTIMEO, 200)
        socket.setsockopt(self.zmq.LINGER, 0)
        last_sequence, last_capture = -1, -math.inf
        try:
            socket.connect(self.endpoint)
            while not self._stop.is_set():
                try:
                    parts = socket.recv_multipart()
                except self.zmq.Again:
                    continue
                packet_arrival = time.monotonic()
                self.last_packet_s=packet_arrival
                # A verified clock/source fault quarantines this camera. Motor
                # control can continue with missing-depth masks, but no new
                # frames may use the invalid mapping.
                if self.fault:continue
                metadata, depth, valid, source = decode_packet(parts, self.capture_config,
                    serial=self.serial,return_source=True,processing=self.processing)
                decode_ready = time.monotonic()
                processing=dict(processing=metadata['processing'],config=metadata.get('processing_config'),
                    sensor_provenance=metadata.get('sensor_provenance'),depth_protocol=metadata.get('depth_protocol'))
                if self.processing_report is None:
                    self.processing_report=processing
                    print(f"Camera processing: {metadata['processing']}; transport: {metadata.get('encoding',PNG)}",flush=True)
                if self.clock_endpoint and metadata['clock_domain']=='unix':
                    with self._clock_lock:
                        mapping=self.hardware_clock
                        sync_age_s=packet_arrival-self.last_clock_check_s
                        sync_warning=self.clock_warning
                    capture=mapping.convert(metadata['capture_s'])
                    uncertainty=mapping.uncertainty_s
                elif metadata['clock_domain'] == 'unix':
                    if abs(time.monotonic() - time.time() - self.wall_offset) > .005:
                        raise RuntimeError('Host clock stepped; restart depth clock calibration')
                    capture = metadata['capture_s'] + self.wall_offset
                    uncertainty = self.uncertainty + .001
                elif self.hardware_clock is not None:
                    capture = self.hardware_clock.convert(metadata['capture_s'])
                    uncertainty = self.hardware_clock.uncertainty_s
                else:
                    raise RuntimeError('Network camera needs hardware clock mapping; receipt is not capture')
                if metadata['sequence'] <= last_sequence or capture <= last_capture:
                    self._reject('Camera sequence or capture timestamp did not advance')
                    continue
                if capture > packet_arrival + uncertainty or packet_arrival - capture > .25:
                    self._reject('Camera capture is future-dated or stale; verify cross-computer clock alignment')
                    continue
                last_sequence, last_capture = metadata['sequence'], capture
                try:
                    observation = self.robot.at_capture(capture)
                except RuntimeError as exc:
                    self._reject(f'Capture-time robot state unavailable: {exc}')
                    continue
                ready = time.monotonic()
                self.buffer.append(depth_m=depth, surface_valid=valid, capture_s=capture,
                    arrival_s=ready, observation=observation, clock_uncertainty_s=uncertainty,
                    sequence=int(metadata['sequence']))
                self.last_capture_timing = dict(capture_s=capture,packet_arrival_s=packet_arrival,
                    ready_s=ready,sequence=metadata['sequence'],
                    pair_error_s=observation.capture_s-capture,processing=metadata['processing'],
                    processing_ms=metadata.get('processing_ms',0.),
                    wire_encoding=metadata.get('encoding',PNG),wire_bytes=len(parts[1]),
                    decode_prepare_ms=(decode_ready-packet_arrival)*1000.)
                for key in ('encoding_ms','publish_max_mbps','publisher_dropped_frames'):
                    if key in metadata:self.last_capture_timing[key]=metadata[key]
                if 'publish_unix_s' in metadata and metadata['clock_domain']=='unix':
                    self.last_capture_timing['source_age_at_publish_s']=metadata['publish_unix_s']-metadata['capture_s']
                if self.clock_endpoint and metadata['clock_domain']=='unix':
                    self.last_capture_timing.update(clock_offset_s=mapping.offset_s,
                        clock_probe_uncertainty_s=uncertainty,clock_sync_age_s=max(0.,sync_age_s),
                        clock_refresh_warning=sync_warning)
                self.paired_frames+=1
                self.last_rejection=None
                if self.frame_recorder is not None:
                    raw_wire=metadata.get('encoding',PNG)==RAW
                    offer=self.frame_recorder.offer_raw if raw_wire else self.frame_recorder.offer_png
                    offer(source if raw_wire else parts[1],depth_scale_m=metadata['depth_scale_m'],
                        intrinsics=metadata['intrinsics'],timing=self.last_capture_timing,
                        source_processing=processing)
        except BaseException as exc:
            if not self._stop.is_set(): self.fault = str(exc)
        finally:
            socket.close()
            context.term()

    def _reject(self,reason):
        self.rejected_frames+=1;self.last_rejection=reason

    def check(self):
        if self.fault: raise DepthUnavailable(self.fault)
        now=time.monotonic()
        if now-self.last_packet_s>(.25 if self.paired_frames else 5.):
            raise DepthUnavailable('Depth stream stopped or disconnected')
        if self.paired_frames and now-self.last_capture_timing['capture_s']>.25:
            timing=self.last_capture_timing
            age_ms=(now-timing['capture_s'])*1000
            details=f'capture age {age_ms:.1f} ms'
            if 'packet_arrival_s' in timing:
                delivery_ms=(timing['packet_arrival_s']-timing['capture_s'])*1000
                details+=f', last delivery delay {delivery_ms:.1f} ms'
            reason=(f'No fresh depth capture paired with robot state '
                    f'({details})')
            if self.last_rejection:reason+=f': {self.last_rejection}'
            raise DepthUnavailable(reason)

    def close(self):
        self._stop.set()
        self.thread.join(timeout=1.)
        if self.clock_thread is not None:self.clock_thread.join(timeout=1.)
