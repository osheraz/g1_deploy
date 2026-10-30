"""Optional RealSense source for the exact metric depth capture contract.

Only the physical depth stream is used. Camera IMU is not substituted for the
pelvis IMU. Hardware-clock timestamps require an externally calibrated mapping;
global/system timestamps are converted from Unix time to local monotonic time.
"""
from __future__ import annotations

import math
import threading
import time
import numpy as np

from motion_diffusion.inference.sensors import calibrated_depth_image


class DepthUnavailable(RuntimeError):
    """A camera health failure, distinct from a control/runtime failure."""


class DepthHealth:
    """Report camera loss without interrupting motor control or inventing depth."""
    def __init__(self,camera,buffer,telemetry):
        self.camera,self.buffer,self.telemetry=camera,buffer,telemetry
        self.reason=None;self.unavailable_since=None;self.last_notice=-math.inf
        self.healthy_since=None;self.recovery_notice_pending=False

    def check(self,*,policy_active=False):
        now=time.monotonic()
        try:
            self.camera.check()
        except DepthUnavailable as exc:
            # A broken clock/source is not an ordinary old frame: quarantine
            # all its captures. Delayed valid captures expire in prepare().
            if self.camera.fault and self.buffer is not None:self.buffer.reset(now)
            reason=str(exc)
            self.healthy_since=None;self.recovery_notice_pending=True
            if self.unavailable_since is None:self.unavailable_since=now
            if self.reason is None or now-self.last_notice>=5.:
                self.telemetry.emit(dict(event='depth_unavailable',reason=reason,
                    monotonic_s=now,unavailable_s=now-self.unavailable_since,
                    policy_active=policy_active,
                    last_capture_timing=dict(self.camera.last_capture_timing)),high_rate=True)
            # Preserve every health transition in telemetry, but don't print
            # every threshold crossing when depth hovers around the age limit.
            if now-self.last_notice>=5.:
                suffix=('Recovery continues with unavailable depth masked.' if policy_active else
                        'Waiting for valid depth before policy start.')
                print(f'Depth unavailable: {reason}. {suffix}',flush=True)
                self.last_notice=now
            self.reason=reason
            return False
        if self.unavailable_since is not None:
            self.telemetry.emit(dict(event='depth_resumed',monotonic_s=now,
                unavailable_s=now-self.unavailable_since,policy_active=policy_active),high_rate=True)
        if self.healthy_since is None:self.healthy_since=now
        if self.recovery_notice_pending and now-self.healthy_since>=.5:
            print('Fresh paired depth stable again.',flush=True)
            self.recovery_notice_pending=False
        self.reason=None;self.unavailable_since=None
        return True


def deployment_capture_config(trained_capture, camera_config, *, enable_actuation=False):
    """Use the physical lens without rewriting the model's saved calibration.

    A lens mismatch is logged separately from the unchanged model contract.
    Motor trials with an older lens contract require an explicit config choice.
    Crop, resolution, mount and depth encoding stay intact.
    """
    from copy import deepcopy
    capture = deepcopy(trained_capture)
    intrinsics = np.asarray(camera_config.get('base_intrinsics', capture['base_intrinsics']), dtype=float)
    width, height = capture['base_resolution']
    if (intrinsics.shape != (4,) or not np.isfinite(intrinsics).all()
            or min(intrinsics[:2]) <= 0 or not 0 <= intrinsics[2] < width
            or not 0 <= intrinsics[3] < height):
        raise ValueError('Camera base_intrinsics must be finite [fx, fy, cx, cy] for this resolution')
    if (enable_actuation and not camera_config.get('allow_intrinsics_mismatch',False)
            and not np.allclose(intrinsics, trained_capture['base_intrinsics'], atol=1.e-3, rtol=0.)):
        raise ValueError('Camera intrinsics differ from the model: use no-motor testing or a matching model before actuation')
    capture['base_intrinsics'] = intrinsics.tolist()
    return capture


def validate_calibration(receipt, capture, serial):
    required=('camera_id','parent_body','translation_m','camera_to_link_quat_xyzw',
              'base_resolution','base_intrinsics','crop_px')
    if receipt.get('physically_measured') is not True or any(receipt.get(key)!=capture[key] for key in required):
        raise ValueError('A measured mount/intrinsics receipt matching the selected camera is required')
    if receipt.get('serial')!=serial:
        raise ValueError('Camera serial differs from calibration receipt')
    uncertainty=float(receipt.get('timestamp_uncertainty_s',float('inf')))
    if not receipt.get('timestamp_alignment_verified') or not 0<=uncertainty<=.004:
        raise ValueError('Measured camera timestamp error bound<=4ms required; global_time alone is not a qualification')
    return uncertainty


def wall_monotonic_offset():
    before=time.monotonic();wall=time.time();after=time.monotonic()
    if after-before>.001:
        raise RuntimeError('Wall/monotonic clock mapping uncertainty exceeds1ms')
    return (before+after)/2-wall


class RealSenseDepth:
    def __init__(self, *, serial,buffer,robot,calibration_receipt,hardware_clock=None,capture_config=None,frame_recorder=None):
        import pyrealsense2 as rs
        self.rs=rs;self.buffer=buffer;self.robot=robot;self.hardware_clock=hardware_clock
        capture=buffer.capture_config if capture_config is None else capture_config
        self.capture_config=capture
        self.frame_recorder=frame_recorder
        self.timestamp_uncertainty=validate_calibration(calibration_receipt,capture,serial)
        self.wall_offset=wall_monotonic_offset()
        self.last_capture_timing={}
        self._stop=threading.Event();self.fault=None;self.thread=None
        self.pipeline=rs.pipeline();config=rs.config();config.enable_device(str(serial))
        width,height=capture['base_resolution']
        config.enable_stream(rs.stream.depth,int(width),int(height),rs.format.z16,30)
        profile=self.pipeline.start(config)
        sensor=profile.get_device().first_depth_sensor()
        if sensor.supports(rs.option.global_time_enabled):sensor.set_option(rs.option.global_time_enabled,1)
        self.scale=float(sensor.get_depth_scale())
        if not 0<self.scale<.1:
            self.pipeline.stop();raise ValueError('Invalid physical depth-unit scale')
        self.intrinsics=profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        actual=[self.intrinsics.fx,self.intrinsics.fy,self.intrinsics.ppx,self.intrinsics.ppy]
        if not np.allclose(actual,capture['base_intrinsics'],atol=1.e-3,rtol=0):
            self.pipeline.stop();raise ValueError('Acquired physical stream intrinsics differ from trained camera')
        def option(name):
            key=getattr(rs.option,name)
            return float(sensor.get_option(key)) if sensor.supports(key) else None
        self.processing_report=dict(processing='raw_depth_no_filters_no_alignment',
            depth_protocol=capture['schema'],serial=str(serial),
            firmware=profile.get_device().get_info(rs.camera_info.firmware_version),
            visual_preset=option('visual_preset'),emitter_enabled=option('emitter_enabled'),
            exposure=option('exposure'),depth_scale_m=self.scale,intrinsics=actual)
        if str(self.intrinsics.model) not in ('distortion.none','distortion.brown_conrady','distortion.inverse_brown_conrady'):
            self.pipeline.stop();raise ValueError('Unqualified physical depth distortion model')
        if max(abs(x) for x in self.intrinsics.coeffs)>1.e-8:
            self.pipeline.stop();raise ValueError('Nonzero depth distortion requires qualified rectification and regenerated calibration')
        self.thread=threading.Thread(target=self._read,daemon=True,name='native-depth-captures');self.thread.start()

    def _read(self):
        last=-math.inf
        try:
            while not self._stop.is_set():
                frames=self.pipeline.wait_for_frames(timeout_ms=200)
                depth=frames.get_depth_frame()
                if not depth:continue
                arrival=time.monotonic();domain=depth.get_frame_timestamp_domain()
                stamp=float(depth.get_timestamp())*.001
                if domain in (self.rs.timestamp_domain.system_time,self.rs.timestamp_domain.global_time):
                    if abs(time.monotonic()-time.time()-self.wall_offset)>.005:
                        raise RuntimeError('Host wall clock stepped; camera timestamp mapping must restart')
                    capture=stamp+self.wall_offset;uncertainty=self.timestamp_uncertainty+.001
                elif self.hardware_clock is not None:
                    capture=self.hardware_clock.convert(stamp);uncertainty=self.hardware_clock.uncertainty_s
                else:raise RuntimeError('Camera uses unmapped hardware clock; receipt time is not capture time')
                if capture<=last:raise RuntimeError('Camera capture clock reset or duplicate frame')
                last=capture
                intr=[self.intrinsics.fx,self.intrinsics.fy,self.intrinsics.ppx,self.intrinsics.ppy]
                source=np.asanyarray(depth.get_data())
                image,valid=calibrated_depth_image(source.astype(np.float32)*self.scale,
                    intrinsics=intr,capture_config=self.capture_config)
                try:obs=self.robot.at_capture(capture)
                except RuntimeError:
                    # Startup capture has no synchronized state; leave it absent.
                    continue
                ready=time.monotonic()
                self.last_capture_timing=dict(capture_s=capture,packet_arrival_s=arrival,ready_s=ready,
                    proprio_capture_s=obs.capture_s,proprio_arrival_s=obs.arrival_s,
                    pair_error_s=obs.capture_s-capture)
                self.buffer.append(depth_m=image,surface_valid=valid,capture_s=capture,arrival_s=ready,
                    observation=obs,clock_uncertainty_s=uncertainty,sequence=int(depth.get_frame_number()))
                if self.frame_recorder is not None:
                    self.frame_recorder.offer_raw(source,depth_scale_m=self.scale,
                        intrinsics=intr,timing=self.last_capture_timing)
        except BaseException as exc:
            if not self._stop.is_set():self.fault=str(exc)

    def check(self):
        if self.fault:raise DepthUnavailable(self.fault)
        if self.last_capture_timing and time.monotonic()-self.last_capture_timing['capture_s']>.25:
            raise DepthUnavailable('No fresh depth capture paired with robot state')

    def close(self):
        self._stop.set();self.pipeline.stop()
        if self.thread:self.thread.join(timeout=1.)
