"""Startup clock alignment; no operating-system clock changes.

Robot tick alignment estimates the fastest observed DDS delivery. Its reported
spread does not measure the unknown constant sensor/transport latency.
Camera host alignment uses round trips to the depth publisher's clock endpoint.
"""
from collections import deque
import math
import time

import numpy as np

from humanoidverse2.observations.streams import ClockMapping


class CameraClockProbeError(RuntimeError):
    """A timed-out or slow probe did not establish a new clock alignment."""


class RobotTickClock:
    def __init__(self, *, warmup_s=1., max_jitter_s=.005):
        self.samples=deque(maxlen=4096)
        self.warmup_s=float(warmup_s);self.max_jitter_s=float(max_jitter_s)
        self.mapping=None;self.last_tick=None;self.report={}

    def observe(self, tick_s, arrival_s):
        if not math.isfinite(tick_s) or not math.isfinite(arrival_s):
            raise ValueError('Nonfinite robot timestamp')
        if self.last_tick is not None and tick_s==self.last_tick:return None
        if self.last_tick is not None and tick_s<self.last_tick:
            raise RuntimeError('Robot tick reset or reordered packet; restart deployment')
        self.last_tick=tick_s
        self.samples.append((tick_s,arrival_s))
        if self.mapping is None:
            # Retry a rolling startup window. A delayed packet during imports
            # must not permanently poison alignment and require a new process.
            while len(self.samples)>1 and tick_s-self.samples[1][0]>=self.warmup_s:
                self.samples.popleft()
            if len(self.samples)<100 or tick_s-self.samples[0][0]<self.warmup_s:return None
            samples=np.asarray(self.samples);x=samples[:,0];y=samples[:,1]
            slope=(y[-1]-y[0])/(x[-1]-x[0])
            if not .98<slope<1.02:raise RuntimeError('LowState tick period does not match seconds')
            offsets=y-x;offset=float(offsets.min())
            jitter=float(np.quantile(offsets,.99)-offset)
            if jitter>self.max_jitter_s:
                self.report=dict(status='waiting_for_stable_delivery',
                    startup_jitter_p99_s=jitter,tick_rate_ratio=slope)
                return None
            self.mapping=ClockMapping(1.,offset,jitter,
                'DDS lower-envelope estimate; constant capture/transport latency unmeasured')
            self.report=dict(method=self.mapping.source,samples=len(samples),offset_s=offset,
                startup_jitter_p99_s=jitter,tick_rate_ratio=slope)
        # Small oscillator drift toward earlier arrival is followed; backwards
        # jumps larger than the budget are faults, not automatic clock resets.
        residual=arrival_s-self.mapping.convert(tick_s)
        if residual < -self.max_jitter_s:raise RuntimeError('Robot clock jumped forward')
        if residual<0:
            self.mapping=ClockMapping(1.,arrival_s-tick_s,self.mapping.uncertainty_s,self.mapping.source)
        return self.mapping.convert(tick_s)


def measure_camera_clock(endpoint, *, samples=16, timeout_ms=1000):
    """Map the publisher's Unix clock to this process's monotonic clock."""
    import zmq
    context=zmq.Context();socket=context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER,0);socket.setsockopt(zmq.RCVTIMEO,timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO,timeout_ms);socket.connect(endpoint)
    measurements=[]
    try:
        for i in range(samples):
            start=time.monotonic();socket.send_json(dict(clock_probe=i))
            reply=socket.recv_json();end=time.monotonic()
            if reply.get('clock_probe')!=i:raise RuntimeError('Camera clock reply mismatch')
            remote=float(reply['unix_s'])
            if not math.isfinite(remote):raise ValueError('Invalid camera clock reply')
            measurements.append((end-start,(start+end)/2-remote))
    except zmq.Again as exc:
        raise CameraClockProbeError(f'Camera clock endpoint unavailable: {endpoint}; forward its port with the depth port') from exc
    finally:socket.close();context.term()
    rtt,offset=min(measurements)
    uncertainty=rtt/2
    if uncertainty>.005:
        raise CameraClockProbeError(f'Camera clock probe too slow: best round trip {rtt*1000:.1f} ms '
                                    f'(offset uncertainty {uncertainty*1000:.1f} ms)')
    mapping=ClockMapping(1.,offset,uncertainty,'depth publisher round-trip host-clock alignment')
    mapping.convert(remote)  # Reject an excessively uncertain startup exchange.
    return mapping,dict(method=mapping.source,offset_s=offset,best_rtt_s=rtt,
        roundtrip_uncertainty_s=uncertainty,samples=samples,
        scope='host clock only; RealSense exposure timestamp relies on SDK global-time mapping')
