"""Read-only Viser view of measured_pose.jsonl (live tail or offline replay).

Reset changes the DISPLAY episode origin only. It never sends commands, resets
DLIO, or changes the deployment episode. Shared training heading functions are
used at 50 Hz; the actual runtime remains responsible for its control-tick phase.
"""

if __name__ == '__main__' and not __package__:
    import _bootstrap  # System Python workers use this checkout, not the policy virtualenv.
from steerable_recovery.paths import ROOT
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(ROOT))
from humanoidverse2.observations.heading import initialize_heading,transport_heading


class MotionDisplayState:
    def __init__(self):self.reset()

    def reset(self):
        self.heading=self.previous=self.origin=self.initial_heading=None
        self.started=None;self.trace=[]

    def update(self, packet):
        p=np.asarray(packet.get('position_m'),dtype=float)
        q=np.asarray(packet.get('imu_quat_xyzw'),dtype=float)
        if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(p).all() or not np.isfinite(q).all() or np.linalg.norm(q)<1e-6:
            return None
        quat=torch.from_numpy(q/np.linalg.norm(q))
        self.heading=(initialize_heading(quat) if self.heading is None else
                      transport_heading(self.heading,self.previous,quat))
        self.previous=quat
        if self.origin is None:
            self.origin=p.copy();self.initial_heading=float(self.heading)
            self.started=packet['capture_monotonic_s']
        c,s=np.cos(self.initial_heading),np.sin(self.initial_heading)
        d=p-self.origin;relative=np.array([c*d[0]+s*d[1],-s*d[0]+c*d[1],d[2]])
        angle=float(self.heading)-self.initial_heading
        yaw=np.arctan2(np.sin(angle),np.cos(angle))
        self.trace.append(relative);self.trace=self.trace[-15000:]
        return relative,yaw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('packets',type=Path);p.add_argument('--replay',action='store_true')
    p.add_argument('--port',type=int,default=8081);p.add_argument('--duration-s',type=float,default=0.)
    p.add_argument('--no-viser',action='store_true',help='Console validation only')
    a=p.parse_args()
    view=MotionDisplayState();server=None;reset_requested=False
    if not a.no_viser:
        import viser
        server=viser.ViserServer(port=a.port)
        status=server.gui.add_markdown('Waiting for measured pose...')
        reset=server.gui.add_button('Reset display episode')
        @reset.on_click
        def request_reset(_):
            nonlocal reset_requested
            reset_requested=True
        server.scene.add_grid('/ground',width=4,height=4)
    print('Measured motion only. Display reset does not reset the robot or deployment.',flush=True)
    source=None;latest=None;pending=None;start=time.monotonic();replay_origin=None;last_print=0.;last_capture=None
    try:
        while not a.duration_s or time.monotonic()-start<a.duration_s:
            tick_start=time.monotonic()
            if source is None:
                if not a.packets.exists():time.sleep(.1);continue
                source=a.packets.open()
            while True:
                if pending is None:
                    offset=source.tell();line=source.readline()
                    if not line:break
                    if not line.endswith('\n'):source.seek(offset);break
                    row=json.loads(line)
                    if not np.isfinite(row.get('capture_monotonic_s',np.nan)):continue
                    pending=row
                if a.replay:
                    stamp=pending['capture_monotonic_s']
                    if replay_origin is None:replay_origin=stamp;start=time.monotonic()
                    if stamp-replay_origin>time.monotonic()-start:break
                latest=pending;pending=None
            if reset_requested:
                view.reset();reset_requested=False;last_capture=None
            if latest is not None:
                capture=latest['capture_monotonic_s']
                out_of_order=last_capture is not None and capture<last_capture
                current=(replay_origin+time.monotonic()-start) if a.replay else time.monotonic()
                age=current-capture
                result=view.update(latest) if last_capture!=capture and not out_of_order else None
                if not out_of_order:last_capture=capture
                if result is not None:
                    relative,yaw=result
                    good=latest.get('tracking_valid') is True and 0<=age<=.05
                    color=(40,170,70) if good else (230,120,30)
                    if server:
                        server.scene.add_point_cloud('/pelvis',points=relative[None].astype(np.float32),colors=color,point_size=.06)
                        trace=np.array(view.trace,dtype=np.float32)
                        if len(trace)>1:server.scene.add_line_segments('/trajectory',points=np.stack([trace[:-1],trace[1:]],axis=1),colors=color,line_width=3.)
                    message=(f'XYZ since display reset: {relative.round(3).tolist()} m | heading {np.degrees(yaw):.1f} deg\n'
                             f'Age {age*1000:.1f} ms | geometry {latest.get("geometry_valid")} | runtime qualified {latest.get("tracking_valid")}\n'
                             f'DLIO/IMU yaw disagreement {np.degrees(latest.get("yaw_disagreement_rad",np.nan)):.1f} deg')
                elif age>.05:message=f'STALE: no new usable measured pose for {age:.2f} s'
                else:message='Waiting for next usable measured pose: '+str(latest.get('invalid_reason',''))
                if tick_start-last_print>=1:
                    print(message,flush=True)
                    if server:status.content=message.replace('\n','\n\n')
                    last_print=tick_start
            time.sleep(max(0.,.02-(time.monotonic()-tick_start)))
    except KeyboardInterrupt:pass
    finally:
        if source:source.close()
        if server:server.stop()


if __name__=='__main__':main()
