#!/usr/bin/env python3
"""Look at the external camera through the deployment recorder's own capture path.

Live view (Viser, port 8080): the same v4l2-ctl command the recorder launches, streamed
to this process instead of a file; frames are cut with the driver's byte counts and
shown with their sequence number, cadence and age (now minus start of exposure).
Viser controls: capture mode (restarts the camera), display scale, JPEG quality,
frame skipping and background-versus-panel display. Display settings never affect
what the recorder saves.

  .venv/bin/python g1_deploy/steerable-recovery/steerable_recovery/sensing/check_camera.py                      # default camera, 1280x720@30
  .venv/bin/python g1_deploy/steerable-recovery/steerable_recovery/sensing/check_camera.py --list-modes
  .venv/bin/python g1_deploy/steerable-recovery/steerable_recovery/sensing/check_camera.py --size 1920x1080

Timed capture: the recorder child itself (file output, nice +10) for N seconds, then
the same post-stop report the deployment writes to recording_result.json.

  .venv/bin/python g1_deploy/steerable-recovery/steerable_recovery/sensing/check_camera.py --seconds 10 --output logs/real_world/camera_check
"""

if __name__ == '__main__' and not __package__:
    import _bootstrap  # System Python workers use this checkout, not the policy virtualenv.
from steerable_recovery.paths import ROOT
import argparse
import json
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from steerable_recovery.recording.session import CAMERA_FRAME_PATTERN, CAMERA_RATE_PATTERN, DEFAULT_CAMERA, DEFAULT_WS, RecorderProcesses, camera_command, camera_modes, camera_report

SOI=b'\xff\xd8'


class StdoutBuffer:
    """Drains the child's stdout continuously; v4l2-ctl logs a frame only after writing it,
    so the reader must never wait for the log line before consuming the pipe."""
    def __init__(self,stream):
        self.buffer=bytearray();self.eof=False;self.cond=threading.Condition()
        threading.Thread(target=self._drain,args=(stream,),daemon=True).start()

    def _drain(self,stream):
        while True:
            chunk=stream.read1(1<<16)
            with self.cond:
                if not chunk:self.eof=True;self.cond.notify_all();return
                self.buffer.extend(chunk);self.cond.notify_all()

    def skip_to(self,marker):
        """Discard and return the text v4l2-ctl prints on stdout before the first frame."""
        with self.cond:
            while (index:=self.buffer.find(marker))<0 and not self.eof:self.cond.wait()
            if index<0:return None
            head=bytes(self.buffer[:index]);del self.buffer[:index];return head

    def take(self,count):
        with self.cond:
            while len(self.buffer)<count and not self.eof:self.cond.wait()
            if len(self.buffer)<count:return None
            frame=bytes(self.buffer[:count]);del self.buffer[:count];return frame


class Capture:
    """One recorder-identical capture child streaming to this process."""
    def __init__(self,device,size,fps,*,start_new_session=True):
        self.proc=subprocess.Popen(camera_command(device,size,fps,'-'),stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,start_new_session=start_new_session)
        self.meta=queue.Queue();self.rate_set=None
        threading.Thread(target=self._lines,daemon=True).start()
        self.stdout=StdoutBuffer(self.proc.stdout)
        self.count=0;self.dropped=0;self.last_seq=None;self.last_ts=None;self.intervals=[]

    def _lines(self):
        for line in self.proc.stderr:
            text=line.decode(errors='replace')
            rate=re.search(CAMERA_RATE_PATTERN,text)
            if rate:self.rate_set=float(rate[1])
            match=re.search(CAMERA_FRAME_PATTERN,text)
            if match:self.meta.put((int(match[1]),int(match[2]),float(match[3])))
        self.meta.put(None)

    def frames(self):
        head=self.stdout.skip_to(SOI)  # v4l2-ctl prints its format report and the rate it set on stdout first
        if head is None:return
        rate=re.search(CAMERA_RATE_PATTERN,head.decode(errors='replace'))
        if rate:self.rate_set=float(rate[1])
        while True:
            item=self.meta.get()
            if item is None:return
            seq,size,ts=item
            frame=self.stdout.take(size)
            if frame is None:return
            self.count+=1
            if self.last_seq is not None:self.dropped+=seq-self.last_seq-1
            if self.last_ts is not None:self.intervals.append(ts-self.last_ts)
            self.last_seq,self.last_ts=seq,ts
            yield seq,ts,frame

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:self.proc.kill()

    def summary(self):
        recent=np.array(self.intervals[1:]) if len(self.intervals)>1 else np.array([])
        return dict(frames=self.count,dropped=self.dropped,rate_set_fps=self.rate_set,
            mean_interval_ms=None if not len(recent) else float(recent.mean()*1000),
            max_interval_ms=None if not len(recent) else float(recent.max()*1000))


def useful_modes(device):
    """At least 1280x720 and at least 30 fps; smaller or slower modes are not worth recording."""
    return [(w,h,f) for w,h,f in camera_modes(device) if w>=1280 and h>=720 and f>=30]


def live(args):
    import cv2
    import viser
    modes=useful_modes(args.device)
    labels=[f'{w}x{h}@{int(f) if f==int(f) else f}' for w,h,f in modes]
    initial=f'{args.size}@{args.fps}'
    if initial not in labels:raise SystemExit(f'{initial} is not offered at >=1280x720 and >=30 fps; --list-modes shows the options')
    server=viser.ViserServer(port=args.port)
    mode=server.gui.add_dropdown('capture mode (restarts camera)',labels,initial_value=initial)
    display=server.gui.add_dropdown('display',['background','panel'],initial_value='background')
    scale=server.gui.add_slider('display scale',0.1,1.0,0.05,1.0)
    quality=server.gui.add_slider('display JPEG quality',20,100,5,args.jpeg_quality)
    every=server.gui.add_slider('show every Nth frame',1,10,1,args.every)
    server.gui.add_markdown('Display settings only change this preview; the recorder saves the raw driver frames.')
    status=server.gui.add_text('frames',' ',disabled=True)
    cadence=server.gui.add_text('cadence',' ',disabled=True)
    age=server.gui.add_text('age (now - exposure)',' ',disabled=True)
    panel={'handle':None}
    current={'capture':None,'restart':threading.Event(),'stop':threading.Event()}
    mode.on_update(lambda _:current['restart'].set())

    def show(image):
        if scale.value<1.0:
            image=cv2.resize(image,None,fx=float(scale.value),fy=float(scale.value),interpolation=cv2.INTER_AREA)
        rgb=image[...,::-1]
        if display.value=='background':
            if panel['handle'] is not None:panel['handle'].remove();panel['handle']=None
            server.scene.set_background_image(rgb,format='jpeg',jpeg_quality=int(quality.value))
        else:
            server.scene.set_background_image(None)
            if panel['handle'] is None:panel['handle']=server.gui.add_image(rgb,format='jpeg',jpeg_quality=int(quality.value))
            else:panel['handle'].image=rgb

    def run():
        while not current['stop'].is_set():
            width,height,fps=re.fullmatch(r'(\d+)x(\d+)@([\d.]+)',mode.value).groups()
            capture=Capture(args.device,f'{width}x{height}',int(float(fps)));current['capture']=capture
            current['restart'].clear()
            print(f'Capturing {mode.value} from {args.device}',flush=True)
            for seq,ts,frame in capture.frames():
                if current['restart'].is_set() or current['stop'].is_set():break
                if capture.count%int(every.value):continue
                image=cv2.imdecode(np.frombuffer(frame,np.uint8),cv2.IMREAD_COLOR)
                if image is None:continue
                show(image)
                recent=capture.intervals[-30:]
                rate=f', driver set {capture.rate_set:g} fps' if capture.rate_set else ''
                status.value=f'{capture.count} frames, seq {seq}, {capture.dropped} dropped{rate}'
                cadence.value=f'{1/np.mean(recent):.1f} Hz, max gap {max(recent)*1000:.1f} ms' if recent else ' '
                age.value=f'{(time.monotonic()-ts)*1000:.0f} ms'
            capture.stop()
            print(json.dumps(capture.summary()),flush=True)
            if not current['restart'].is_set():break

    worker=threading.Thread(target=run,daemon=True);worker.start()
    print(f'Viser on port {args.port}. Ctrl+C stops.',flush=True)
    try:
        while worker.is_alive():time.sleep(.2)
    except KeyboardInterrupt:
        current['stop'].set()
        if current['capture'] is not None:current['capture'].stop()
        worker.join(timeout=5)


def timed(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False);(output/'camera').mkdir()
    plan=[('camera',camera_command(args.device,args.size,args.fps,str(output/'camera/frames.mjpg')))]
    processes=RecorderProcesses()
    processes.launch(plan,output,Path(DEFAULT_WS))
    started=time.monotonic();marker=None
    try:
        while time.monotonic()-started<args.seconds:
            time.sleep(.5);processes.check()
            if marker is None and time.monotonic()-started>=args.seconds/2:marker=time.monotonic()
    except KeyboardInterrupt:
        pass
    processes.close()
    report=camera_report(output,args.fps,None,marker)
    (output/'camera_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=1),flush=True)
    print(f'Raw dump, index and remuxed video under {output}/camera; driver log {output}/camera.log',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--device',default=DEFAULT_CAMERA)
    p.add_argument('--size',default='1920x1080');p.add_argument('--fps',type=int,default=30)
    p.add_argument('--list-modes',action='store_true',help='print the camera MJPG modes at >=1280x720 and >=30 fps, then exit')
    p.add_argument('--seconds',type=float,default=0.,help='timed capture through the recorder child instead of the live view')
    p.add_argument('--output',type=Path,help='timed capture directory (must not exist)')
    p.add_argument('--every',type=int,default=1,help='live: display every Nth frame')
    p.add_argument('--jpeg-quality',type=int,default=80)
    p.add_argument('--port',type=int,default=8080)
    args=p.parse_args()
    if not Path(args.device).exists():p.error(f'{args.device} not found')
    if args.list_modes:
        for w,h,f in useful_modes(args.device):print(f'{w}x{h} @ {f:g} fps')
        return
    if args.seconds:
        if args.output is None:p.error('--seconds needs --output')
        timed(args)
    else:live(args)


if __name__=='__main__':main()
