"""Keep the external camera warm; save original MJPEG only after policy A.

Runs in a low-priority child. Reuses check_camera's byte-count/timestamp reader;
there is no decoding or encoding, and no camera work in the control process.
"""

if __name__ == '__main__' and not __package__:
    import _bootstrap  # System Python workers use this checkout, not the policy virtualenv.
from steerable_recovery.paths import ROOT
import argparse
from collections import deque
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time


class TriggeredCameraWriter:
    def __init__(self,directory,log):
        self.directory=Path(directory);self.log=log
        self.start_s=None;self.pending=deque(maxlen=16)
        self.stream=None;self.frames=0
        self.writing=False

    def start(self,monotonic_s):
        if self.start_s is None:self.start_s=float(monotonic_s)

    def offer(self,sequence,stamp,frame):
        self.pending.append((sequence,stamp,frame))
        if self.start_s is None:return
        while self.pending:
            seq,ts,data=self.pending.popleft()
            if ts<self.start_s:continue
            if self.stream is None:self.stream=(self.directory/'frames.mjpg').open('xb')
            self.writing=True
            try:
                self.stream.write(data)
                self.log.write(f'cap dqbuf: 0 seq: {seq} bytesused: {len(data)} ts: {ts:.6f}\n')
                self.log.flush();self.frames+=1
            finally:
                self.writing=False

    def close(self):
        if self.stream is not None:self.stream.close()


def main():
    from steerable_recovery.sensing.check_camera import Capture
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',required=True);p.add_argument('--size',required=True)
    p.add_argument('--fps',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    writer=TriggeredCameraWriter(args.output,sys.stdout)
    stop_requested=False
    def interrupt(*_):
        nonlocal stop_requested
        stop_requested=True
        # Finish a frame and its timestamp together; otherwise wake a blocked read.
        if not writer.writing:raise KeyboardInterrupt
    signal.signal(signal.SIGINT,interrupt)
    def control():
        for line in sys.stdin:
            command=json.loads(line)
            if command.get('command')=='start':writer.start(command['monotonic_s'])
        os.kill(os.getpid(),signal.SIGINT)
    capture=None
    try:
        # Stay in the recorder's process group so cleanup also owns the driver.
        capture=Capture(args.device,args.size,args.fps,start_new_session=False)
        capture.intervals=deque(maxlen=120)
        threading.Thread(target=control,daemon=True).start()
        last_status=-float('inf');reported_rate=False
        for seq,ts,frame in capture.frames():
            now=time.monotonic()
            if capture.rate_set is not None and not reported_rate:
                print(f'Frame rate set to {capture.rate_set:g} fps',flush=True);reported_rate=True
            writer.offer(seq,ts,frame)
            if stop_requested:break
            if now-last_status>=.5:
                status=dict(arrival_monotonic_s=now,capture_monotonic_s=ts,
                    ready=True,recording=writer.start_s is not None,frames=writer.frames)
                temporary=args.output/'status.tmp'
                temporary.write_text(json.dumps(status));temporary.replace(args.output/'status.json')
                last_status=now
    except KeyboardInterrupt:pass
    finally:
        if capture is not None:capture.stop()
        writer.close()


if __name__=='__main__':main()
