"""LiDAR diagnostic layers for the shared Viser robot/depth display."""
from collections import deque
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[3]
from steerable_recovery.hardware.pelvis_odometry import G1Body


def display_window(samples,window_ms):
    """Select complete packets intersecting a sensor-time window in one session.

    Returns a per-point cutoff too: a packet may straddle the window boundary.
    Window zero means exactly the latest packet. No clocks are mixed here.
    """
    if not samples:return (),0
    latest=samples[-1];session=latest[0]['session']
    if window_ms==0:return (latest,),0
    end=int(latest[2].max()) if len(latest[2]) else int(latest[0]['sensor_stamp_ns'])
    cutoff=end-round(float(window_ms)*1e6)
    selected=tuple(s for s in samples if s[0]['session']==session and len(s[2]) and int(s[2].max())>cutoff)
    return selected,cutoff


class LidarView:
    def __init__(self,display,urdf_path,*,drive_robot=False,calibration=None):
        self.display=display;self.server=display.server;self.drive_robot=drive_robot
        self.body=G1Body(urdf_path);self.last=None;self._last_draw=0.;self.candidate=None
        self._cache={};self._cache_options=None;self._last_status=0.
        self.draw_times=deque(maxlen=120);self.work_ms=deque(maxlen=120)
        self.state_gaps=0;self._last_gap=None
        if calibration:
            data=json.loads(Path(calibration).read_text())
            self.candidate=np.asarray(data['rotation_lidar_to_torso'],dtype=float)
            if (self.candidate.shape!=(3,3) or not np.allclose(self.candidate.T@self.candidate,np.eye(3),atol=1e-5)
                    or not np.isclose(np.linalg.det(self.candidate),1.,atol=1e-5)):
                raise ValueError('Invalid calibration rotation')
        with self.server.gui.add_folder('LiDAR'):
            self.mode=self.server.gui.add_dropdown('Points',('All returns','Environment','Robot returns'),initial_value='All returns')
            self.max_range=self.server.gui.add_slider('Display range (m)',min=.2,max=15.,step=.1,initial_value=3.)
            self.crop=self.server.gui.add_checkbox('Compare old 135 degree crop',False)
            self.window=self.server.gui.add_dropdown('Display persistence',('100 ms rolling','Latest packet only'),initial_value='100 ms rolling')
            self.frame=self.server.gui.add_dropdown('Frame',('Pelvis origin','Sensor'),initial_value='Pelvis origin')
            self.mount=self.server.gui.add_dropdown('Mount',('Nominal','Candidate (diagnostic)') if self.candidate is not None else ('Nominal',))
            self.freeze=self.server.gui.add_checkbox('Freeze display',False)
            self.status=self.server.gui.add_markdown('Waiting for LiDAR.')
            self.coverage=self.server.gui.add_markdown('')
        self.cloud=self.server.scene.add_point_cloud('/lidar/points',points=np.zeros((0,3),np.float32),colors=np.zeros((0,3),np.uint8),point_size=.012)
        self.axes=self.server.scene.add_frame('/lidar/sensor',axes_length=.12,axes_radius=.003)

    def summary(self):
        now=time.monotonic();times=[t for t in self.draw_times if now-t<=1.]
        hz=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else 0.
        p95=float(np.percentile(self.work_ms,95)) if self.work_ms else 0.
        return f'viewer: scene_submit={hz:.1f}Hz work_p95={p95:.1f}ms missing_state_frames={self.state_gaps} (browser FPS not measured)'

    def _prepare(self,sample):
        header,raw,stamps,_=sample
        state=header.get('state')
        if state is None and self.frame.value=='Pelvis origin':return None
        ranges=np.linalg.norm(raw,axis=1)
        valid=np.isfinite(raw).all(axis=1)&(ranges>.1)&(ranges<=self.max_range.value)
        az=np.arctan2(raw[:,1],raw[:,0])
        if self.crop.value:valid&=np.abs(az)<=np.deg2rad(67.5)
        points=raw[valid].astype(float);body_mask=np.zeros(len(points),bool)
        world_lidar=np.eye(4)
        if state is not None:
            self.body.set_joints(state['q']);transform=self.body.pelvis_from_lidar().copy()
            nominal=transform.copy()
            if self.mount.value.startswith('Candidate'):
                torso_lidar=self.body.transform('mid360_link','torso_link').copy()
                torso_lidar[:3,:3]=self.candidate
                transform=self.body.transform('torso_link','pelvis')@torso_lidar
            equivalent=np.linalg.inv(nominal)@transform
            body_mask=self.body.body_mask(points@equivalent[:3,:3].T+equivalent[:3,3])
            quat=np.asarray(state['quat_wxyz'])
            world_pelvis=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            world_lidar[:3,:3]=world_pelvis@transform[:3,:3]
            world_lidar[:3,3]=world_pelvis@transform[:3,3]
        colors=np.tile(np.array([60,175,240],np.uint8),(len(points),1));colors[body_mask]=[255,90,60]
        if self.frame.value=='Pelvis origin':points=points@world_lidar[:3,:3].T+world_lidar[:3,3]
        return points.astype(np.float32),colors,stamps[valid],body_mask,az[valid],world_lidar

    def update(self,client):
        samples=client.recent()
        if not samples:return
        sample=samples[-1];header,_,_,arrival=sample;now=time.monotonic();age=now-arrival
        key=(header['session'],header['sequence']);state=header.get('state')
        missing=state is None and self.frame.value=='Pelvis origin'
        if missing and self._last_gap!=key:self.state_gaps+=1;self._last_gap=key
        if now-self._last_status>=.25:
            self._last_status=now
            self.status.content=(f"**{'STALE' if age>.5 else 'Receiving'} · {header['source']}**  \n"
                f"Received {client.hz:.1f} Hz · arrival age {age*1000:.0f} ms · {client.skipped} skipped packets  \n"
                f"{self.summary()}  \nInput: {header.get('frame','unknown')} · {header.get('units','unknown')}  \n"
                'Capture-clock accuracy unqualified. Pelvis origin fixed. 100 ms persistence is a display overlay; '
                'no translation or within-packet motion compensation.'
                +('  \n**No paired robot state: cloud hidden; coordinate frame preserved.**' if missing else ''))
        if self.freeze.value:return
        # Never reinterpret the same scene coordinates when a state packet drops.
        if missing:
            with self.server.atomic():
                self.cloud.visible=False;self.axes.visible=False
                if self.drive_robot:self.display.base_frame.visible=False
            self._cache.clear();self.last=None
            return
        if now-self._last_draw<1/60:return
        options=(self.mode.value,self.max_range.value,self.crop.value,self.frame.value,self.mount.value,self.window.value)
        draw_key=(key,options)
        if draw_key==self.last:return
        started=time.monotonic()
        if options!=self._cache_options:self._cache.clear();self._cache_options=options
        selected,cutoff=display_window(samples,100 if self.window.value.startswith('100') else 0)
        keys={(s[0]['session'],s[0]['sequence']) for s in selected}
        self._cache={k:v for k,v in self._cache.items() if k in keys}
        clouds=[];colors=[];angles=[];removed=0;total=0
        latest_transform=np.eye(4)
        for s in selected:
            k=(s[0]['session'],s[0]['sequence'])
            if k not in self._cache:self._cache[k]=self._prepare(s)
            prepared=self._cache[k]
            if prepared is None:continue
            points,col,stamps,body,az,transform=prepared
            keep=stamps>cutoff if cutoff else np.ones(len(points),bool)
            total+=int(keep.sum());removed+=int(body[keep].sum());angles.append(az[keep])
            if self.mode.value=='Environment':keep=keep&~body
            elif self.mode.value=='Robot returns':keep=keep&body
            clouds.append(points[keep]);colors.append(col[keep])
            if k==key:latest_transform=transform
        xyz=np.concatenate(clouds) if clouds else np.empty((0,3),np.float32)
        rgb=np.concatenate(colors) if colors else np.empty((0,3),np.uint8)
        with self.server.atomic():
            self.cloud.visible=True;self.axes.visible=True
            if self.drive_robot:
                self.display.base_frame.visible=state is not None and self.frame.value=='Pelvis origin'
                if state is not None:
                    msg=SimpleNamespace(tick=state['tick'],mode_pr=0,mode_machine=state['mode_machine'],
                                        imu_state=SimpleNamespace(quaternion=state['quat_wxyz']))
                    self.display.update(msg,np.asarray(state['q']))
            if self.frame.value=='Pelvis origin':
                self.axes.position=latest_transform[:3,3]
                self.axes.wxyz=Rotation.from_matrix(latest_transform[:3,:3]).as_quat()[[3,0,1,2]]
            else:self.axes.position=(0,0,0);self.axes.wxyz=(1,0,0,0)
            self.cloud.points=xyz;self.cloud.colors=rgb
        self.last=draw_key;self._last_draw=now;self.draw_times.append(time.monotonic())
        self.work_ms.append((time.monotonic()-started)*1000)
        if now-self._last_status<.02:
            counts=np.histogram(np.concatenate(angles) if angles else [],bins=np.linspace(-np.pi,np.pi,9))[0]
            self.coverage.content=(f'Displayed window: **{total}** returns · body candidates: **{removed}**  \n'
                f'Counts by 45° sector: {counts.tolist()}  \nBody masks provisional; not planner inputs.')
