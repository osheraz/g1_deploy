#!/usr/bin/env python3
"""Resting-start search for reproducible lab stairs/ramps. Simulation only.

prepare builds geometry and settles starts; search runs the existing deployment
model wrapper; view opens a selected case with sim_diffusion.py. All dimensions
are metres. Results are diagnostics, never hardware safety certification.
"""
from pathlib import Path
import argparse
import json
import sys
import xml.etree.ElementTree as ET
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_OUT = ROOT/'logs/lab_recovery_9x24x36'


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def make_scene(path, spec):
    """Boxes for steps; one convex wedge for a ramp. Warp sees those same surfaces."""
    import numpy as np
    from motion_diffusion.behavioral.mujoco import _robot_xml, DEFAULT_ROBOT
    root=_robot_xml(DEFAULT_ROBOT)
    compiler=root.find('compiler');compiler.set('meshdir',str(DEFAULT_ROBOT.parent))
    world=root.find('worldbody');asset=root.find('asset')
    if asset is None:asset=ET.SubElement(root,'asset')
    def box(name,x,y,z,dx,dy,dz):
        ET.SubElement(world,'geom',name=name,type='box',pos=f'{x} {y} {z}',
            size=f'{dx/2} {dy/2} {dz/2}',rgba='.58 .47 .30 1',contype='1',conaffinity='1')
    h,w=spec['height'],spec['width'];edge=spec.get('edge',0.)
    base=spec.get('base_height',0.)
    if base:
        box('lab_lower_landing',spec['base_edge']+spec['base_length']/2,0,base/2,spec['base_length'],w,base)
    if h<=0 or w<=0:raise ValueError('Positive height and width required')
    if spec['kind']=='stairs':
        d,n=spec['tread'],spec['steps']
        if d<=0 or not isinstance(n,int) or n<1:raise ValueError('Positive tread and integer step count required')
        for i in range(n):box(f'lab_step_{i}',edge+(i+.5)*d,0,(base+(i+1)*h)/2,d,w,base+(i+1)*h)
    elif spec['kind']=='ramp':
        length=spec['length']
        if length<=0:raise ValueError('Positive ramp length required')
        # A solid convex triangular prism has exact collision and ray geometry.
        v=np.array([[edge,-w/2,0],[edge,w/2,0],[edge+length,-w/2,0],
            [edge+length,w/2,0],[edge+length,-w/2,h],[edge+length,w/2,h]])
        f=[0,2,1,1,2,3,0,4,2,1,3,5,0,1,5,0,5,4,2,4,5,2,5,3]
        f=np.asarray(f).reshape(-1,3)[:,::-1].ravel().tolist()  # Outward-facing surface normals.
        ET.SubElement(asset,'mesh',name='lab_ramp_mesh',vertex=' '.join(map(str,v.ravel())),face=' '.join(map(str,f)))
        ET.SubElement(world,'geom',name='lab_ramp',type='mesh',mesh='lab_ramp_mesh',
            rgba='.58 .47 .30 1',contype='1',conaffinity='1')
    elif spec['kind']!='flat':raise ValueError('Unknown lab geometry')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(ET.tostring(root,encoding='unicode'))
    return path


def binding(model):
    import yaml
    from motion_diffusion.behavioral.hardware.bundle import load_bundle
    from g1_deploy.deploy_diffusion import model_settings
    selected=model_settings(yaml.safe_load((ROOT/'g1_deploy/configs/diffusion.yaml').read_text()),model)
    bundle,manifest=load_bundle(selected['bundle'])
    return selected,bundle,manifest['models'][model]


def plant_config(model):
    import torch
    from omegaconf import OmegaConf
    from motion_diffusion.behavioral.hardware.bundle import asset_path
    _,bundle,row=binding(model)
    checkpoint=torch.load(asset_path(bundle,row['controller']),map_location='cpu',weights_only=True,mmap=True)
    contract=checkpoint['action_vae']['model_contract']
    pd=json.loads(asset_path(bundle,row['pd_binding']).read_text())['deployment_pd']
    robot=OmegaConf.to_container(OmegaConf.load(asset_path(bundle,row['controller_training_config'])).robot,resolve=True)
    return contract,pd,robot


def candidates(height):
    result=[]
    def add(name,spec,posture,head_angle=0.,xy=(0.,0.)):
        result.append(dict(name=name,scene=dict(dict(height=height,width=.9144),**spec),posture=posture,
                           head_angle_deg=head_angle,root_xy=list(xy)))
    for posture in ('prone','supine'):
        add(f'flat_{posture}',dict(kind='flat'),posture)
        # One broad landing, then increasingly constrained risers and side margins.
        for edge in (.05,.25,.45):
            add(f'step_{int(edge*100):02d}_{posture}',dict(kind='stairs',steps=1,tread=.6096,edge=edge),posture)
        add(f'step_wide_{posture}',dict(kind='stairs',steps=1,tread=.6096,edge=.25,width=1.8288),posture)
        add(f'step_feet_high_{posture}',dict(kind='stairs',steps=1,tread=.6096,edge=.25),posture,180.)
        add(f'step_parallel_{posture}',dict(kind='stairs',steps=1,tread=.6096,edge=.25),posture,90.)
        for width in (.9144,1.8288):
            add(f'two_steps_{width:g}m_{posture}',dict(kind='stairs',steps=2,tread=.6096,edge=.25,width=width),posture)
            add(f'raised_{width:g}m_{posture}',dict(kind='stairs',steps=1,tread=.6096,edge=.25,width=width,
                base_height=height,base_edge=-.6096,base_length=1.8288),posture)
        for length in (1.2192,1.8288):
            for angle in (0.,180.):
                add(f'ramp_{length:g}_{int(angle)}_{posture}',dict(kind='ramp',length=length,edge=-length/2),posture,angle)
        if posture=='supine':
            for angle in (0.,180.):
                add(f'ramp_wide_{int(angle)}_{posture}',dict(kind='ramp',length=1.8288,edge=-.9144,width=1.8288),posture,angle)
    return result


def prepare(output,height,case_names=None):
    import numpy as np
    import mujoco
    import torch
    from scipy.spatial.transform import Rotation
    from motion_diffusion.behavioral.mujoco import scene_model,MujocoNativePlant
    from motion_diffusion.behavioral.start_state import snapshot_start
    torch.set_num_threads(1)
    contract,pd,robot=plant_config('r8_s25_diffusion')
    catalog=[]
    selected=[c for c in candidates(height) if not case_names or c['name'] in case_names]
    if case_names and set(case_names)!={c['name'] for c in selected}:raise ValueError('Unknown case name')
    if case_names and (output/'catalog.json').exists():
        raise ValueError('Partial preparation requires a new output directory to preserve the existing catalog')
    for case in selected:
        directory=output/'cases'/case['name'];directory.mkdir(parents=True,exist_ok=True)
        path=make_scene(directory/'scene.xml',case['scene'])
        plant=MujocoNativePlant(scene_model(path),contract,pd,robot,command_limits='bounded')
        source=ROOT/'g1_deploy/config/multimodal/flat_three'/f'{case["posture"]}.json'
        preset=json.loads(source.read_text())
        q=np.array([preset['joint_positions_rad'][n] for n in plant.names])
        rot=Rotation.from_quat(preset['root_quat_xyzw'])
        head=rot.as_matrix()[:,2];yaw=np.deg2rad(case['head_angle_deg'])-np.arctan2(head[1],head[0])
        rot=Rotation.from_rotvec([0,0,yaw])*rot
        plant.reset(case['posture'],initial_joints=q,initial_quat_xyzw=rot.as_quat(),root_xy=case['root_xy'])
        obs=plant.observation();info=plant.reset_info
        requested_sign=1 if case['posture']=='prone' else -1
        terrain=set(plant.terrain_geoms.tolist())
        penetration=max([-c.dist for c in plant.data.contact if c.geom1 in terrain or c.geom2 in terrain]+[0.])
        # These are preparation diagnostics, not hardware operating limits.
        eligible=bool(info['settle_converged'] and info['passive_root_drift_100ms_m']<.02
            and info['passive_joint_drift_100ms_rad']<.15 and obs.gravity_body[0]*requested_sign>.7 and penetration<.01)
        snapshot=snapshot_start(plant,source=dict(case=case,source_pose=str(source),scope='passively settled simulation; not physical positioning validation'))
        write(directory/'start.json',snapshot)
        case.update(eligible=eligible,preparation=info,terrain_penetration_m=penetration,
            gravity_body=obs.gravity_body.tolist(),root_position_m=snapshot['root_position_m'])
        write(directory/'case.json',case);catalog.append(case)
        print(case['name'],'eligible',eligible,'gravity',np.round(obs.gravity_body,2),'drift',round(info['passive_root_drift_100ms_m'],4),flush=True)
    write(output/'catalog.json',dict(schema='hv2.lab_recovery_search.v1',height_m=height,
        assumptions='User dimensions interpreted height x tread x width: 9x24x36 inches. Wide variants use two side by side. Ramp length is proposed. No walls, cables, mat compliance or unstable blocks.',cases=catalog))


def load_policy(model):
    from motion_diffusion.behavioral.hardware.bundle import asset_path
    from motion_diffusion.behavioral.hardware.cli import load_controller
    from motion_diffusion.behavioral.hardware.steering import load_hardware_planner
    from motion_diffusion.behavioral.hardware.engines import configure_controller,configure_planner
    from motion_diffusion.behavioral.deploy import NativeRecoveryPolicy
    selected,bundle,row=binding(model)
    controller=load_controller(dict(controller_checkpoint=str(asset_path(bundle,row['controller'])),
        tracker_device='cuda:0',pd_sidecar=str(asset_path(bundle,row['pd_binding']))))
    planner_path=asset_path(bundle,row['planner'])
    planner=load_hardware_planner(planner_path,'cuda:0',seed=42,sampling_rng='paired_episode_v2',**row['reference_inference'])
    def engine(key):return dict(directory=str(ROOT/selected['exports'][key]),backend='tensorrt',precision='fp32',execution_mode='cuda_graph')
    configure_controller(controller,engine('controller'));configure_planner(planner,planner_path,engine(model))
    if not all(e.validation_report['passed'] for e in (planner.model.inference_backend,planner.model.depth_encoder.inference_backend)):
        raise ValueError('Planner golden validation failed')
    capture=controller.depth_config['capture']
    if capture!=planner.model.behavioral_config['depth_condition']['capture']:raise ValueError('Depth contracts differ')
    return NativeRecoveryPolicy(controller,planner),capture,row


def measure(directory):
    import numpy as np
    summary=json.loads((directory/'summary.json').read_text())
    if summary['status']!='duration_reached':raise ValueError('Incomplete/interrupted trial cannot be scored')
    cfg=json.loads((directory/'config.json').read_text());scene=cfg['case']['scene']
    with np.load(directory/'trajectory.npz') as f:
        upright=(f['root_position'][:,2]-f['terrain_height_m']>=.55)&(-f['gravity'][:,2]>=.75)
        dq=f['dq'];root=f['root_position'];tail=min(100,len(dq))
        stable=bool(np.all(upright[-tail:]) and np.sqrt(np.mean(dq[-tail:]**2))<.5)
        first=next((i for i in range(49,len(upright)) if upright[i-49:i+1].all()),None)
        later_fall=bool(first is not None and np.any(~upright[first+1:]))
        if scene['kind']=='ramp':
            inside=(abs(root[:,1])<=scene['width']/2)&(root[:,0]>=scene['edge'])&(root[:,0]<=scene['edge']+scene['length'])
        elif scene.get('base_height'):
            inside=(abs(root[:,1])<=scene['width']/2)&(root[:,0]>=scene['base_edge'])&(root[:,0]<=scene['base_edge']+scene['base_length'])
        else:inside=np.ones(len(root),bool)  # Lower landing is the ground, with no lower side drop.
        departed=bool(np.any(~inside))
        return dict(held_upright_1s=summary['upright_hold_1s'],standing_final_2s=stable,fall_after_rise=later_fall,
            left_support_footprint=departed,completed_without_departure=bool(stable and not departed and not later_fall),
            max_joint_speed_rad_s=float(abs(dq).max()),final_joint_speed_rms_rad_s=float(np.sqrt(np.mean(dq[-tail:]**2))),
            horizontal_excursion_m=float(np.linalg.norm(root[:,:2]-root[0,:2],axis=1).max()),
            max_lateral_excursion_m=float(abs(root[:,1]-root[0,1]).max()),
            min_root_height_above_terrain_m=float((root[:,2]-f['terrain_height_m']).min()),
            max_raw_action=summary['max_raw_action'],torque_clipped_substep_joints=int(f['torque_clipped'].sum()),
            saturated_action_ticks=summary['saturated_action_ticks'],physics_warnings=summary['physics_warnings'])


def search(a):
    import torch
    from contextlib import redirect_stdout
    from motion_diffusion.behavioral.mujoco import scene_model,MujocoNativePlant
    from g1_deploy.sim_diffusion import arguments,run_episode
    torch.set_num_threads(1)
    catalog=json.loads((a.output/'catalog.json').read_text())
    cases=[c for c in catalog['cases'] if c['eligible'] and (not a.cases or c['name'] in a.cases)]
    for model in a.models:
        policy,capture,bundle_row=load_policy(model)
        contract,pd,robot=plant_config(model)
        for case in cases:
            base=a.output/'cases'/case['name']
            for hz in (a.rates or ([2.5] if 's25' in model else [5.])):
                policy.planner.replan_every_ticks=round(50/hz)
                for seed in a.seeds:
                    suffix=f'_delay{a.camera_delay_ms:g}ms' if a.camera_delay_ms is not None else ''
                    if a.jitter_deg:suffix+=f'_jitter{a.jitter_deg:g}deg'
                    if a.seconds!=12.:suffix+=f'_{a.seconds:g}s'
                    out=a.output/'trials'/case['name']/model/f'hz{hz:g}_seed{seed}{suffix}'
                    if (out/'metrics.json').exists():
                        old=json.loads((out/'config.json').read_text())
                        if old['case']['scene']!=case['scene'] or old['seconds']!=a.seconds or old['planner']!=bundle_row['planner']:
                            raise ValueError(f'Cached trial uses different geometry/duration/model: {out}; select a new output folder')
                        if a.contact_metrics and not (out/'contact_metrics.json').exists():
                            raise ValueError(f'Cached trial lacks contact sampling: {out}; use a new seed or output folder')
                        continue
                    out.mkdir(parents=True,exist_ok=True)
                    args=arguments(['--model',model,'--backend','tensorrt','--start',case['posture'],
                        '--seconds',str(a.seconds),'--seed',str(seed),'--noise-policy','fresh','--replan-hz',str(hz)])
                    args.pose_seed=args.sensor_seed=42;args.start_offset={};args.start_jitter=0.
                    args.start_snapshot=json.loads((base/'start.json').read_text())
                    args.start_pose_jitter_deg=a.jitter_deg;args.start_pose_seed=seed
                    plant=MujocoNativePlant(scene_model(base/'scene.xml'),contract,pd,robot,command_limits='bounded')
                    contact_audit=ContactAudit(plant) if a.contact_metrics else None
                    if contact_audit:plant.physics_step_observer=contact_audit
                    sensor_capture=dict(capture)
                    if a.camera_delay_ms is not None:sensor_capture['latency_range_s']=[a.camera_delay_ms/1000]*2
                    write(out/'config.json',dict(case=case,model=model,seed=seed,replan_hz=hz,seconds=a.seconds,
                        backend='TensorRT FP32',noise='fresh',sensor_seed=42,controller=bundle_row['controller'],planner=bundle_row['planner'],
                        depth='shared training Warp renderer',timing='synchronous simulated time; does not test offboard timing'))
                    cfg=json.loads((out/'config.json').read_text())
                    cfg.update(camera_delay_ms=a.camera_delay_ms,jitter_deg=a.jitter_deg,capture=sensor_capture)
                    write(out/'config.json',cfg)
                    try:
                        with (out/'run.log').open('w') as log,redirect_stdout(log):
                            result=run_episode(policy,plant,sensor_capture,args,out,seed)
                        if result['status']=='interrupted':raise KeyboardInterrupt
                        metrics=measure(out)
                    except Exception as exc:
                        metrics=dict(error=f'{type(exc).__name__}: {exc}',held_upright_1s=False,standing_final_2s=False,
                            fall_after_rise=False,left_support_footprint=False,completed_without_departure=False,
                            max_joint_speed_rad_s=None,horizontal_excursion_m=None)
                        print('FAILED',case['name'],model,hz,seed,metrics['error'],flush=True)
                    write(out/'metrics.json',metrics)
                    if contact_audit:write(out/'contact_metrics.json',contact_audit.summary())
                    print(case['name'],model,hz,seed,'held',metrics['held_upright_1s'],'standing',metrics['standing_final_2s'],
                          'speed',metrics['max_joint_speed_rad_s'],'travel',metrics['horizontal_excursion_m'],flush=True)
        del policy
        torch.cuda.empty_cache()
    report(a.output)


class ContactAudit:
    """Read-only substep measurements; forces are simulation estimates, not hardware limits."""
    def __init__(self,plant):
        import mujoco
        self.terrain=set(plant.terrain_geoms)
        self.head={int(g) for g in plant.collisions if plant.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
            and mujoco.mj_id2name(plant.model,mujoco.mjtObj.mjOBJ_MESH,plant.model.geom_dataid[g])=='head_link'}
        self.peak=0.;self.impulse=0.;self.samples=0;self.penetration=0.
        self.peak_time=0.;self.peak_after_startup=0.;self.last_contact_time=0.

    def __call__(self,plant):
        import mujoco
        import numpy as np
        for i,c in enumerate(plant.data.contact):
            pair={int(c.geom1),int(c.geom2)}
            if not(pair&self.terrain):continue
            self.penetration=max(self.penetration,-float(c.dist))
            if pair&self.head:
                force=np.zeros(6);mujoco.mj_contactForce(plant.model,plant.data,i,force)
                normal=max(0.,float(force[0]))
                if normal>self.peak:self.peak=normal;self.peak_time=float(plant.data.time)
                if plant.data.time>=.5:self.peak_after_startup=max(self.peak_after_startup,normal)
                if normal>1.:self.last_contact_time=float(plant.data.time)
                self.impulse+=normal*plant.model.opt.timestep;self.samples+=1

    def summary(self):
        return dict(head_normal_force_peak_N=self.peak,head_normal_impulse_Ns=self.impulse,
            head_peak_time_s=self.peak_time,head_force_peak_after_05s_N=self.peak_after_startup,
            head_last_contact_above_1N_s=self.last_contact_time,
            head_contact_samples=self.samples,max_terrain_penetration_m=self.penetration,
            sampling='Every 5ms physics substep; contact solve values, no hardware injury threshold implied')


def report(output):
    from collections import defaultdict
    groups=defaultdict(list)
    for file in sorted((output/'trials').glob('*/*/*/metrics.json')):
        cfg=json.loads((file.parent/'config.json').read_text());metrics=json.loads(file.read_text())
        contact=file.parent/'contact_metrics.json'
        if contact.exists():metrics['contacts']=json.loads(contact.read_text())
        groups[(cfg['case']['name'],cfg['model'],cfg['replan_hz'],cfg.get('camera_delay_ms'),cfg.get('jitter_deg',0),cfg['seconds'])].append(dict(seed=cfg['seed'],**metrics))
    result=[]
    for (case,model,hz,delay,jitter,seconds),runs in groups.items():
        result.append(dict(case=case,model=model,hz=hz,camera_delay_ms=delay,jitter_deg=jitter,seconds=seconds,n=len(runs),standing=sum(r['standing_final_2s'] for r in runs),
            held=sum(r['held_upright_1s'] for r in runs),later_falls=sum(r['fall_after_rise'] for r in runs),
            completed=sum(r['completed_without_departure'] for r in runs),departed=sum(r['left_support_footprint'] for r in runs),
            errors=sum(bool(r.get('error')) for r in runs),
            contact_trials=sum('contacts' in r for r in runs),
            max_head_contact_N=max((r['contacts']['head_normal_force_peak_N'] for r in runs if 'contacts' in r),default=None),
            max_head_impulse_Ns=max((r['contacts']['head_normal_impulse_Ns'] for r in runs if 'contacts' in r),default=None),
            max_speed=max((r['max_joint_speed_rad_s'] for r in runs if r['max_joint_speed_rad_s'] is not None),default=None),
            max_travel=max((r['horizontal_excursion_m'] for r in runs if r['horizontal_excursion_m'] is not None),default=None),runs=runs))
    result.sort(key=lambda r:(-r['completed']/r['n'],r['later_falls']/r['n'],r['max_speed'] or float('inf'),r['max_travel'] or float('inf')))
    write(output/'results.json',result)
    lines=['# Lab recovery simulation search','','User blocks: 9x24x36 inches, interpreted as height x tread depth x width. Wide variants use adjacent blocks. Ramp lengths are proposals. No hardware certification.',
        '','Standing means all final 2s upright (pelvis >=0.55m above local terrain; upright cosine >=0.75), joint RMS speed <0.5rad/s.',
        'A one-second rise alone is not counted as a completed recovery. All sampled seeds remain in the results.',
        'Footprint checks track the pelvis, not the whole support polygon. Floor-level lower landings permit recovery on the surrounding floor.',
        'A refall means losing the stated upright criterion after first holding it for 1s; borderline height crossings can trigger it.',
        'Contact peaks are individual simulated head/terrain contact forces, not a hardware damage limit. Joint-jitter contact totals include positioning.',
        'Errors include starts that failed to settle; they count against completion. Nominal delay is the saved 0-40ms camera range.',
        '','| Case | Model | Hz | Duration s | Delay ms / jitter deg | Completion without departure/refall | Standing | Left footprint | Errors | Peak speed rad/s | Max travel m | Head peak N |',
        '|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in result:
        speed='n/a' if r['max_speed'] is None else f'{r["max_speed"]:.1f}'
        travel='n/a' if r['max_travel'] is None else f'{r["max_travel"]:.2f}'
        head='not sampled' if r['max_head_contact_N'] is None else f'{r["max_head_contact_N"]:.0f}'
        lines.append(f'| {r["case"]} | {r["model"]} | {r["hz"]} | {r["seconds"]} | {r["camera_delay_ms"]} / {r["jitter_deg"]} | {r["completed"]}/{r["n"]} | {r["standing"]}/{r["n"]} | {r["departed"]}/{r["n"]} | {r["errors"]} | {speed} | {travel} | {head} |')
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(output/'RESULTS.md',flush=True)


def render_cases(a):
    """Static pose references, not a live viewer or a rollout recording."""
    import os
    os.environ.setdefault('MUJOCO_GL','egl')
    import mujoco
    import numpy as np
    from PIL import Image
    from motion_diffusion.behavioral.mujoco import scene_model,MujocoNativePlant
    from motion_diffusion.behavioral.start_state import restore_start
    contract,pd,robot=plant_config(a.models[0])
    for name in a.cases or []:
        base=a.output/'cases'/name
        p=MujocoNativePlant(scene_model(base/'scene.xml'),contract,pd,robot,command_limits='bounded')
        restore_start(p,json.loads((base/'start.json').read_text()))
        p.model.vis.global_.offwidth=960;p.model.vis.global_.offheight=720
        camera=mujoco.MjvCamera();camera.lookat[:]=p.data.xpos[p.pelvis]+[.1,0,.1]
        camera.distance=2.5;camera.azimuth=125;camera.elevation=-28
        options=mujoco.MjvOption();options.geomgroup[:]=[1,1,0,0,0,0]
        with mujoco.Renderer(p.model,width=960,height=720) as renderer:
            renderer.update_scene(p.data,camera=camera,scene_option=options)
            Image.fromarray(renderer.render()).save(base/'start.png')
        print(base/'start.png',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','search','report','render','view'])
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--height',type=float,default=.2286)
    p.add_argument('--cases',nargs='+');p.add_argument('--models',nargs='+',default=['r8_s25_diffusion','r8_s50r_diffusion'])
    p.add_argument('--rates',nargs='+',type=float);p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--seconds',type=float,default=12.)
    p.add_argument('--camera-delay-ms',type=float,help='Fixed capture-to-delivery delay stress test; default saved 0-40ms')
    p.add_argument('--jitter-deg',type=float,default=0.,help='Joint positioning perturbation, independent of noise; prepared before policy')
    p.add_argument('--contact-metrics',action='store_true',help='Read contact forces at every physics substep in confirmation trials')
    a=p.parse_args();a.output=a.output.resolve()
    import math
    if a.height<=0 or not math.isfinite(a.height):p.error('height must be finite and positive')
    if a.seconds<2 or not math.isfinite(a.seconds):p.error('search duration must be at least 2s')
    if a.camera_delay_ms is not None and (not math.isfinite(a.camera_delay_ms) or a.camera_delay_ms<0):p.error('camera delay must be finite and nonnegative')
    if not math.isfinite(a.jitter_deg) or not 0<=a.jitter_deg<=5:p.error('joint perturbation must be between 0 and 5 degrees')
    if a.rates:
        from motion_diffusion.behavioral.hardware.bundle import replan_ticks_for_hz
        for model in a.models:
            for rate in a.rates:replan_ticks_for_hz(rate,horizon=25 if 's25' in model else 50)
    if a.action=='prepare':prepare(a.output,a.height,a.cases)
    elif a.action=='search':search(a)
    elif a.action=='report':report(a.output)
    elif a.action=='render':render_cases(a)
    else:
        if not a.cases or len(a.cases)!=1:p.error('view needs --cases CASE_NAME')
        from g1_deploy.sim_diffusion import main as simulate
        base=a.output/'cases'/a.cases[0];case=json.loads((base/'case.json').read_text())
        model=a.models[0];hz=(a.rates or ([2.5] if 's25' in model else [5.]))[0]
        simulate(['--model',model,'--backend','tensorrt','--robot',str(base/'scene.xml'),
            '--start-state',str(base/'start.json'),'--start',case['posture'],'--viewer','--fixed-camera',
            '--noise-policy','fresh','--replan-hz',str(hz),'--seed',str(a.seeds[0]),'--sensor-seed','42'])


if __name__=='__main__':main()
