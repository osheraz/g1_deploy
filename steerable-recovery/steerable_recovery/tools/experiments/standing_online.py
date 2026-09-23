"""Paired online S50R recovery/standing experiments; never opens robot I/O.

Runs full feedback physics for every arm. The unchanged-planner prefix selects
an early standing Y time; candidate prefixes must match exactly before Y.
"""
from __future__ import annotations

import argparse
from copy import copy
import json
from pathlib import Path
import time

from steerable_recovery.paths import ROOT
import numpy as np


class TargetRateFilter:
    """Prototype tested here first; only limits post-Y target slew, not torque."""
    def __init__(self, default, scale, speed, tau=0.):
        self.default = np.asarray(default); self.scale = np.asarray(scale)
        self.speed, self.tau = speed, tau

    def apply(self, action, previous):
        old = self.default+self.scale*previous
        desired = self.default+self.scale*action
        alpha = 1. if not self.tau else -np.expm1(-.02/self.tau)
        target = old+np.clip(alpha*(desired-old), -.02*self.speed, .02*self.speed)
        return ((target-self.default)/self.scale).astype(np.float32)


class QuietStand:
    """Experimental settle / posture-hold / rebalance supervisor."""
    def __init__(self,plant,fast=False):
        self.plant=plant;self.fast=fast;self.mode='settling';self.targets=[]
        self.start=None;self.quiet_since=None;self.release=None;self.transitions=[]

    def apply(self,action,previous,obs,now):
        p=self.plant;old=p.default+p.scale*previous;desired=p.default+p.scale*action
        gyro=float(np.linalg.norm(obs.gyro_body));speed=float(np.abs(obs.dq[:12]).max())
        quiet=(gyro<(.5 if self.fast else .35) and speed<(2. if self.fast else 1.)
               and -obs.gravity_body[2]>.94 and
               (not self.targets or np.abs(old-self.targets[-1]).max()<(.08 if self.fast else .05)))
        if self.mode=='holding':
            angle_cos=float(np.dot(obs.gravity_body,self.gravity)/(np.linalg.norm(obs.gravity_body)*np.linalg.norm(self.gravity)))
            moving=(gyro>.65 or speed>3. or angle_cos<np.cos(np.deg2rad(8)) or -obs.gravity_body[2]<.9)
            if moving:
                self.mode='settling';self.quiet_since=None;self.targets=[]
                self.release=(now,old.copy());self.transitions.append(dict(time=now,mode='settling'))
            else:
                u=np.clip((now-self.start)/.3,0,1);w=u*u*(3-2*u)
                target=self.from_target+w*(self.hold_target-self.from_target)
                return ((target-p.default)/p.scale).astype(np.float32)
        if quiet:
            if self.quiet_since is None:self.quiet_since=now
            self.targets.append(old.copy());self.targets=self.targets[-15:]
        else:self.quiet_since=None;self.targets=[]
        if self.quiet_since is not None and now-self.quiet_since>=(.12 if self.fast else .2):
            self.mode='holding';self.start=now;self.from_target=old.copy()
            self.hold_target=np.mean(self.targets,axis=0);self.gravity=obs.gravity_body.copy()
            self.transitions.append(dict(time=now,mode='holding'))
            return previous.copy()
        if self.release is not None:
            u=np.clip((now-self.release[0])/.1,0,1)
            if u<1:return ((self.release[1]+u*(desired-self.release[1])-p.default)/p.scale).astype(np.float32)
            self.release=None
        return action


def activation(data):
    # Same joint convention as the native29 tracker: leg blocks each have six DOFs.
    q = data['q']
    good = ((-data['gravity'][:,2] >= .92) & (np.abs(q[:,[3,9]]).max(-1)<=1.)
            & (np.abs(q[:,[0,6]]).max(-1)<=.7)
            & (data['root_position'][:,2]-data['terrain_height_m']>=.55))
    count=0
    for i, valid in enumerate(good):
        count = count+1 if valid else 0
        if count>=3:return i
    return None


def metrics(path, switch, end_s=None):
    archive=np.load(path/'trajectory.npz')
    d={k:archive[k] for k in archive.files}
    if end_s is not None:
        length=len(d['time']); keep=d['time']<end_s-1.e-9
        d={k:v[keep] if v.ndim and len(v)==length else v for k,v in d.items()}
    summary=json.loads((path/'summary.json').read_text())
    result=dict(status=summary['status'], physics_warnings=summary['physics_warnings'],
                switch_s=switch, seconds=float(d['time'][-1]+.02))
    if switch is None:return result
    at=int(round(switch/.02)); late=d['time']>=switch+2
    qpos=d['root_position']; ground=d['terrain_height_m']
    upright=-d['gravity'][:,2]
    fell=(upright[at:]<.6) | (qpos[at:,2]-ground[at:]<.35)
    result.update(fell_after_y=bool(fell.any()),
        min_upright=float(upright[at:].min()),
        max_target_step_rad=float(np.abs(np.diff(d['target'][max(0,at-1):],axis=0)).max()),
        max_speed_rad_s=float(np.abs(d['dq'][at:]).max()),
        peak_torque_nm=float(d['torque_peak'][at:].max()),
        torque_clipped=int(d['torque_clipped'][at:].sum()),
        final_height_m=float(qpos[-1,2]-ground[-1]), final_upright=float(upright[-1]),
        support_height_at_y_m=float(ground[at]),support_height_at_end_m=float(ground[-1]),
        pelvis_at_y_m=qpos[at].tolist(),pelvis_at_end_m=qpos[-1].tolist(),
        drift_cm=float(100*np.linalg.norm(qpos[-1,:2]-qpos[at,:2])))
    if late.any():
        result.update(late_peak_speed_rad_s=float(np.abs(d['dq'][late]).max()),
            late_mean_speed_rad_s=float(np.abs(d['dq'][late]).mean()),
            late_xy_span_cm=(100*np.ptp(qpos[late,:2],axis=0)).tolist(),
            late_heading_span_deg=float(np.rad2deg(np.ptp(np.unwrap(d['causal_heading_rad'][late])))),
            late_peak_target_step_rad=float(np.abs(np.diff(d['target'][late],axis=0)).max()),
            quiet_at_end=bool((upright[-50:]>.9).all() and
                             (np.abs(d['dq'][-50:]).max()<.5) and not fell.any()))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms',nargs='+',default=['encoded'],choices=[
        'encoded','continue','rate2','rate4','rate6','rate4_filter','captured_rate4',
        'quiet_hold','fast_hold','damp1.25','damp1.5','damp2','damp3_ankle',
        'velocity_adaptive','velocity1','slow1'])
    parser.add_argument('--friction',type=float,default=1.)
    parser.add_argument('--push-speed',type=float,default=0.,help='Explicit test-only pelvis XY velocity increment 0.5 s after Y')
    parser.add_argument('--quick',action='store_true',help='End branches six seconds after Y')
    parser.add_argument('--activation-delay-s',type=float,default=0.,help='Test a later manual Y after the first standing condition')
    args,simargs=parser.parse_known_args()
    if not np.isfinite(args.activation_delay_s) or args.activation_delay_s<0:
        parser.error('Activation delay must be finite and nonnegative')
    if not np.isfinite([args.friction,args.push_speed]).all() or min(args.friction,args.push_speed)<0:
        parser.error('Friction and push speed must be finite and nonnegative')
    import torch
    torch.set_num_threads(1)
    # Keep alternative reference constructions out of the hardware path.
    from motion_diffusion.behavioral import standing
    from .standing_reference_variants import ExperimentalStandingReference
    standing.StandingReference=ExperimentalStandingReference
    from steerable_recovery.cli import sim
    original=sim.run_episode
    campaign=[]
    def paired(policy,plant,capture,options,output,seed,viewer=None,controls=None):
        if viewer:raise ValueError('Headless paired experiment only')
        if options.model not in ('s50r_diffusion','s50r_causal_diffusion'):
            raise ValueError('Select the S50R or S50R causal pair explicitly')
        reference=policy.standing_reference
        base_config=json.loads((output/'config.json').read_text())
        plant.model.geom_friction[:,0]=args.friction
        base=output/'planner';base.mkdir()
        baseargs=copy(options);baseargs.standing_hold_at=None;baseargs.standing_hold_on_upright=False
        del policy.standing_reference
        config=dict(base_config,standing_hold=dict(enabled=False),
                    experiment=dict(friction=args.friction,push_speed_m_s=0.))
        (base/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        print(f'BASELINE {output.name}, seed={seed}, friction={args.friction}',flush=True)
        began=time.monotonic()
        baseline=original(policy,plant,capture,baseargs,base,seed)
        data=np.load(base/'trajectory.npz');index=activation(data)
        if index is not None:
            index+=int(round(args.activation_delay_s/.02))
            if index>=len(data['time']):index=None
        switch=None if index is None else float(data['time'][index])
        common_end=None if switch is None else min(options.seconds,switch+6.) if args.quick else options.seconds
        item=dict(model=options.model,seed=seed,posture=options.start,scene=base_config['scene'],output=str(output),
            friction=args.friction,push_speed_m_s=args.push_speed,switch_s=switch,
            results=dict(planner=metrics(base,switch,common_end)))
        if switch is not None and switch<=options.seconds-3:
            for arm in args.arms:
                dest=output/arm;dest.mkdir()
                selected=copy(options);selected.standing_hold_at=switch;selected.standing_hold_on_upright=False
                if args.quick:selected.seconds=min(options.seconds,switch+6.)
                reference.pose_mode='captured' if arm.startswith('captured') else 'default'
                reference.adaptive_transition=arm=='velocity_adaptive'
                reference.blend_s=1. if arm in ('velocity1','slow1') else .5
                original_activate=reference.activate
                if arm.startswith('velocity'):
                    def activate(*a,**kw):
                        import torch
                        kw['dq']=torch.as_tensor(plant.observation().dq,device=policy.controller.device)[None]
                        return original_activate(*a,**kw)
                    reference.activate=activate
                policy.standing_reference=reference
                reference.reset()
                if arm=='continue':
                    del policy.standing_reference
                    selected.standing_hold_at=None
                speed=None
                if 'rate' in arm:speed=float(arm.rsplit('rate',1)[1].split('_')[0])
                tau=.04 if arm.endswith('_filter') else 0.
                filter_=None if speed is None else TargetRateFilter(plant.default,plant.scale,speed,tau)
                quiet_hold=QuietStand(plant,fast=arm=='fast_hold') if arm in ('quiet_hold','fast_hold') else None
                previous=np.zeros(29,np.float32);last=plant.step; pushed=False
                base_kd=plant.kd.copy()
                damping=np.ones(29)
                if arm.startswith('damp'):
                    factor=float(arm[4:].split('_')[0])
                    damping[[4,5,10,11] if arm.endswith('_ankle') else list(range(15))]=factor
                import mujoco
                contact=dict(peak_normal_force_n=0.,normal_impulse_sum_ns=0.)
                force=np.zeros(6)
                old_observer=getattr(plant,'physics_step_observer',None)
                def observer(p):
                    if old_observer is not None:old_observer(p)
                    if p.data.time<switch:return
                    for ci in range(p.data.ncon):
                        c=p.data.contact[ci]
                        bodies=p.model.geom_bodyid[[c.geom1,c.geom2]]
                        if (bodies==0).sum()!=1:continue
                        mujoco.mj_contactForce(p.model,p.data,ci,force)
                        contact['peak_normal_force_n']=max(contact['peak_normal_force_n'],float(abs(force[0])))
                        contact['normal_impulse_sum_ns']+=float(abs(force[0])*p.model.opt.timestep)
                plant.physics_step_observer=observer
                def step(action):
                    nonlocal previous,pushed
                    if reference.active and arm.startswith('damp'):
                        u=np.clip((plant.data.time-switch)/.5,0,1);w=u*u*(3-2*u)
                        plant.kd=base_kd*(1+w*(damping-1))
                    if reference.active and filter_ is not None:action=filter_.apply(action,previous)
                    if reference.active and quiet_hold is not None:
                        action=quiet_hold.apply(action,previous,plant.observation(),float(plant.data.time))
                    if args.push_speed and not pushed and plant.data.time>=switch+.5:
                        plant.data.qvel[seed%2]+=args.push_speed*(1 if seed%3 else -1)
                        pushed=True
                    result=last(action);previous=result[0].copy();return result
                plant.step=step
                config=dict(base_config,standing_hold=dict(enabled=arm!='continue',at_s=switch,pose=reference.pose_mode,
                    blend_s=reference.blend_s,velocity_matched=arm.startswith('velocity'),
                    adaptive_transition=reference.adaptive_transition,
                    speed_limit_rad_s=speed,filter_tau_s=tau),
                    experiment=dict(friction=args.friction,push_speed_m_s=args.push_speed,
                        kd_multiplier=damping.tolist(),damping_ramp_s=.5,
                        activation_delay_s=args.activation_delay_s))
                (dest/'config.json').write_text(json.dumps(config,indent=2)+'\n')
                print(f'BRANCH {output.name}/{arm}: Y={switch:.2f}s',flush=True)
                try:original(policy,plant,capture,selected,dest,seed)
                finally:
                    plant.step=last
                    plant.kd=base_kd
                    reference.activate=original_activate
                    plant.physics_step_observer=old_observer
                candidate=np.load(dest/'trajectory.npz')
                errors={key:float(np.abs(data[key][:index]-candidate[key][:index]).max())
                        for key in ('target','qpos','native_code')}
                if any(v>1.e-5 for v in errors.values()):raise AssertionError(f'Unmatched prefix: {errors}')
                result=metrics(dest,switch);result['prefix_errors']=errors;result.update(contact)
                if arm.startswith('damp'):result['kd_multiplier']=damping.tolist()
                if quiet_hold is not None:result['transitions']=quiet_hold.transitions
                item['results'][arm]=result
                print('RESULT '+json.dumps(dict(seed=seed,arm=arm,**result)),flush=True)
        else:item['skipped_reason']='Did not reach Y condition with >=3 s evaluation remaining'
        policy.standing_reference=reference
        item['wall_s']=time.monotonic()-began;campaign.append(item)
        (output.parent/'standing_comparison.json').write_text(json.dumps(campaign,indent=2)+'\n')
        print(f'CASE DONE: {item["wall_s"]:.1f} s wall; {len(campaign)} cases completed',flush=True)
        # A warned baseline is a recorded failed case, not a reason to skip other seeds.
        return dict(baseline,status='duration_reached')
    sim.run_episode=paired
    sim.main(simargs+['--standing-hold'])


if __name__=='__main__':main()
