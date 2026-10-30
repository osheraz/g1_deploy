#!/usr/bin/env python3
"""Exercise the hardware history/queue path in MuJoCo, without robot I/O.

Accepts sim_diffusion.py options plus --start-session and --arms. The actual
AsyncNativeRecovery runs with a deterministic two-tick simulated planner delay.
This is closed-loop simulation, not a live compute or physical safety test.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from steerable_recovery.paths import ROOT
sys.path.insert(0,str(ROOT))


def run(policy,plant,capture,args,output,seed,own):
    import numpy as np
    import torch
    from motion_diffusion.inference.execution import AsyncNativeRecovery, sample_snapshot, _cpu_snapshot
    from motion_diffusion.inference.mujoco import WarpDepth
    from steerable_recovery.tools.diagnostics.startup_commands import restore_recorded_start, startup_fraction, shape_startup
    planner=policy.planner;model=planner.model;controller=policy.controller
    info=dict(H=model.H,H_z=model.H_z,F=model.F,latent_dim=model.action_dim,
        relative_attitude=planner.relative_attitude,contract=planner.contract,
        depth_config=model.behavioral_config.get('depth_condition',{}),
        replan_every_ticks=planner.replan_every_ticks,blend_ticks=planner.blend_ticks,
        sampling_contract=deepcopy(planner.sampling_contract))
    limiter=plant.command_limiter;results={};first={}
    restored_delta=limiter.max_delta.copy()
    original_delta=restored_delta/limiter.velocity_fraction
    original_release=limiter.release_after_s
    original_window=limiter.active_for_s
    for arm in own.arms:
        directory=output/arm;directory.mkdir()
        rows=restore_recorded_start(plant,own.start_session) if own.start_session else plant.reset(args.start,seed=seed)
        recorded_plan=None
        if own.recorded_first_plan:
            if not rows:raise ValueError('--recorded-first-plan needs --start-session')
            recorded_plan=next(row for row in rows if row.get('event')=='plan')['prediction']
        plant.command_limiter=limiter if arm.startswith(('bounded','rate','startup')) else None
        limiter.release_after_s=original_release if arm=='startup50' else None
        # startup50 follows today's deployment profile; bounded/rate arms
        # intentionally compare continuous bounds.
        limiter.active_for_s=original_window if arm=='startup50' else None
        if limiter:limiter.reset()
        if arm!='startup50':limiter.max_delta=original_delta*(float(arm[4:].split('_')[0])/100 if arm.startswith('rate') else 1.)
        sensor=WarpDepth(plant,capture,args.sensor_seed,device=controller.device)
        sensor.prime_stationary()
        planner.generator.manual_seed(seed)
        class Worker:
            def __init__(self):self.info=deepcopy(info);self.pending=None;self.first=True
            def submit(self,request):
                if self.pending is not None:return False
                if self.first and recorded_plan is not None:
                    prediction={k:torch.tensor(v,dtype=torch.float32) for k,v in recorded_plan.items()}
                    reply=dict(prediction=prediction,origin_tick=request['origin_tick'],generation=request['generation'],
                               sample_wall_s=0.,anchor=_cpu_snapshot(request['anchor']))
                else:
                    noise=dict(state=torch.randn(1,model.F,model.state_out_dim,device=model.device,generator=planner.generator),
                               action=torch.randn(1,model.F,model.action_dim,device=model.device,generator=planner.generator))
                    reply=sample_snapshot(planner,request,noise)
                self.pending=(request['origin_tick']+2,reply);self.first=False
                return True
            def poll(self):
                if self.pending is None or runtime.tick<self.pending[0]:return None
                result=self.pending[1];self.pending=None;return result
        class Depth:
            def prepare(self,now,config,*,current_heading,device):
                return sensor.prepare(round(now/.02),config,device,current_heading)
        runtime=AsyncNativeRecovery(controller,Worker(),max_plan_age_ticks=2)
        request=(plant.observation().q-plant.default)/plant.scale
        code=np.zeros(controller.latent_dim,np.float32);issued_count=0;trace=[];previous_target=None
        for tick in range(round(args.seconds/.02)):
            obs=plant.observation();sensor.capture_tick(tick,obs)
            result=runtime.step(obs,request,code,now_s=float(plant.data.time),depth_buffer=Depth())
            # Verify the actual preceding motor request in both consumers.
            if tick:
                np.testing.assert_allclose(runtime.tracker_history.issued[0,-1],request,atol=1e-6)
                np.testing.assert_allclose(runtime.requests[0,-1],request,atol=1e-6)
            action=(obs.q-plant.default)/plant.scale if result is None else result['requested_action'][0].cpu().numpy()
            proposed=action.copy();raw=0. if result is None else float(controller.last_unclipped_action.abs().max())
            fraction=1.
            if result is not None:
                mode=arm.removeprefix('bounded_') if not arm.startswith('rate') else (arm.split('_')[1] if '_' in arm else 'training')
                fraction=startup_fraction(issued_count,mode,own.startup_seconds)
                action=shape_startup(action,obs.q,plant.default,plant.scale,fraction)
                if issued_count==0:first[arm]=dict(q=obs.q.copy(),history=runtime.tracker_history.flatten().numpy().copy(),
                                                 code=result['native_code'].numpy().copy(),proposed=proposed.copy())
            # Hold the measured pose without shaping; the episode clock still
            # advances and does not extend a finite deployment window.
            active_limiter=plant.command_limiter
            if result is None:plant.command_limiter=None
            request,target,torque,clipped=plant.step(action)
            plant.command_limiter=active_limiter
            if previous_target is None:previous_target=obs.q
            trace.append(dict(t=tick*.02,q=obs.q,dq=obs.dq,target=target,action=request.copy(),
                proposed=proposed,target_delta=target-previous_target,raw=raw,
                upright=-obs.gravity_body[2],height=plant.data.xpos[plant.pelvis,2],
                fraction=fraction,issued=result is not None))
            previous_target=target
            if result is not None:
                code=result['native_code'][0].numpy().copy();runtime.acknowledge();issued_count+=1
            if any(w.number for w in plant.data.warning):break
        values={k:np.asarray([r[k] for r in trace]) for k in trace[0]}
        np.savez_compressed(directory/'trajectory.npz',**values)
        upright=(values['height']>=.55)&(values['upright']>=.75)
        hold=0;first_hold=None
        for i,good in enumerate(upright):
            hold=hold+1 if good else 0
            if hold>=50 and first_hold is None:first_hold=(i+1)*.02
        early=values['t']<1.;command=np.flatnonzero(values['issued'])[0]
        results[arm]=dict(first_upright_hold_s=first_hold,final_second_upright_fraction=float(upright[-50:].mean()),
            peak_dq_first_second=float(abs(values['dq'][early]).max()),peak_dq=float(abs(values['dq']).max()),
            max_raw=float(values['raw'].max()),first_target_step=float(abs(values['target_delta'][command]).max()),
            max_target_step_first_second=float(abs(values['target_delta'][early]).max()),
            issued_commands=issued_count,physics_warning=any(w.number for w in plant.data.warning),
            command_limits=None if plant.command_limiter is None else plant.command_limiter.contract(),
            matched_recorded_plan=own.recorded_first_plan)
        (directory/'summary.json').write_text(json.dumps(results[arm],indent=2)+'\n')
        print(f'{seed} {arm}: {results[arm]}',flush=True)
    reference=first[own.arms[0]]
    assert all(all(np.array_equal(item[k],reference[k]) for k in reference) for item in first.values())
    report=dict(status='duration_reached',results=results,first_inputs_identical=True,issued_feedback_verified=True,
                runtime='actual hardware AsyncNativeRecovery; simulated 40ms planner delivery',
                limitation='Shared URDF/Warp and recorded posture; depth pixels, base velocity/height and real contacts are not reconstructed')
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    plant.command_limiter=limiter
    limiter.max_delta=restored_delta
    limiter.release_after_s=original_release
    limiter.active_for_s=original_window
    return report


def main():
    extra=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    extra.add_argument('--start-session',type=Path)
    extra.add_argument('--recorded-first-plan',action='store_true')
    extra.add_argument('--startup-seconds',type=float,default=1.)
    extra.add_argument('--arms',nargs='+',choices=['training','bounded','gentle','bounded_gentle','vigor','bounded_vigor','rate25','rate50','rate10','rate25_gentle','startup50'],
                       default=['training','startup50'])
    own,remaining=extra.parse_known_args()
    sys.argv=[sys.argv[0],*remaining]
    from steerable_recovery.cli import sim as sim
    original=sim.arguments
    def arguments():
        args=original()
        if args.viewer or args.video or args.seconds is None:raise ValueError('Finite headless validation only')
        args.command_limits='bounded'
        return args
    sim.arguments=arguments
    sim.run_episode=lambda policy,plant,capture,args,output,seed,*a,**kw:run(policy,plant,capture,args,output,seed,own)
    sim.main()


if __name__=='__main__':main()
