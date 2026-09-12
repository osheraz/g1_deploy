#!/usr/bin/env python3
"""Compare native diffusion/full-horizon and flow/replanning on flat MuJoCo.

python g1_deploy/sim_diffusion.py --model both --video
python g1_deploy/sim_diffusion.py --model diffusion --viewer
python g1_deploy/sim_diffusion.py --model flow --viewer --start supine
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import json
import os
import queue
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def arguments():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model',choices=('diffusion','flow','both'),default='both')
    parser.add_argument('--bundle',type=Path,help='Defaults to configs/diffusion.yaml bundle')
    parser.add_argument('--mjcf',type=Path,help='Defaults to the repository SONIC G1-29 MJCF')
    parser.add_argument('--seconds',type=float,default=10.,help='Simulated seconds per model/episode')
    parser.add_argument('--episodes',type=int,default=1)
    parser.add_argument('--start',choices=('standing','prone','supine','left_side','right_side','seated','random'),default='supine')
    parser.add_argument('--postures',nargs='+',choices=('standing','prone','supine','left_side','right_side','seated','random'),
                        help='Headless sweep: run --episodes seeds for each posture')
    parser.add_argument('--joint-jitter',type=float,default=0.,help='Seeded joint perturbation amplitude in radians, at most .3')
    parser.add_argument('--backend',choices=('pytorch','tensorrt','onnx'),default='tensorrt')
    parser.add_argument('--planner-precision',choices=('fp32','tf32'),default='fp32')
    parser.add_argument('--engine-mode',choices=('standard','buffered','cuda_graph'),default='cuda_graph')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--pose-seed',type=int,help='Fix posture/joint randomness independently of planner seed')
    parser.add_argument('--sensor-seed',type=int,help='Fix camera delivery randomness independently of planner seed')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--viewer',action='store_true',help='Interactive MuJoCo window; requires a display')
    parser.add_argument('--video',action='store_true',help='Save an MP4 per run; supports headless EGL')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not 0<args.seconds<=120 or args.episodes<1:parser.error('Use 0 < seconds <= 120 and positive episodes')
    if not 0<=args.joint_jitter<=.3:parser.error('--joint-jitter must be between 0 and .3 radians')
    if args.backend!='tensorrt' and args.planner_precision!='fp32':parser.error('--planner-precision tf32 requires TensorRT')
    if args.viewer and args.postures:parser.error('Use posture keys in the viewer; --postures is a headless sweep')
    if args.viewer and not os.environ.get('DISPLAY') and sys.platform.startswith('linux'):
        parser.error('--viewer needs DISPLAY; use --video for a headless SSH session')
    return args


class ViewerControls:
    """The GUI thread only queues events; the control thread owns resets."""
    def __init__(self):self.events=queue.SimpleQueue()
    def key(self,keycode):
        if keycode in (ord('R'),ord('N'),ord('M'),32,259,*range(ord('1'),ord('8'))):
            self.events.put(keycode)
    def poll(self):
        try:return self.events.get_nowait()
        except queue.Empty:return None


def run_episode(policy,plant,capture,args,output,seed,viewer=None,controls=None):
    import mujoco
    import numpy as np
    import torch
    from humanoidverse2.agents.action_vae.heading import initialize_heading,transport_heading
    from motion_diffusion.behavioral.mujoco import MujocoDepth
    pose_seed=getattr(args,'pose_seed',None)
    sensor_seed=getattr(args,'sensor_seed',None)
    pose_seed=seed if pose_seed is None else pose_seed
    sensor_seed=seed if sensor_seed is None else sensor_seed
    with viewer.lock() if viewer else nullcontext():
        plant.reset(args.start,seed=pose_seed,joint_jitter=args.joint_jitter)
    if viewer:viewer.sync()
    sensor=MujocoDepth(plant,capture,sensor_seed) if capture else None
    print(f'  Reset: {plant.reset_info}',flush=True)
    device=policy.controller.device
    previous=torch.zeros(1,29,device=device);code=torch.zeros(1,policy.controller.latent_dim,device=device)
    heading=previous_quat=None
    frames=[];replans=[];plans=[];status='duration_reached'
    began=time.monotonic()
    with ExitStack() as stack:
        renderer=writer=None
        camera=mujoco.MjvCamera();camera.distance=2.5;camera.azimuth=135;camera.elevation=-18
        if viewer:
            viewer.set_texts((None,None,f'{args.start} | seed {seed}\n'
                'R reset | N new seed | SPACE pause | M next model\n'
                '1 stand  2 front  3 back  4 left  5 right  6 seated  7 random',None))
        if args.video:
            import imageio.v2 as imageio
            with viewer.lock() if viewer else nullcontext():
                plant.model.vis.global_.offwidth=960;plant.model.vis.global_.offheight=720
            renderer=mujoco.Renderer(plant.model,height=720,width=960);stack.callback(renderer.close)
            writer=imageio.get_writer(str(output/'rollout.mp4'),fps=50,codec='libx264',quality=7)
            stack.callback(writer.close)
        render_options=mujoco.MjvOption();render_options.geomgroup[:]=[1,1,0,0,0,0]
        try:
            tick=0;paused=False;halted=False;end_announced=False
            while tick<round(args.seconds/.02) or viewer:
                if viewer and not viewer.is_running():status='viewer_closed';break
                # A reset through MuJoCo's built-in UI also resets policy history.
                if abs(float(plant.data.time)-tick*.02)>1e-6:
                    status='reset_requested';break
                event=controls.poll() if controls else None
                if event==32:paused=not paused
                elif event is not None:
                    status='next_model' if event==ord('M') else 'reset_requested'
                    if event==ord('N'):args.next_seed=seed+1
                    if ord('1')<=event<=ord('7'):
                        from motion_diffusion.behavioral.mujoco import START_POSTURES
                        args.start=START_POSTURES[event-ord('1')]
                    break
                if viewer and (paused or halted or tick>=round(args.seconds/.02)):
                    if not paused and not end_announced:
                        viewer.set_texts((None,None,f'Episode stopped: {args.start} | seed {seed}\n'
                                          'R reset | N new seed | 1-7 posture | M next model',None))
                        end_announced=True
                    viewer.sync();time.sleep(.01);continue
                wall_start=time.monotonic()
                obs=plant.observation()
                quat=torch.tensor(obs.quat_xyzw,device=device).reshape(1,4)
                heading=initialize_heading(quat) if heading is None else transport_heading(heading,previous_quat,quat)
                previous_quat=quat.clone()
                sensor_args={};availability=[]
                if sensor:
                    sensor.capture_tick(tick,obs)
                    for prefix,config in (('',policy.planner.model.behavioral_config.get('depth_condition',{})),
                                          ('tracker_',policy.controller.depth_config)):
                        if not config.get('enabled'):continue
                        depth=sensor.prepare(tick,config,device,heading)
                        sensor_args.update({prefix+'depth_'+key:value for key,value in depth.items()})
                        if not prefix:availability=depth['available'][0].cpu().numpy()
                result=policy.step(obs.q[None],obs.dq[None],obs.gyro_body[None],obs.gravity_body[None],previous,
                    pelvis_quat=quat,previous_executed_code=code,reset_mask=torch.tensor([tick==0],device=device),
                    control_tick=torch.tensor([tick],device=device),episode_seeds=[seed],**sensor_args)
                raw_max=float(policy.controller.last_unclipped_action.abs().max())
                action=result['requested_action'][0].cpu().numpy()
                replanned=bool(result['replanned'][0])
                if replanned:
                    replans.append(tick)
                    plan=dict(tick=tick,state=result['predicted_state'][0].cpu().numpy().tolist())
                    if hasattr(policy,'planner'):
                        plan['native_code']=policy.planner.last_predicted_codes[0].cpu().numpy().tolist()
                    plans.append(plan)
                if tick==0 and hasattr(policy,'planner'):
                    # Freeze the actual condition/noise used by the first plan,
                    # so comparisons can prove that only sampling noise changed.
                    audit={f'history_{key}':value.detach().cpu().numpy().copy()
                           for key,value in policy.planner.hist.items()}
                    audit.update({key:value.detach().cpu().numpy().copy() for key,value in sensor_args.items()})
                    audit.update(state_noise=policy.planner.state_noise.detach().cpu().numpy().copy(),
                        code_noise=policy.planner.action_noise.detach().cpu().numpy().copy(),
                        initial_qpos=plant.data.qpos.copy(),
                        initial_qvel=plant.data.qvel.copy())
                    np.savez_compressed(output/'first_condition.npz',**audit)
                # No motor-session guards in a simulated experiment. Retain
                # training action and torque clamps; log saturation and falls.
                qpos=plant.data.qpos.copy();root=plant.data.xpos[plant.pelvis].copy()
                with viewer.lock() if viewer else nullcontext():
                    issued,target,torque,clipped=plant.step(action)
                previous=torch.tensor(issued,device=device).reshape(1,29)
                code=result['native_code'].detach()
                frames.append(dict(time=tick*.02,q=obs.q,dq=obs.dq,gravity=obs.gravity_body,
                    qpos=qpos,root_position=root,action=issued,target=target,native_code=code[0].cpu().numpy(),
                    raw_action_max=raw_max,saturated_joints=int((np.abs(action)>=plant.clip).sum()),
                    torque_peak=torque,torque_clipped=clipped,depth_available=np.asarray(availability,dtype=bool),
                    wall_ms=(time.monotonic()-wall_start)*1000))
                lookat=plant.data.xpos[plant.pelvis].copy();lookat[2]=max(.35,lookat[2])
                if renderer:
                    camera.lookat[:]=lookat
                    renderer.update_scene(plant.data,camera=camera,scene_option=render_options)
                    writer.append_data(renderer.render())
                if viewer:
                    with viewer.lock():viewer.cam.lookat[:]=lookat
                    viewer.sync()
                    remaining=.02-(time.monotonic()-wall_start)
                    if remaining>0:time.sleep(remaining)
                if tick%50==0:
                    print(f'  t={tick*.02:5.1f}s height={root[2]:.3f}m upright={-obs.gravity_body[2]:.3f} '
                          f'raw_action={raw_max:.2f} plans={len(replans)}',flush=True)
                if any(int(w.number)>0 for w in plant.data.warning):
                    status='physics_warning'
                    if viewer:halted=True
                    else:break
                tick+=1
        except KeyboardInterrupt:
            status='interrupted'
        except Exception:
            status='failed'
            raise
        finally:
            if frames:
                arrays={key:np.asarray([f[key] for f in frames]) for key in frames[0]}
                arrays['final_qpos']=plant.data.qpos.copy()
                np.savez_compressed(output/'trajectory.npz',**arrays)
            upright=[f['root_position'][2]>=.55 and -f['gravity'][2]>=.75 for f in frames]
            held=0;first_hold=None
            for i,good in enumerate(upright):
                held=held+1 if good else 0
                if held>=50 and first_hold is None:first_hold=(i+1)*.02
            summary=dict(status=status,reset=plant.reset_info,planner_seed=seed,sensor_seed=sensor_seed,
                frames=len(frames),simulated_seconds=len(frames)*.02,
                upright_hold_1s=first_hold is not None,first_upright_hold_s=first_hold,
                final_second_upright_fraction=float(np.mean(upright[-50:])) if upright else None,
                upright_metric='pelvis world height >= .55 m and upright cosine >= .75 for 50 control ticks; diagnostic, not training success',
                wall_seconds=time.monotonic()-began,replan_ticks=replans,
                min_pelvis_height_m=min((float(f['root_position'][2]) for f in frames),default=None),
                final_pelvis_height_m=float(plant.data.xpos[plant.pelvis,2]),
                saturated_action_ticks=sum(f['saturated_joints']>0 for f in frames),
                max_raw_action=max((f['raw_action_max'] for f in frames),default=None),
                final_upright_cos=float(-plant.observation().gravity_body[2]),
                physics_warnings={str(mujoco.mjtWarning(i)):int(w.number) for i,w in enumerate(plant.data.warning) if w.number})
            (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            (output/'plans.json').write_text(json.dumps(plans)+'\n')
    return summary


def main():
    args=arguments()
    if args.video and not args.viewer:os.environ.setdefault('MUJOCO_GL','egl')
    import torch
    import yaml
    from omegaconf import OmegaConf
    from motion_diffusion.behavioral.deploy import NativeRecoveryPolicy
    from motion_diffusion.behavioral.hardware.cli import load_controller
    from motion_diffusion.behavioral.runtime import load_frozen_planner
    from motion_diffusion.behavioral.hardware.bundle import load_bundle,asset_path
    from motion_diffusion.behavioral.hardware.export_pd import derive_pd
    from motion_diffusion.behavioral.hardware.engines import configure_controller,configure_planner
    from motion_diffusion.behavioral.mujoco import DEFAULT_MJCF,flat_model,MujocoNativePlant
    torch.set_num_threads(1)
    config=yaml.safe_load((ROOT/'g1_deploy/configs/diffusion.yaml').read_text())
    bundle,manifest=load_bundle(args.bundle or config['bundle'])
    output=(args.output or ROOT/'logs/native_recovery'/f'mujoco_flat_{time.time_ns()}').resolve()
    output.mkdir(parents=True,exist_ok=False)
    print(f'Output: {output}',flush=True)
    runs={}
    for name in (('diffusion','flow') if args.model=='both' else (args.model,)):
        row=manifest['models'][name]
        controller_path=asset_path(bundle,row['controller']);planner_path=asset_path(bundle,row['planner'])
        training=OmegaConf.load(asset_path(bundle,row['controller_training_config']))
        controller=load_controller(dict(controller_checkpoint=str(controller_path),tracker_device=args.device,
            pd_sidecar=str(asset_path(bundle,row['pd_binding']))))
        inference=dict(row['reference_inference'])
        planner=load_frozen_planner(planner_path,args.device,seed=args.seed,sampling_rng='paired_episode_v2',**inference)
        if name=='diffusion' and (planner.replan_every_ticks!=planner.model.F or planner.blend_ticks):
            raise ValueError('Diffusion recipe must execute its complete horizon without blending')
        if name=='flow' and not 0<planner.replan_every_ticks<planner.model.F:
            raise ValueError('Flow recipe must replan before its horizon ends')
        if args.backend!='pytorch':
            def engine(key,precision):
                directory=Path(config['exports'][key])
                if not directory.is_absolute():directory=ROOT/directory
                return dict(directory=str(directory),backend=args.backend,precision=precision,execution_mode=args.engine_mode,
                            validation_mode='measure' if precision=='tf32' else 'strict')
            configure_controller(controller,engine('controller','fp32'))
            configure_planner(planner,planner_path,engine(name,args.planner_precision))
        policy=NativeRecoveryPolicy(controller,planner)
        validation={key:engine.validation_report for key,engine in (
            ('denoiser',planner.model.inference_backend),
            ('depth',None if planner.model.depth_encoder is None else planner.model.depth_encoder.inference_backend))
            if engine is not None}
        if any(not v['passed'] for v in validation.values()):
            print('Experimental precision: neural parity tolerance exceeded; measuring closed-loop behavior.',flush=True)
        robot=OmegaConf.to_container(training.robot,resolve=True)
        pd=derive_pd(controller.contract,dict(robot=robot,simulator=dict(config=dict(sim=OmegaConf.to_container(training.simulator.config.sim,resolve=True)))))
        model=flat_model(args.mjcf or DEFAULT_MJCF)
        plant=MujocoNativePlant(model,controller.contract,pd,robot)
        depth_configs=[c for c in (controller.depth_config,planner.model.behavioral_config.get('depth_condition',{})) if c.get('enabled')]
        capture=depth_configs[0]['capture'] if depth_configs else None
        if any(c['capture']!=capture for c in depth_configs):raise ValueError('Tracker/planner camera mounts differ')
        print(f'{name}: horizon={planner.model.F}, replan={planner.replan_every_ticks}, blend={planner.blend_ticks} ticks; '
              f'{inference["sampler"]}{inference["num_inference_steps"]}; start={args.start}',flush=True)
        viewer=None;controls=None
        if args.viewer:
            import mujoco.viewer
            controls=ViewerControls()
            viewer=mujoco.viewer.launch_passive(model,plant.data,key_callback=controls.key)
            with viewer.lock():
                viewer.cam.distance=2.5;viewer.cam.azimuth=135;viewer.cam.elevation=-18
                viewer.opt.geomgroup[:]=[1,1,0,0,0,0]
        episode=0;seed=args.seed
        postures=args.postures or [args.start]
        while episode<args.episodes*len(postures) or viewer:
            if not viewer:
                args.start=postures[episode//args.episodes]
                seed=args.seed+episode%args.episodes
            directory=output/f'{name}_{episode:03d}';directory.mkdir()
            (directory/'config.json').write_text(json.dumps(dict(model=name,seed=seed,start=args.start,joint_jitter=args.joint_jitter,
                pose_seed=args.pose_seed,sensor_seed=args.sensor_seed,
                backend=args.backend,planner_precision=args.planner_precision,engine_mode=args.engine_mode,
                neural_validation=validation,
                mjcf=str((args.mjcf or DEFAULT_MJCF).resolve()),bundle=str(bundle),
                controller_sha256=row['controller']['sha256'],planner_sha256=row['planner']['sha256'],
                inference=inference,sampling_rng='paired_episode_v2',pd=pd,camera=capture,
                effort_limits=plant.effort.tolist(),physics_dt=float(model.opt.timestep),control_dt=.02,
                timing='synchronous simulation clock; inference pauses physics',
                plant='SONIC MuJoCo asset, saved nominal PD/armature/friction/effort, flat friction1; no domain randomization'),indent=2)+'\n')
            try:result=run_episode(policy,plant,capture,args,directory,seed,viewer,controls)
            except BaseException:
                if viewer:viewer.close()
                raise
            runs[directory.name]=result
            (output/'summary.json').write_text(json.dumps(runs,indent=2)+'\n')
            episode+=1
            seed=getattr(args,'next_seed',seed if viewer else seed+1)
            if result['status'] in ('interrupted','viewer_closed','next_model','physics_warning'):
                if viewer:viewer.close()
                if result['status']!='next_model':return
                break
        if viewer:viewer.close()
        del policy,planner,controller
        if torch.cuda.is_available():torch.cuda.empty_cache()
    print(f'Finished. Results: {output}',flush=True)


if __name__=='__main__':main()
