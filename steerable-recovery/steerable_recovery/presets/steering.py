#!/usr/bin/env python3
"""Stanley's three pelvis XYZ/orientation trajectories from shared supine151.

Default: live MuJoCo. --robot: normal operator-controlled deployment, recordings
and live odometry. --sensor-only: measure the complete robot inference path with
no motor output. --feedback estimated explicitly selects experimental XYZ.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
POSES=ROOT/'g1_deploy/config/multimodal/stanley_flat'
TRAJECTORIES=ROOT/'g1_deploy/config/steering/trajectories'
SELECTED_WAYPOINTS={1:'dense',2:'coarse',3:'coarse'}
MODEL_RATES={'r8_s25_diffusion':2.5,'r8_s50r_diffusion':5.}
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'g1_deploy'))


def goal_for(args):
    mode=SELECTED_WAYPOINTS[args.option] if args.waypoints=='selected' else args.waypoints
    return TRAJECTORIES/mode/f'option{args.option}.json'


def command_for(args):
    goal=goal_for(args)
    pose_root=TRAJECTORIES
    rate=MODEL_RATES[args.model] if args.replan_hz is None else args.replan_hz
    common=['--model',args.model,'--backend','tensorrt',
            '--goal-json',str(goal),'--seed',str(args.seed),'--noise-policy','fresh',
            '--replan-hz',str(rate)]
    strength=getattr(args,'guidance_scale',None)
    suffix='' if strength is None else f'_g{strength:g}'
    if strength is not None:common+=['--guidance-scale',str(strength)]
    if args.robot:
        output=args.output or ROOT/('logs/deployment_validation' if args.sensor_only else 'logs/real_world')/f'pelvis_{args.model}_{rate:g}hz_{args.waypoints}_option{args.option}{suffix}_{time.time_ns()}'
        cmd=[sys.executable,str(ROOT/'g1_deploy/deploy_diffusion.py'),*common,
             '--duration-s',str(60. if args.seconds is None else args.seconds),'--live-odometry',
             '--record-sensors','--record-depth','--record-model',
             '--steering-feedback',args.feedback,
             '--record-label',f'flat_supine151_{args.model}_{rate:g}hz_pelvis_{args.waypoints}_option{args.option}{suffix}','--output',str(output)]
        if not args.sensor_only:
            cmd+=['--actuate','--start-pose',str(pose_root/'supine_151_robot.json')]
        if args.external_camera:cmd+=['--external-camera']
        if args.odometry_clock:cmd+=['--odometry-clock',str(args.odometry_clock)]
    else:
        cmd=[sys.executable,str(ROOT/'g1_deploy/sim_diffusion.py'),*common,
             '--terrain','flat','--start','supine','--start-state',str(pose_root/'supine_151.json'),
             '--sensor-seed','42','--noise-only','--fixed-camera']
        if not args.headless:cmd+=['--viewer']
        if args.seconds is not None:cmd+=['--seconds',str(args.seconds)]
        if args.output:cmd+=['--output',str(args.output)]
    return cmd


def benchmark(args):
    """Actual spawned planner and TRT tracker, synthetic observations, no sensors."""
    import torch
    from core.config import _load_config
    from g1_deploy.deploy_diffusion import model_settings,hardware_settings,sampling_seed
    from motion_diffusion.behavioral.hardware import bundle
    from motion_diffusion.behavioral.hardware.cli import load_controller
    from motion_diffusion.behavioral.hardware.engines import configure_controller
    from motion_diffusion.behavioral.hardware.execution import PlannerProcess,AsyncNativeRecovery
    torch.set_num_threads(1)
    path=ROOT/'g1_deploy/configs/diffusion.yaml'
    settings=model_settings(_load_config(path),args.model)
    root,manifest=bundle.load_bundle(settings['bundle']);row=manifest['models'][args.model]
    ckpt=torch.load(bundle.asset_path(root,row['controller']),map_location='cpu',weights_only=True,mmap=True)
    hardware=hardware_settings(settings,ckpt['action_vae']['model_contract'])
    hardware['pelvis_odometry_port']=15560
    rate=MODEL_RATES[args.model] if args.replan_hz is None else args.replan_hz
    config,_=bundle.resolve_session(root,manifest,args.model,path,hardware_config=hardware,
        task='motion-command',goal_path=goal_for(args),
        guidance_scale=args.guidance_scale,
        guided_timing=settings.get('guided_timing'),
        seed=sampling_seed(str(args.seed)),replan_ticks=bundle.replan_ticks_for_hz(rate,horizon=row['inspected_contract']['F']))
    actor=load_controller(config)
    export=ROOT/settings['exports']['controller']
    configure_controller(actor,dict(backend='tensorrt',directory=str(export),precision='fp32',execution_mode='cuda_graph'))
    worker=PlannerProcess(config['planner_checkpoint'],device=config['planner_device'],options=config['planner_options'])
    try:
        runtime=AsyncNativeRecovery(actor,worker,max_plan_age_ticks=config['max_plan_age_ticks'])
        cold=runtime.warmup()
        warm=runtime.warmup()
        report=dict(schema='hv2.guided_inference_check.v1',option=args.option,
            model=args.model,replan_hz=rate,max_plan_age_ticks=config['max_plan_age_ticks'],
            guidance_scale=config['planner_options']['motion_goal']['guidance_scale'],
            motors_opened=False,sensors_opened=False,cold=cold,warm=warm,
            planner_backend='pytorch',tracker_backend='tensorrt',
            scope='Synthetic pre-clock warmup; not concurrent sensor/recording or dynamic accuracy qualification')
        output=args.output or ROOT/'logs/deployment_validation'/f'steering_benchmark_{time.time_ns()}'
        output.mkdir(parents=True,exist_ok=False)
        (output/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f'Saved {output}/benchmark.json')
        print('Warm planner samples (ms):', [round(v*1000,1) for v in warm['planner']['sample_wall_s']])
    finally:worker.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--option',type=int,choices=(1,2,3),required=True)
    p.add_argument('--model',choices=tuple(MODEL_RATES),default='r8_s50r_diffusion',
        help='S25: 0.5 s horizon, defaults to 2.5 Hz; S50R: 1 s horizon, defaults to 5 Hz')
    p.add_argument('--waypoints',choices=('selected','coarse','dense'),default='selected',help="Selected: dense route1, six-point routes2/3. Overrides retain XYZ, orientation and physical timestamps")
    p.add_argument('--feedback',choices=('qualified','estimated'),default='qualified',help='Estimated explicitly allows fresh geometry-valid XYZ with unqualified clock bounds')
    p.add_argument('--robot',action='store_true')
    p.add_argument('--sensor-only',action='store_true')
    p.add_argument('--benchmark',action='store_true',help='Synthetic guided compute check; opens no sensors or motors')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--seconds',type=float)
    p.add_argument('--seed',default='42',help='Use the same seed across routes to isolate steering; random is supported')
    p.add_argument('--replan-hz',type=float,help='Override planner cadence, e.g. 2.5, 5 or 10 Hz; tracker stays 50 Hz')
    p.add_argument('--guidance-scale',type=float,help='Planner steering strength, e.g. 50 or 100; default uses the source goal (300)')
    p.add_argument('--external-camera',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--odometry-clock',type=Path,help='Independent measured clock calibration; no fabricated default')
    p.add_argument('--output',type=Path)
    p.add_argument('--print-command',action='store_true')
    args=p.parse_args(argv)
    if args.guidance_scale is not None:
        import math
        if not math.isfinite(args.guidance_scale) or args.guidance_scale<=0:p.error('--guidance-scale must be finite and positive')
    if args.sensor_only and not args.robot:p.error('--sensor-only requires --robot')
    if args.benchmark and args.robot:p.error('--benchmark opens no robot connection')
    if args.headless and args.robot:p.error('--headless is a simulation option')
    if args.benchmark:return benchmark(args)
    goal=json.loads(goal_for(args).read_text());schedule=goal['target']['pose_waypoints']
    print(f"Pelvis trajectory {args.option}: {len(schedule['times_s'])} XYZ/orientation waypoints, "
          f"{schedule['times_s'][-1]:.2f} s; final pose held afterward. "
          f"Guidance scale: {goal['guidance_scale'] if args.guidance_scale is None else args.guidance_scale:g}.",flush=True)
    print(f"XYZ is pelvis displacement from policy start in fixed initial-heading axes; +Z is up. "
          f"Final Z requests a {schedule['positions_m'][-1][2]:.3f} m rise, not floor clearance.",flush=True)
    cmd=command_for(args)
    print(shlex.join(cmd),flush=True)
    if args.print_command:return
    os.chdir(ROOT);os.execv(sys.executable,cmd)


if __name__=='__main__':main()
