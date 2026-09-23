"""Run one imported terrain-steering example in MuJoCo or on the robot."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time

from steerable_recovery.paths import ROOT, PACKAGE_ROOT, CONFIG_ROOT, RESULTS_ROOT, VALIDATION_ROOT

LEGACY_CATALOG=CONFIG_ROOT/'steering/terrain_gallery/catalog.json'
CATALOG=CONFIG_ROOT/'steering/lab_gallery/catalog.json'


def selection(catalog, name):
    data=json.loads(Path(catalog).read_text())
    if data.get('schema')!='hv2.terrain_gallery.v1':raise ValueError('Invalid terrain gallery catalog')
    ids=data['aliases'].get(name,[name] if name in data['runs'] else [])
    if len(ids)!=1:
        available=[k for k,v in data['aliases'].items() if len(v)==1]
        raise ValueError('Select one individual run. Available: '+', '.join(available))
    return data['runs'][ids[0]],ids[0]


def command_for(args):
    row,rid=selection(args.catalog,args.example);root=args.catalog.resolve().parent
    seed=row['seed'] if args.seed is None else args.seed
    rate=row['replan_hz'] if args.replan_hz is None else args.replan_hz
    common=['--model',row['model'],'--backend','tensorrt','--goal-json',str(root/row['goal']),
        '--seed',str(seed),'--noise-policy','fresh','--replan-hz',str(rate)]
    if args.guidance_scale is not None:common+=['--guidance-scale',str(args.guidance_scale)]
    blend=getattr(args,'blend_ticks',None)
    if blend is None:blend=row.get('blend_ticks')
    if blend is not None:common+=['--blend-ticks',str(blend)]
    hold=getattr(args,'standing_hold',None)
    if (row.get('standing_hold',False) if hold is None else hold):common+=['--standing-hold']
    if args.check:
        return [sys.executable,str(PACKAGE_ROOT/'cli/deploy.py'),*common,'--check']
    if args.robot:
        output=args.output or (VALIDATION_ROOT if args.sensor_only else RESULTS_ROOT)/f'gallery_{args.example}_{time.time_ns()}'
        cmd=[sys.executable,str(PACKAGE_ROOT/'cli/deploy.py'),*common,
            '--live-odometry','--steering-feedback',args.feedback,
            '--record-sensors','--record-depth','--record-model',
            '--record-label',f'gallery_{args.example}', '--output',str(output),
            '--duration-s',str(args.seconds if args.seconds is not None else 60.),
            '--depth-processing',args.depth_processing]
        if args.external_camera:cmd+=['--external-camera']
        if not args.sensor_only:
            cmd+=['--actuate']
            if args.position_start:cmd+=['--start-pose',str(root/row['robot_pose'])]
        if args.odometry_clock:cmd+=['--odometry-clock',str(args.odometry_clock)]
    else:
        cmd=[sys.executable,str(PACKAGE_ROOT/'cli/sim.py'),*common,
            '--gallery-start',str(root/row['start']),'--start',row['posture'],
            '--sensor-seed','42','--noise-only','--fixed-camera']
        if not args.headless:cmd+=['--viewer']
        if args.seconds is not None:cmd+=['--seconds',str(args.seconds)]
        if args.output:cmd+=['--output',str(args.output)]
        if getattr(args,'standing_hold_on_upright',False):cmd+=['--standing-hold-on-upright']
    return cmd


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--example',required=True,help='Exact run, e.g. robust_01_minimal; list prints all runs')
    p.add_argument('--catalog',type=Path,default=CATALOG)
    p.add_argument('--robot',action='store_true');p.add_argument('--sensor-only',action='store_true')
    p.add_argument('--check',action='store_true',help='Model and goal forward check; no sensors or motors')
    p.add_argument('--benchmark',action='store_true',help='Guided PyTorch planner/TRT tracker timing; no sensors or motors')
    p.add_argument('--headless',action='store_true');p.add_argument('--seconds',type=float)
    p.add_argument('--standing-hold',action=argparse.BooleanOptionalAction,default=None,help='Y switches to encoded standing intent after recovery (robot or simulation)')
    p.add_argument('--standing-hold-on-upright',action='store_true',help='MuJoCo diagnostic: simulate Y after .3 s upright')
    p.add_argument('--seed',help='Default preserves gallery seed; random samples a new episode')
    p.add_argument('--blend-ticks',type=int,help='Override latent overlap; omitted preserves source recipe')
    p.add_argument('--replan-hz',type=float);p.add_argument('--guidance-scale',type=float)
    p.add_argument('--feedback',choices=('qualified','estimated'),default='qualified')
    p.add_argument('--position-start',action='store_true',help='Offer original joints on optional START; A still starts from measured pose')
    p.add_argument('--depth-processing',choices=('raw','vigor','vigor_legacy'),default='vigor_legacy')
    p.add_argument('--external-camera',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--odometry-clock',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--print-command',action='store_true')
    a=p.parse_args(argv)
    if a.example=='list':
        data=json.loads(a.catalog.read_text())
        for name,ids in data['aliases'].items():
            if len(ids)==1:
                r=data['runs'][ids[0]]
                print(f'{name:20} {r["model"]:28} {r["terrain"]:20} {r["posture"]:7} {r["mode"]:14} gain={r["gain"]:g} rate={r["replan_hz"]:g} Hz blend={r.get("blend_ticks",0)} ticks')
        return
    if a.sensor_only and not a.robot:p.error('--sensor-only requires deployment')
    if a.headless and a.robot:p.error('--headless is for simulation')
    if a.robot and a.standing_hold_on_upright:p.error('Automatic standing-hold activation is a MuJoCo diagnostic only')
    if a.position_start and (not a.robot or a.sensor_only):p.error('--position-start is an optional actuated robot preparation')
    if a.benchmark and a.robot:p.error('--benchmark opens no robot connection')
    if a.guidance_scale is not None and (not math.isfinite(a.guidance_scale) or a.guidance_scale<=0):p.error('Guidance must be positive and finite')
    try:row,rid=selection(a.catalog,a.example)
    except ValueError as e:p.error(str(e))
    print(f'{a.example}: {row["model"]}, {row["terrain"]}, motion {row["motion_id"]}, {row["mode"]}.',flush=True)
    print('Command axes: fixed heading at policy start; XYZ is pelvis displacement, Z is rise from start, not height above floor.',flush=True)
    if 'start_id' in row:
        ending=(f"Progress-based XYZ; release at {row['release_threshold_deg']:g} degrees of {row['direction']} side roll."
                if row.get('steering_method')=='progress_xyz' else
                f"Timed XYZ + orientation; guidance ends at {row.get('release_after_s',3.):g} s.")
        if row.get('fade_start_deg') is not None:
            ending+=f" Fade starts at {row['fade_start_deg']:g} degrees, with a 15% floor until release."
        print(f"Placement: {row['start_id']}, {row['edge']}, {row['posture']}; {row['behavior_axis']}. "
              f"Source cadence {row['replan_hz']:g} Hz / {row['blend_ticks']} blend ticks. {ending}",flush=True)
    if (row.get('standing_hold',False) if a.standing_hold is None else a.standing_hold):
        print(('Remote Y after A' if a.robot else 'Keyboard Y')+
              ' switches to encoded standing; remains active until the session ends. No position lock.',flush=True)
    if a.robot:
        print('Place the body on supports matching the example. A starts from the measured pose. '+
            ('START optionally moves to the recorded joints.' if a.position_start else 'No preset joint movement.'),flush=True)
    else:print('Recorded start and terrain; nominal MuJoCo dynamics, not an exact Isaac batch/DR replay.',flush=True)
    if a.benchmark:
        from .steering import benchmark
        a.model=row['model'];a.seed=str(row['seed']) if a.seed is None else a.seed
        if a.replan_hz is None:a.replan_hz=row['replan_hz']
        if a.blend_ticks is None:a.blend_ticks=row.get('blend_ticks')
        return benchmark(a,goal_path=a.catalog.resolve().parent/row['goal'])
    cmd=command_for(a);print(shlex.join(cmd),flush=True)
    if not a.print_command:os.chdir(ROOT);os.execv(sys.executable,cmd)

if __name__=='__main__':main()
