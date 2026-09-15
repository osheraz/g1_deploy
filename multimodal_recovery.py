#!/usr/bin/env python3
"""Live recovery demonstrations: same resting state, different diffusion noise.

python g1_deploy/multimodal_recovery.py --start supine
N: next selected seed. R: repeat current seed. SPACE: pause. Close window: exit.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DEFAULT_CATALOG=ROOT/'g1_deploy/config/multimodal/r8_s25/catalog.json'


def configured_bundle(catalog):
    """Bind selected seeds to the evaluated weights and sampling recipe."""
    import yaml
    from g1_deploy.deploy_diffusion import model_settings
    from motion_diffusion.behavioral.hardware.bundle import load_bundle
    settings=model_settings(yaml.safe_load((ROOT/'g1_deploy/configs/diffusion.yaml').read_text()),catalog['model'])
    bundle,manifest=load_bundle(settings['bundle'])
    row=manifest['models'][catalog['model']]
    for key in ('controller','planner'):
        if row[key]['sha256']!=catalog[key+'_sha256']:
            raise ValueError(f'Configured {key} differs from the evaluated demo weights; rerun the selection search')
    if row['reference_inference']!=catalog['reference_inference']:
        raise ValueError('Configured sampling recipe differs from the evaluated demo')
    return bundle  # sim_diffusion verifies the actual asset bytes before loading.


def simulation_arguments(args,catalog):
    row=next(r for r in catalog['starts'] if r['category']==args.start)
    path=args.catalog.resolve().parent/row['start_state']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=row['start_sha256']:
        raise ValueError('Demo starting-state file differs from the selected catalog')
    random_seed=getattr(args,'random',False)
    if random_seed:
        from g1_deploy.deploy_diffusion import sampling_seed
        seeds=[sampling_seed('random')]
    else:seeds=args.seeds or row['seeds']
    if len(seeds)!=len(set(seeds)) or any(s<0 or s>=2**63 for s in seeds):
        raise ValueError('Use distinct seeds in [0, 2**63)')
    output=args.output or ROOT/'logs/multimodal_recovery'/f'demo_{args.start}_{time.time_ns()}'
    command=['--model',catalog['model'],'--backend',catalog['backend'],
        '--terrain','flat','--start',row['posture'],'--start-state',str(path),
        *(['--seed',str(seeds[0])] if random_seed else ['--seed-list',*[str(s) for s in seeds]]),
        '--sensor-seed',str(catalog['sensor_seed']),
        '--pose-seed','42','--command-limits',catalog['command_limits'],
        '--depth-startup',catalog['depth_startup'],'--tracker-startup',catalog['tracker_startup'],
        '--output',str(output)]
    if catalog['backend']=='tensorrt':
        command+=['--planner-precision',catalog.get('planner_precision','fp32'),
                  '--engine-mode',catalog.get('engine_mode','cuda_graph')]
    if not args.headless:command+=['--viewer']
    if catalog.get('fixed_camera',False):command+=['--fixed-camera']
    if getattr(args,'replan_hz',None) is not None:command+=['--replan-hz',str(args.replan_hz)]
    if getattr(args,'noise_policy',None) is not None:command+=['--noise-policy',args.noise_policy]
    if getattr(args,'start_pose_jitter_deg',0.):command+=['--start-pose-jitter-deg',str(args.start_pose_jitter_deg)]
    if getattr(args,'start_pose_seed',None) is not None:command+=['--start-pose-seed',str(args.start_pose_seed)]
    if args.seconds is not None:command+=['--seconds',str(args.seconds)]
    elif args.headless:command+=['--seconds',str(catalog['seconds'])]
    if args.video:command+=['--video']
    return command,row,seeds


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--start',choices=('prone','supine','sideways'),default='supine')
    p.add_argument('--catalog',type=Path,default=DEFAULT_CATALOG)
    p.add_argument('--replan-hz',type=float,help='Override planning cadence, e.g. 2.5, 5 or 10 Hz; tracker remains 50 Hz')
    p.add_argument('--noise-policy',choices=('fresh','fixed_episode'),
        help='fresh: new noise every plan; fixed_episode: sample once and reuse; default follows model bundle')
    p.add_argument('--start-pose-jitter-deg',type=float,default=0.,help='Small joint perturbation before each episode; e.g. 3 degrees; N changes pose and noise')
    p.add_argument('--start-pose-seed',help='Independent pose seed: integer or random; logged for replay')
    seed_options=p.add_mutually_exclusive_group()
    seed_options.add_argument('--seeds',type=int,nargs='+',help='Explore other seeds; default cycles the selected contrasting pair')
    seed_options.add_argument('--random',action='store_true',help='Fresh random seed each launch; N advances to new noise instead of cycling selected examples')
    p.add_argument('--headless',action='store_true',help='Compute all requested seeds and save trajectories without opening a window')
    p.add_argument('--video',action='store_true',help='Also save MP4; off by default')
    p.add_argument('--seconds',type=float,help='Default: unlimited live, evaluated duration headless')
    p.add_argument('--output',type=Path)
    p.add_argument('--list',action='store_true',help='Show selected poses/seeds without loading models')
    args=p.parse_args(argv)
    catalog=json.loads(args.catalog.read_text())
    if catalog.get('schema')!='hv2.multimodal_demo.v1':p.error('Unknown demo catalog schema')
    if args.list:
        for r in catalog['starts']:
            print(f'{r["category"]}: seeds {r["seeds"]}; heading gap {r["metrics"]["heading_gap_deg"]:.1f} degrees; '
                  f'route difference {r["metrics"]["route_distance_m"]:.2f} m')
        return
    command,row,seeds=simulation_arguments(args,catalog)
    command+=['--bundle',str(configured_bundle(catalog))]
    print(f'{catalog["model"]} / {catalog["backend"]}; {args.start}; seeds {seeds}. '
          +('Joint jitter enabled: N changes pose and planner noise; R repeats both.' if args.start_pose_jitter_deg else
            'Exact same resting pose, camera seed and reset history. N changes noise only.'),flush=True)
    if args.seeds or args.random:print('Exploring requested noise; outcomes may differ from the selected examples.',flush=True)
    if args.replan_hz is not None:
        saved_hz=50/catalog['reference_inference']['replan_every_ticks']
        print(f'Replanning override: {args.replan_hz:g} Hz (catalog: {saved_hz:g} Hz). '
              'Tracker remains 50 Hz; the recovery paths may change.',flush=True)
    from g1_deploy.sim_diffusion import main as simulate
    simulate(command)


if __name__=='__main__':main()
