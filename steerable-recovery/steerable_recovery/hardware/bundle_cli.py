"""Prepare/check paired bundles and launch explicitly selected hardware sessions."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
from steerable_recovery.paths import ROOT, CONFIG_ROOT
from motion_diffusion.inference.bundle import (prepare, load_bundle, asset_path, inspect_pair, resolve_session, select_backends, add_backend_arguments, write_json, sha256)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare',help='Verify and assemble selected weights; no sensors')
    p.add_argument('--catalog',required=True);p.add_argument('--output',required=True)
    p.add_argument('--models',nargs='+',default=['diffusion','flow'])
    p.add_argument('--asset-mode',choices=['symlink','copy'],default='symlink')
    p=sub.add_parser('check',help='Validate transferred bundle/model pairing on CPU; no sensors')
    p.add_argument('--bundle',default=os.environ.get('HV2_RECOVERY_BUNDLE'),required=not os.environ.get('HV2_RECOVERY_BUNDLE'))
    p.add_argument('--model',default='all')
    p.add_argument('--sample',action='store_true',help='Also run a synthetic missing-history startup forward')
    p=sub.add_parser('run',help='Real sensors with default no motor output')
    p.add_argument('--bundle',default=os.environ.get('HV2_RECOVERY_BUNDLE'),required=not os.environ.get('HV2_RECOVERY_BUNDLE'))
    p.add_argument('--model',required=True)
    p.add_argument('--hardware',default=os.environ.get('HV2_RECOVERY_HARDWARE'),required=not os.environ.get('HV2_RECOVERY_HARDWARE'))
    p.add_argument('--output',required=not os.environ.get('HV2_RECOVERY_SESSIONS'),
        help='Fresh session directory; alternatively set HV2_RECOVERY_SESSIONS for automatic unique subdirectories')
    p.add_argument('--task',choices=['recovery','yaw','height-local','height-trajectory','pelvis-path','motion-command'],default='recovery')
    p.add_argument('--duration-s',type=float,default=30.);p.add_argument('--actuate',action='store_true')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-plan-age-ticks',type=int,help='Override the bundle\'s explicit provisional reply-age budget')
    p.add_argument('--replan-ticks',type=int);p.add_argument('--guidance-scale',type=float)
    p.add_argument('--guidance-cap',type=float);p.add_argument('--yaw-deg',type=float)
    p.add_argument('--rise-m',type=float);p.add_argument('--rise-time-s',type=float);p.add_argument('--goal-json')
    add_backend_arguments(p)
    args=parser.parse_args()
    if args.command=='prepare':
        value=prepare(args.catalog,args.output,args.models,asset_mode=args.asset_mode, hardware_template=CONFIG_ROOT / 'hardware.template.json')
        print(json.dumps(dict(bundle=args.output,models=list(value['models']),hardware_qualified=False)))
        return
    root,manifest=load_bundle(args.bundle)
    if args.command=='check':
        names=list(manifest['models']) if args.model=='all' else [args.model]
        for name in names:
            row=manifest['models'][name]
            for asset in ('planner','controller','pd_binding','controller_training_config'):asset_path(root,row[asset])
            value=inspect_pair(asset_path(root,row['planner']),asset_path(root,row['controller']),
                row['reference_inference'],row['objective'],sample=args.sample)
            print(json.dumps(dict(model=name,verified=value),allow_nan=False))
        return
    if not math.isfinite(args.duration_s) or args.duration_s<=0:parser.error('Positive finite duration required')
    config,receipt=resolve_session(root,manifest,args.model,args.hardware,task=args.task,
        max_age=args.max_plan_age_ticks,replan_ticks=args.replan_ticks,seed=args.seed,
        guidance_scale=args.guidance_scale,guidance_cap=args.guidance_cap,yaw_deg=args.yaw_deg,
        rise_m=args.rise_m,rise_time_s=args.rise_time_s,goal_path=args.goal_json)
    select_backends(config,args)
    receipt['neural_backends']={k:config[k] for k in ('controller_engine','planner_engine') if k in config}
    output=(Path(args.output) if args.output else Path(os.environ['HV2_RECOVERY_SESSIONS']) /
        f'{args.model}_{args.task}_{time.strftime("%Y%m%d_%H%M%S")}_{time.time_ns()}').absolute()
    output.mkdir(parents=True,exist_ok=False)
    receipt.update(mode='actuation' if args.actuate else 'sensor_dry_run',duration_s=args.duration_s,output=str(output))
    try:
        git=['git','-C',str(ROOT)]
        receipt['git_sha']=subprocess.check_output([*git,'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
        receipt['git_dirty']=bool(subprocess.check_output([*git,'status','--porcelain'],text=True).strip())
    except (OSError,subprocess.CalledProcessError):receipt['git_sha']=None
    if receipt['camera_calibration'] is not None:
        snapshot=output/'camera_calibration.json'
        write_json(snapshot,receipt['camera_calibration']['content'])
        receipt['camera_calibration']['snapshot_sha256']=sha256(snapshot)
        config['camera']['calibration_receipt']=str(snapshot)
    write_json(output/'session.json',receipt);write_json(output/'resolved.json',config)
    print(json.dumps(dict(event='selected',**receipt),allow_nan=False),flush=True)
    from steerable_recovery.hardware.runtime import run
    start=time.monotonic()
    try:
        result=run(config,enable_actuation=args.actuate,duration_s=args.duration_s,
            telemetry_jsonl=output/'telemetry.jsonl.gz')
    except BaseException as error:
        write_json(output/'FAILED.json',dict(error=str(error),wall_s=time.monotonic()-start))
        raise
    write_json(output/'COMPLETE.json',dict(**result,wall_s=time.monotonic()-start))


if __name__ == '__main__':
    main()
