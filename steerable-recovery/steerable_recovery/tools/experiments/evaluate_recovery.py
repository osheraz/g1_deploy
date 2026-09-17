#!/usr/bin/env python3
"""Run matched ground-start MuJoCo episodes for PyTorch, TRT FP32 and experimental TRT TF32."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=3,help='Seeds per posture/model/backend')
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--joint-jitter',type=float,default=.05)
    p.add_argument('--postures',nargs='+',default=['standing','prone','supine','left_side','right_side','seated'])
    p.add_argument('--video',action='store_true')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    output=(args.output or ROOT/'logs/native_recovery'/f'recovery_precision_{time.time_ns()}').resolve()
    output.mkdir(parents=True,exist_ok=False)
    report={'scope':'Matched seeds and reset postures; synchronous physics, no deployment-delay or robot test',
            'arguments':vars(args)|{'output':str(output)},'profiles':{}}
    for profile,flags in [('pytorch',['--backend','pytorch']),('trt_fp32',['--backend','tensorrt']),
                          ('trt_tf32',['--backend','tensorrt','--planner-precision','tf32'])]:
        command=[sys.executable,str(ROOT/'g1_deploy/sim_diffusion.py'),'--model','both',
            '--postures',*args.postures,'--seconds',str(args.seconds),'--episodes',str(args.episodes),
            '--seed',str(args.seed),'--joint-jitter',str(args.joint_jitter),'--output',str(output/profile),*flags]
        if args.video:command.append('--video')
        print('Running',profile,flush=True)
        with (output/f'{profile}.log').open('w') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        path=output/profile/'summary.json'
        runs=json.loads(path.read_text()) if path.exists() else {}
        report['profiles'][profile]=dict(exit_code=result.returncode,runs=runs)
        (output/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(f'Report: {output}/report.json',flush=True)
    if any(x['exit_code'] for x in report['profiles'].values()):raise SystemExit(1)

if __name__=='__main__':main()
