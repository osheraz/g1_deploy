#!/usr/bin/env python3
"""Extract Stanley's coarse and dense full-pelvis commands; opens no sensors.

The originals are retained under config/multimodal/stanley_flat/reference_routes.
Copies the selected schedule exactly; saves source hashes and donor provenance.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'g1_deploy/config/multimodal/stanley_flat/reference_routes'
DEST=ROOT/'g1_deploy/config/steering/trajectories'


def extract_goal(source,*,waypoints='coarse'):
    if waypoints not in ('coarse','dense'):raise ValueError('Choose coarse or dense waypoints')
    source=Path(source);original=json.loads(source.read_text());target=original['target']
    schedule=target['coarse_waypoints' if waypoints=='coarse' else 'pose_waypoints']
    goal={key:deepcopy(original[key]) for key in ('schema','clock','frame','mode','guidance_scale',
        'guidance_max_grad_norm','position_error_scale_m','yaw_error_scale_deg','simulation_reference','hardware_qualified')}
    goal['target']=dict(pose_waypoints=deepcopy(schedule))
    goal['provenance']=dict(source_file=str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),selection=waypoints,
        episode_id=target['episode_id'],mode_support=target['mode_support'],
        schedule_changes='None: original XYZ, full orientation, timestamps and tolerances.',
        execution='Original schedule holds its final pose after the final timestamp; no custom cutoff.')
    return goal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEST)
    args=p.parse_args()
    for i in range(3):
        for mode in ('coarse','dense'):
            goal=extract_goal(SOURCE/f'flat_pose_{i}.json',waypoints=mode)
            dest=args.output/mode/f'option{i+1}.json';dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text(json.dumps(goal,indent=2,allow_nan=False)+'\n')
            print(f'{dest}: {len(goal["target"]["pose_waypoints"]["times_s"])} waypoints')

if __name__=='__main__':main()
