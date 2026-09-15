#!/usr/bin/env python3
"""Mine teacher training starts, then measure seed diversity in the deployed model."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from itertools import combinations
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

RELEASE=Path('/nfs/turbo/coe-stellayu-extra/osher/projects/hv2/releases/fork3_rise_salmon15501_isaac6_20260911')


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    import tempfile,os
    text=json.dumps(value,indent=2,allow_nan=False)+'\n'
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix='.'+path.name,suffix='.tmp',delete=False) as f:
        f.write(text);temporary=f.name
    os.replace(temporary,path)


def posture(quat):
    import numpy as np
    from scipy.spatial.transform import Rotation
    gravity=-Rotation.from_quat(quat).as_matrix()[2]
    if abs(gravity[0])>.7:return 'prone' if gravity[0]>0 else 'supine'
    if abs(gravity[1])>.7:return 'left_side' if gravity[1]>0 else 'right_side'
    return None


def yaw_gap(a,b):
    import numpy as np
    return float(abs(np.arctan2(np.sin(a-b),np.cos(a-b))))


def mine(release,output,per_posture):
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from humanoidverse2.agents.action_vae.heading import initialize_heading
    from scripts.fall_recovery.datasets.repertoire import _ordered_distance
    print('Reading the selected teacher training split and matched-reset audit...',flush=True)
    packs=release/'packs/site_heldout_15k'
    manifest=json.loads((packs/'split_manifest.json').read_text())
    train={r['raw_file']:r for r in manifest['episodes'] if r['split']=='train'}
    audit=json.loads((packs/'repertoire_selection.json').read_text())
    groups=defaultdict(list)
    for row in audit['episodes']:
        if row['file'] in train and row.get('accepted') and row.get('execution'):
            groups[tuple(row['execution']['setup_identity'])].append(row)
    ranked=[]
    for group,rows in groups.items():
        best=None
        for a,b in combinations(rows,2):
            if a['motion_id']==b['motion_id']:continue
            ea,eb=a['execution'],b['execution']
            yaw=yaw_gap(ea['realized_held_success_yaw_rad'],eb['realized_held_success_yaw_rad'])
            pa,pb=np.array(ea['realized_held_success_path_xy']),np.array(eb['realized_held_success_path_xy'])
            distance=float(np.sqrt(np.mean(np.sum((pa-pb)**2,axis=1))))
            score=yaw/(np.pi/2)+distance/.5
            if best is None or score>best[0]:best=(score,a,b,yaw,distance)
        if best is not None:ranked.append((best[0],group,best))
    print(f'{len(train)} train episodes; {len(ranked)} matched-reset groups with alternatives. Reading source start labels...',flush=True)
    source=torch.load(release/'source_reference.pt',map_location='cpu',weights_only=False,mmap=True)
    labels={i:posture(np.array(e['root_wxyz_xyz'])[0,[1,2,3,0]]) for i,e in enumerate(source['entries'])}
    del source
    counts=defaultdict(int);candidates=[];seen=set();teacher_arrays={}
    raw=release/'work/full/raw'
    for _,group,(_,a,b,yaw,distance) in sorted(ranked,key=lambda v:v[0],reverse=True):
        label=labels[group[1]];category='sideways' if label in ('left_side','right_side') else label
        if category not in ('prone','supine','sideways') or counts[category]>=per_posture or group[1] in seen:continue
        path=raw/a['file']
        with np.load(path,allow_pickle=False) as ep:
            q=ep['dof_pos'].astype(float);quat=ep['root_rot_xyzw'].astype(float)
            velocity=ep['dof_vel'].astype(float);names=ep['dof_names'].tolist()
            if posture(quat[0])!=label:continue
            # Avoid selecting the same original body pose copied to many sites.
            if any(c['category']==category and np.sqrt(np.mean((np.array(c['joints'])-q[0])**2))<.12 for c in candidates):continue
            first=dict(q=q[0],quat=quat[0],reset_q=ep['replay_reset_q'].copy(),reset_root=ep['replay_reset_root'].copy())
            end=int(a['metrics']['first_success_frame'])+1
            root=ep['root_pos'][:end].astype(float)
            qa=q[:end].copy();ra=quat[:end].copy()
        with np.load(raw/b['file'],allow_pickle=False) as ep:
            qb=ep['dof_pos'].astype(float);rb=ep['root_rot_xyzw'].astype(float)
            endb=int(b['metrics']['first_success_frame'])+1
            reset_equal=bool(np.array_equal(first['reset_q'],ep['replay_reset_q']) and np.array_equal(first['reset_root'],ep['replay_reset_root']))
            measured_joint_gap=float(np.max(abs(qb[0]-q[0])))
            measured_rotation_gap=float((Rotation.from_quat(quat[0]).inv()*Rotation.from_quat(rb[0])).magnitude())
            rootb=ep['root_pos'][:endb].astype(float)
        if not reset_equal:continue
        heading=float(initialize_heading(torch.tensor(quat[0],dtype=torch.float64)))
        normalized=(Rotation.from_rotvec([0,0,-heading])*Rotation.from_quat(quat[0])).as_quat()
        def body_path(q,r):
            arc=np.r_[0,np.cumsum(np.sqrt(np.mean(np.diff(q,axis=0)**2,axis=1)))];arc/=max(float(arc[-1]),1e-9)
            up=Rotation.from_quat(r).as_matrix()[:,2]
            return np.stack([np.interp(np.linspace(0,1,24),arc,v) for v in np.c_[q,up].T],axis=-1)
        body_distance=_ordered_distance(body_path(qa,ra),body_path(qb[:endb],rb[:endb]))
        name=f'{category}_{counts[category]:02d}_anchor{group[1]}'
        score=yaw/(np.pi/2)+distance/.5+body_distance/.3
        item=dict(id=name,category=category,posture=label,joints=q[0].tolist(),dof_names=names,
            pelvis_quat_xyzw=normalized.tolist(),teacher_score=score,
            source=dict(release=str(release),split='train',setup_identity=list(group),
                raw_files=[a['file'],b['file']],raw_sha256=[a['sha256'],b['sha256']],
                reset_inputs_equal=reset_equal,measured_first_joint_gap_rad=measured_joint_gap,
                measured_first_rotation_gap_rad=measured_rotation_gap,
                first_joint_speed_rms_rad_s=float(np.sqrt(np.mean(velocity[0]**2))),
                heading_gap_deg=float(np.rad2deg(yaw)),path_gap_m=distance,body_path_distance=body_distance,
                terrain=a['terrain_type'],references=[train[a['file']]['source_motion_name'],train[b['file']]['source_motion_name']]))
        candidates.append(item);seen.add(group[1]);counts[category]+=1
        teacher_arrays[name+'_q_a']=qa;teacher_arrays[name+'_quat_a']=ra;teacher_arrays[name+'_root_a']=root
        teacher_arrays[name+'_q_b']=qb[:endb];teacher_arrays[name+'_quat_b']=rb[:endb];teacher_arrays[name+'_root_b']=rootb
        print(f'{name}: teacher heading gap {np.rad2deg(yaw):.1f} deg, route {distance:.2f} m, body {body_distance:.2f}, measured start gap {measured_joint_gap:.3g} rad',flush=True)
        if all(counts[c]>=per_posture for c in ('prone','supine','sideways')):break
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'teacher_candidates.json',dict(schema='hv2.teacher_multimodal_candidates.v1',
        selection='Train split only, selected quality-passing teacher trajectories. Ranked route/heading pairs with identical commanded reset; recorded first-state differences reported. Futures are not claims about the diffusion model.',
        matched_groups=len(ranked),train_episodes=len(train),candidates=candidates))
    np.savez_compressed(output/'teacher_witnesses.npz',**teacher_arrays)
    return candidates


def prepare(output,model):
    import numpy as np
    import torch,yaml
    from omegaconf import OmegaConf
    from g1_deploy.deploy_diffusion import model_settings
    from motion_diffusion.behavioral.hardware.bundle import load_bundle,asset_path
    from motion_diffusion.behavioral.mujoco import flat_model,MujocoNativePlant
    from motion_diffusion.behavioral.start_state import snapshot_start
    cfg=yaml.safe_load((ROOT/'g1_deploy/configs/diffusion.yaml').read_text())
    root,manifest=load_bundle(model_settings(cfg,model)['bundle']);row=manifest['models'][model]
    checkpoint=torch.load(asset_path(root,row['controller']),map_location='cpu',weights_only=True,mmap=True)
    contract=checkpoint['action_vae']['model_contract']
    pd=json.loads(asset_path(root,row['pd_binding']).read_text())['deployment_pd']
    robot=OmegaConf.to_container(OmegaConf.load(asset_path(root,row['controller_training_config'])).robot,resolve=True)
    plant=MujocoNativePlant(flat_model(),contract,pd,robot,command_limits='bounded')
    candidates=json.loads((output/'teacher_candidates.json').read_text())['candidates'];prepared=[]
    for candidate in candidates:
        order=[candidate['dof_names'].index(n) for n in plant.names]
        plant.reset(candidate['posture'],initial_joints=np.array(candidate['joints'])[order],
            initial_quat_xyzw=candidate['pelvis_quat_xyzw'])
        actual=posture(plant.observation().quat_xyzw)
        info=plant.reset_info
        keep=(actual==candidate['posture'] and info['settle_converged']
              and info['passive_root_drift_100ms_m']<.02 and info['passive_joint_drift_100ms_rad']<.15)
        candidate=dict(candidate,rest_preparation=info,actual_posture=actual,eligible=keep)
        if keep:
            snapshot=snapshot_start(plant,source=candidate['source'])
            path=output/'starts'/f'{candidate["id"]}.json';write_json(path,snapshot)
            candidate['start_state']=str(path.resolve())
        prepared.append(candidate)
        print(f'{candidate["id"]}: settled={info["settle_converged"]}, posture={actual}, keep={keep}',flush=True)
    write_json(output/'prepared.json',dict(model=model,candidates=prepared))
    return prepared


def describe_run(directory):
    import numpy as np
    summary=json.loads((directory/'summary.json').read_text())
    config=json.loads((directory/'config.json').read_text())
    with np.load(directory/'trajectory.npz',allow_pickle=False) as f:traj={k:f[k] for k in f.files}
    end=round((summary['first_upright_hold_s'] or summary['simulated_seconds'])/.02)
    end=max(2,min(end,len(traj['q'])))
    q=traj['q'][:end];g=traj['gravity'][:end]
    arc=np.r_[0,np.cumsum(np.sqrt(np.mean(np.diff(q,axis=0)**2,axis=1)))];arc/=max(float(arc[-1]),1e-9)
    knots=np.linspace(0,1,24)
    body=np.stack([np.interp(knots,arc,v) for v in np.c_[q,g].T],axis=-1)
    root=traj['root_position'][:end,:2];root=root-root[0]
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(root,axis=0),axis=1))];distance/=max(float(distance[-1]),1e-9)
    route=np.stack([np.interp(knots,distance,v) for v in root.T],axis=-1)
    speed=float(np.max(abs(traj['dq'])))
    tail=float(np.sqrt(np.mean(traj['dq'][-50:]**2)))
    quiet=bool(summary['upright_hold_1s'] and summary['final_second_upright_fraction']>=.9 and tail<.5)
    usable=bool(quiet and speed<25. and not summary['physics_warnings'] and not summary['saturated_action_ticks'])
    heading=float(np.median(np.unwrap(traj['causal_heading_rad'][max(0,end-25):end])))
    initial=hashlib.sha256();noise=hashlib.sha256()
    with np.load(directory/'first_condition.npz',allow_pickle=False) as f:
        for key in sorted(f.files):
            digest=noise if key in ('state_noise','code_noise') else initial
            a=f[key];digest.update(key.encode());digest.update(str((a.shape,a.dtype.str)).encode());digest.update(a.tobytes())
    return dict(directory=str(directory.resolve()),seed=config['seed'],
        first_condition_sha256=initial.hexdigest(),first_noise_sha256=noise.hexdigest(),
        success=summary['upright_hold_1s'],quiet_completion=quiet,usable=usable,
        recovery_hold_s=summary['first_upright_hold_s'],max_joint_speed_rad_s=speed,
        final_joint_speed_rms_rad_s=tail,max_raw_action=summary['max_raw_action'],
        saturated_ticks=summary['saturated_action_ticks'],physics_warnings=summary['physics_warnings'],
        heading_rad=heading,body=body.tolist(),route=route.tolist())


def rank_pairs(runs):
    import numpy as np
    from scripts.fall_recovery.datasets.repertoire import _ordered_distance
    pairs=[]
    for a,b in combinations(runs,2):
        if not a['usable'] or not b['usable']:continue
        if a['first_condition_sha256']!=b['first_condition_sha256']:
            raise ValueError('Candidate comparison changed the initial model condition, not only noise')
        if a['first_noise_sha256']==b['first_noise_sha256']:continue
        aa,bb=np.array(a['body']),np.array(b['body'])
        joints=_ordered_distance(aa[:,:29],bb[:,:29]);up=_ordered_distance(aa[:,29:],bb[:,29:])
        route=float(np.sqrt(np.mean(np.sum((np.array(a['route'])-b['route'])**2,axis=1))))
        yaw=yaw_gap(a['heading_rad'],b['heading_rad'])
        pairs.append(dict(seeds=[a['seed'],b['seed']],directories=[a['directory'],b['directory']],
            score=joints/.3+up/.4+route/.5+yaw/(np.pi/2),joint_path_distance_rad=joints,
            up_path_distance=up,route_distance_m=route,heading_gap_deg=float(np.rad2deg(yaw))))
    return sorted(pairs,key=lambda r:r['score'],reverse=True)


def search(output,model,backend,seeds,seconds,ids=None):
    import subprocess
    import yaml
    from g1_deploy.deploy_diffusion import model_settings
    from motion_diffusion.behavioral.hardware.bundle import load_bundle
    settings=model_settings(yaml.safe_load((ROOT/'g1_deploy/configs/diffusion.yaml').read_text()),model)
    _,manifest=load_bundle(settings['bundle']);binding=manifest['models'][model]
    for prior in (output/'trials').glob('*/*/*/config.json'):
        config=json.loads(prior.read_text())
        if config['model']!=model or any(config[k+'_sha256']!=binding[k]['sha256'] for k in ('controller','planner')) or config['reference_inference']!=binding['reference_inference']:
            raise ValueError('This search directory contains a different model or sampling recipe; use a new --output')
    prepared=json.loads((output/'prepared.json').read_text()) if (output/'prepared.json').exists() else None
    candidates=prepare(output,model) if prepared is None else prepared['candidates']
    if prepared is not None and prepared['model']!=model:raise ValueError('Prepared starts belong to a different model')
    results=[]
    for candidate in candidates:
        if not candidate['eligible'] or (ids and candidate['id'] not in ids):continue
        start=Path(candidate['start_state'])
        signature=hashlib.sha256(json.dumps(dict(seeds=seeds,seconds=seconds,model=model,backend=backend,
            start_sha256=hashlib.sha256(start.read_bytes()).hexdigest()),sort_keys=True).encode()).hexdigest()[:12]
        directory=output/'trials'/candidate['id']/signature
        directory.parent.mkdir(parents=True,exist_ok=True)
        complete=all((directory/f'{model}_{i:03d}'/'summary.json').exists() for i in range(len(seeds)))
        if not complete:
            if directory.exists():
                import shutil,time
                archive=output/'diagnostics'/f'incomplete_{candidate["id"]}_{time.time_ns()}'
                archive.parent.mkdir(parents=True,exist_ok=True)
                shutil.move(str(directory),str(archive))
            command=[sys.executable,str(ROOT/'g1_deploy/sim_diffusion.py'),'--model',model,'--backend',backend,
                '--start',candidate['posture'],'--start-state',str(start),'--seed-list',*[str(s) for s in seeds],
                '--sensor-seed','42','--pose-seed','42','--seconds',str(seconds),'--output',str(directory)]
            print(f'Testing {candidate["id"]}: seeds {seeds}',flush=True)
            with (directory.parent/(signature+'.log')).open('w') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        runs=[]
        current_config=json.loads(next(directory.glob('*/config.json')).read_text())
        comparable=('model','backend','planner_sha256','controller_sha256','inference',
                    'command_limit_mode','sensor_seed','tracker_startup','depth_startup',
                    'planner_precision','engine_mode','fixed_start_snapshot')
        for sweep in sorted(directory.parent.iterdir()):
            if not sweep.is_dir() or not (sweep/'summary.json').exists():continue
            for trial in sorted(sweep.iterdir()):
                if trial.is_dir() and (trial/'summary.json').exists() and (trial/'first_condition.npz').exists():
                    config=json.loads((trial/'config.json').read_text())
                    if any(config.get(k)!=current_config.get(k) for k in comparable):continue
                    # Older receipts lack requested_seconds. Their longest sibling
                    # supplies the duration; new receipts retain early failures too.
                    sweep_seconds=config.get('requested_seconds')
                    if sweep_seconds is None:
                        sweep_seconds=max(r['simulated_seconds'] for r in json.loads((sweep/'summary.json').read_text()).values())
                    if abs(sweep_seconds-seconds)>.021:continue
                    runs.append(describe_run(trial))
        # Rerunning an already tested seed does not add another mode.
        unique={r['seed']:r for r in runs};runs=list(unique.values())
        if len({r['first_condition_sha256'] for r in runs})>1:
            raise ValueError(f'{candidate["id"]}: starting conditions differ across seeds')
        pairs=rank_pairs(runs)
        row=dict(id=candidate['id'],category=candidate['category'],posture=candidate['posture'],
            start_state=str(start.resolve()),source=candidate['source'],runs=runs,pairs=pairs,
            usable=sum(r['usable'] for r in runs),tested=len(runs))
        results.append(row)
        write_json(output/'trials'/candidate['id']/'analysis.json',row)
        print(f'{candidate["id"]}: {row["usable"]}/{row["tested"]} usable; best pair '+
              (str(pairs[0]) if pairs else 'none'),flush=True)
    return results


def export_catalog(output,destination, *, selected_ids=None):
    """Package small starts/receipts; the raw teacher corpus is not needed to show them."""
    import shutil
    rows=[json.loads(p.read_text()) for p in (output/'trials').glob('*/analysis.json')]
    selected=[]
    for category in ('prone','supine','sideways'):
        choices=[r for r in rows if r['category']==category and r['pairs'] and
                 (not selected_ids or r['id'] in selected_ids)]
        if not choices:raise ValueError(f'No usable contrasting pair for {category}')
        best=max(choices,key=lambda r:r['pairs'][0]['score'])
        pair=best['pairs'][0];source=Path(best['start_state'])
        destination.mkdir(parents=True,exist_ok=True)
        path=destination/f'{category}.json';shutil.copyfile(source,path)
        selected.append(dict(category=category,posture=best['posture'],candidate_id=best['id'],
            start_state=path.name,start_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            seeds=pair['seeds'],metrics={k:v for k,v in pair.items() if k not in ('directories','seeds')},
            search_tested=best['tested'],search_usable=best['usable'],
            initial_condition_sha256=best['runs'][0]['first_condition_sha256'],
            source=best['source'],selected_trials=pair['directories']))
    model=json.loads((output/'prepared.json').read_text())['model']
    first=Path(selected[0]['selected_trials'][0]);config=json.loads((first/'config.json').read_text())
    seconds=float(json.loads((first/'summary.json').read_text())['simulated_seconds'])
    for selected_start in selected:
        for directory in selected_start['selected_trials']:
            directory=Path(directory);other=json.loads((directory/'config.json').read_text())
            for key in ('model','backend','controller_sha256','planner_sha256','reference_inference',
                        'planner_precision','engine_mode','sensor_seed','command_limit_mode','depth_startup','tracker_startup'):
                if other[key]!=config[key]:raise ValueError(f'Selected demonstrations differ in {key}; export a consistent set')
            if abs(float(json.loads((directory/'summary.json').read_text())['simulated_seconds'])-seconds)>.021:
                raise ValueError('Selected demonstrations have different evaluation durations')
    write_json(destination/'catalog.json',dict(schema='hv2.multimodal_demo.v1',model=model,
        backend=config['backend'],controller_sha256=config['controller_sha256'],planner_sha256=config['planner_sha256'],
        reference_inference=config['reference_inference'],
        planner_precision=config['planner_precision'],engine_mode=config['engine_mode'],
        sensor_seed=config['sensor_seed'],seconds=seconds,command_limits=config['command_limit_mode'],
        depth_startup=config['depth_startup'],tracker_startup=config['tracker_startup'],
        source_search=str(output.resolve()),
        scope='Selected simulated demonstrations, not an unbiased success benchmark or hardware-tested starting poses. Teacher starts are transferred to flat MuJoCo and passively settled; each seed restores exactly the same resting state and initial model inputs.',
        starts=selected))
    print(f'Demo catalog: {destination/"catalog.json"}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,default=RELEASE)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mine',action='store_true')
    p.add_argument('--search',action='store_true')
    p.add_argument('--candidates-per-posture',type=int,default=4)
    p.add_argument('--model',default='r8_s25_diffusion')
    p.add_argument('--backend',choices=('tensorrt','pytorch'),default='tensorrt')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45])
    p.add_argument('--seconds',type=float,default=12.)
    p.add_argument('--ids',nargs='+')
    p.add_argument('--export',type=Path,help='Write a portable demo catalog after inspecting the search results')
    args=p.parse_args()
    if args.mine:mine(args.release,args.output,args.candidates_per_posture)
    if args.search:search(args.output,args.model,args.backend,args.seeds,args.seconds,args.ids)
    if args.export:export_catalog(args.output,args.export,selected_ids=args.ids)
    if not (args.mine or args.search or args.export):p.error('Select --mine, --search or --export')


if __name__=='__main__':main()
