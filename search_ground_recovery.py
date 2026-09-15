#!/usr/bin/env python3
"""Search teacher starts and diffusion noise for different *ground* recoveries."""
from pathlib import Path
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from g1_deploy.search_multimodal_recovery import RELEASE,write_json,posture,prepare,search


def analyze_ground(output):
    import numpy as np
    from scripts.fall_recovery.datasets.repertoire import _ordered_distance
    results=[]
    for path in sorted((output/'trials').glob('*/analysis.json')):
        row=json.loads(path.read_text());paths={}
        if row['category'] not in ('prone','supine'):continue
        for run in row['runs']:
            if not run['usable']:continue
            with np.load(Path(run['directory'])/'trajectory.npz') as f:
                height=f['root_position'][:,2]-f['terrain_height_m']
                descriptor,end=ground_path(f['q'],f['gravity'],height)
                paths[run['seed']]=(descriptor,run)
                run['ground_duration_s']=end*.02
        pairs=[]
        for (a,(pa,ra)),(b,(pb,rb)) in combinations(paths.items(),2):
            if ra['first_condition_sha256']!=rb['first_condition_sha256']:raise ValueError('Different initial conditions')
            joint=_ordered_distance(pa[:,:23],pb[:,:23]);up=_ordered_distance(pa[:,23:],pb[:,23:])
            pairs.append(dict(seeds=[a,b],score=joint/.3+up/.15,joint_ground_path_rad=joint,
                up_ground_path_distance=up,directories=[ra['directory'],rb['directory']]))
        pairs.sort(key=lambda x:x['score'],reverse=True)
        selected=[]
        if pairs:
            selected=list(pairs[0]['seeds'])
            distances={tuple(sorted(p['seeds'])):p['score'] for p in pairs}
            while len(selected)<min(6,len(paths)):
                candidates=[(min(distances[tuple(sorted((s,t)))] for t in selected),s) for s in paths if s not in selected]
                score,seed=max(candidates)
                if score<.75:break
                selected.append(seed)
        result=dict(id=row['id'],category=row['category'],tested=row['tested'],usable=row['usable'],
            seeds=selected,pairs=pairs,runs=row['runs'],start_state=row['start_state'],source=row['source'])
        write_json(path.parent/'ground_analysis.json',result);results.append(result)
    results.sort(key=lambda r:r['pairs'][0]['score'] if r['pairs'] else -1,reverse=True)
    write_json(output/'ground_ranking.json',dict(candidates=results,
        ranking='Dynamic-time-warped ground joint/body-up paths, stops before upright rise, no final yaw reward. Farthest-point seed selection is a demo heuristic, not a mode-count estimate.'))
    for category in ('prone','supine'):
        print(category,[(r['id'],r['usable'],r['tested'],r['seeds'],round(r['pairs'][0]['score'],2)) for r in results if r['category']==category and r['pairs']][:8],flush=True)
    return results


def render_examples(output,ids):
    import os
    os.environ.setdefault('MUJOCO_GL','egl')
    import cv2,mujoco,numpy as np
    from motion_diffusion.behavioral.mujoco import flat_model
    model=flat_model();data=mujoco.MjData(model);renderer=mujoco.Renderer(model,300,400)
    camera=mujoco.MjvCamera();camera.distance=2.3;camera.azimuth=130;camera.elevation=-35;camera.lookat[:]=[0,0,.25]
    opt=mujoco.MjvOption();opt.geomgroup[:]=[1,1,0,0,0,0]
    directory=output/'review';directory.mkdir(exist_ok=True)
    try:
        for name in ids:
            row=json.loads((output/'trials'/name/'ground_analysis.json').read_text());images=[]
            for seed in row['seeds']:
                run=next(r for r in row['runs'] if r['seed']==seed)
                with np.load(Path(run['directory'])/'trajectory.npz') as f:
                    qpos=f['qpos'];frames=np.linspace(0,round(run['ground_duration_s']/.02)-1,5).round().astype(int)
                    frames=np.r_[frames,min(len(qpos)-1,round(run['recovery_hold_s']/.02))]
                    strips=[]
                    for frame in frames:
                        data.qpos[:]=qpos[frame];mujoco.mj_forward(model,data)
                        renderer.update_scene(data,camera=camera,scene_option=opt)
                        im=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
                        cv2.putText(im,f'{name} seed {seed} t={frame*.02:.2f}s',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.37,(255,255,255),1)
                        strips.append(im)
                    images.append(np.concatenate(strips,axis=1))
            if images:cv2.imwrite(str(directory/f'{name}.jpg'),np.concatenate(images,axis=0))
    finally:renderer.close()


def export_ground(output,destination,ids,seeds_override=None):
    """Export reviewed exact simulation starts and matching optional joint goals."""
    import hashlib,shutil
    import numpy as np
    import torch,yaml
    from g1_deploy.deploy_diffusion import model_settings,hardware_settings
    from motion_diffusion.behavioral.hardware import bundle
    from motion_diffusion.behavioral.hardware.unitree import NativePDContract
    from motion_diffusion.behavioral.hardware.start_pose import validate_start_pose
    settings=model_settings(yaml.safe_load((ROOT/'g1_deploy/configs/diffusion.yaml').read_text()),'r8_s25_diffusion')
    root,manifest=bundle.load_bundle(settings['bundle']);model=manifest['models']['r8_s25_diffusion']
    checkpoint=torch.load(bundle.asset_path(root,model['controller']),map_location='cpu',weights_only=True,mmap=True)
    contract=checkpoint['action_vae']['model_contract'];hardware=hardware_settings(settings,contract)
    pd=json.loads(bundle.asset_path(root,model['pd_binding']).read_text())['deployment_pd']
    hardware['pd']['training_pd']=pd;contract=dict(contract,deployment_pd=pd)
    pd=NativePDContract(contract,hardware['pd']);selected=[];configs=[]
    for name in ids:
        row=json.loads((output/'trials'/name/'ground_analysis.json').read_text())
        seeds=seeds_override.get(row['category'],row['seeds']) if seeds_override else row['seeds']
        if len(seeds)<2:raise ValueError('Need at least two reviewed successful seeds')
        runs=[next(r for r in row['runs'] if r['seed']==s and r['usable']) for s in seeds]
        if len({r['first_condition_sha256'] for r in runs})!=1:raise ValueError('Different initial conditions')
        source=Path(row['start_state']);snapshot=json.loads(source.read_text());category=row['category']
        destination.mkdir(parents=True,exist_ok=True);target=destination/f'{category}.json';shutil.copyfile(source,target)
        held_target=snapshot.get('preparation',{}).get('pd_target_joints_rad',snapshot['joint_positions_rad'])
        q=np.array([held_target[n] for n in pd.dof_names]);goal=np.clip(q,pd.lower,pd.upper)
        pose=dict(schema='hv2.joint_start_pose.v1',name=f'{category}_multimodal_{name}',posture=category,
            joint_positions_rad=dict(zip(pd.dof_names,goal.tolist())),source=snapshot['source'],
            simulation_start_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            joint_limit_adjustment_max_rad=float(np.max(abs(goal-q))),
            scope='Optional slow joint positioning from an already prone/supine physical robot. Simulation-derived, not hardware-validated; does not reset base pose, contact forces, camera or estimator history.')
        validate_start_pose(pose,pd);write_json(destination/f'{category}_robot.json',pose)
        reference=Path(runs[0]['directory']);config=json.loads((reference/'config.json').read_text());configs.append(config)
        selected.append(dict(category=category,posture=category,candidate_id=name,start_state=target.name,
            start_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),robot_start_pose=f'{category}_robot.json',
            seeds=seeds,ground_metrics=row['pairs'][0],search_tested=row['tested'],search_usable=row['usable'],
            selected_trials=[r['directory'] for r in runs],source=row['source'],
            initial_condition_sha256=runs[0]['first_condition_sha256']))
    if sorted(r['category'] for r in selected)!=['prone','supine']:raise ValueError('Select one prone and one supine start')
    keys=('backend','controller_sha256','planner_sha256','reference_inference','planner_precision','engine_mode',
          'sensor_seed','command_limit_mode','depth_startup','tracker_startup')
    if any(any(c[k]!=configs[0][k] for k in keys) for c in configs):raise ValueError('Different evaluated contracts')
    config=configs[0]
    duration=float(json.loads((Path(selected[0]['selected_trials'][0])/'summary.json').read_text())['simulated_seconds'])
    catalog=dict(schema='hv2.multimodal_demo.v1',model='r8_s25_diffusion',**{k:config[k] for k in keys if k!='command_limit_mode'},
        seconds=duration,command_limits=config['command_limit_mode'],source_search=str(output.resolve()),
        scope='Selected ground-path demonstrations from one fixed state per posture. Seed diversity is not an estimated mode count or a hardware prediction.',starts=selected)
    write_json(destination/'catalog.json',catalog)


def check_robot_preparation(catalog_path,output):
    """Rehearse the existing slow positioning path in MuJoCo, without motors."""
    import numpy as np
    import torch,yaml
    from omegaconf import OmegaConf
    from g1_deploy.deploy_diffusion import model_settings
    from motion_diffusion.behavioral.hardware.bundle import load_bundle,asset_path
    from motion_diffusion.behavioral.hardware.start_pose import start_pose_path
    from motion_diffusion.behavioral.mujoco import flat_model,MujocoNativePlant
    from motion_diffusion.behavioral.start_state import snapshot_start,restore_start
    catalog=json.loads(catalog_path.read_text());output.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((ROOT/'g1_deploy/configs/diffusion.yaml').read_text())
    root,manifest=load_bundle(model_settings(cfg,catalog['model'])['bundle']);model=manifest['models'][catalog['model']]
    checkpoint=torch.load(asset_path(root,model['controller']),map_location='cpu',weights_only=True,mmap=True)
    pd=json.loads(asset_path(root,model['pd_binding']).read_text())['deployment_pd']
    robot=OmegaConf.to_container(OmegaConf.load(asset_path(root,model['controller_training_config'])).robot,resolve=True)
    plant=MujocoNativePlant(flat_model(),checkpoint['action_vae']['model_contract'],pd,robot,command_limits='bounded')
    results=[]
    for row in catalog['starts']:
        pose=json.loads((catalog_path.parent/row['robot_start_pose']).read_text())
        goal=np.array([pose['joint_positions_rad'][n] for n in plant.names])
        for initial in ('ordinary','selected'):
            if initial=='ordinary':plant.reset(row['posture'])
            else:restore_start(plant,json.loads((catalog_path.parent/row['start_state']).read_text()))
            q0=plant.observation().q;_,duration=start_pose_path(q0,goal,0.);positions=[];speed=[]
            for tick in range(round((duration+2.)/.02)):
                target,_=start_pose_path(q0,goal,tick*.02);plant.step((target-plant.default)/plant.scale)
                positions.append(plant.data.qpos.copy());speed.append(plant.observation().dq)
            obs=plant.observation();result=dict(category=row['category'],initial=initial,
                final_posture=posture(obs.quat_xyzw),max_joint_error_rad=float(np.max(abs(obs.q-goal))),
                final_joint_speed_rms_rad_s=float(np.sqrt(np.mean(obs.dq**2))),
                max_joint_speed_rad_s=float(np.max(abs(np.array(speed)))),root_displacement_m=float(np.linalg.norm(positions[-1][:3]-positions[0][:3])))
            results.append(result);print(result,flush=True)
            np.savez_compressed(output/f'{row["category"]}_{initial}.npz',qpos=np.array(positions),dq=np.array(speed))
            if initial=='selected':
                plant.data.qvel[:]=0.;plant.data.time=0.
                write_json(output/f'{row["category"]}_prepared.json',snapshot_start(plant,source=dict(
                    from_catalog=str(catalog_path.resolve()),preparation_target=pose,result=result)))
    write_json(output/'preparation.json',dict(scope='MuJoCo slow-positioning rehearsal, not hardware validation',results=results))


def ground_path(q,gravity,height,contacts=None):
    """Stop before the upright rise; align by body/joint progress, not time."""
    import numpy as np
    lifted=(height>.4)&(-gravity[:,2]>.55)
    held=np.convolve(lifted.astype(int),np.ones(8,dtype=int),'valid') if len(lifted)>=8 else np.zeros(0)
    indices=np.flatnonzero(held==8)
    end=max(2,int(indices[0])+1) if len(indices) else len(q)
    q=q[:end];gravity=gravity[:end]
    # Wrist rotations do not define a recovery strategy.
    keep=[i for i in range(29) if i not in (19,20,21,26,27,28)]
    joints=q[:,keep]
    progress=np.r_[0,np.cumsum(np.linalg.norm(np.diff(np.c_[joints/4,gravity],axis=0),axis=1))]
    progress/=max(progress[-1],1e-9)
    values=np.c_[joints,gravity]
    if contacts is not None:values=np.c_[values,contacts[:end]]
    path=np.stack([np.interp(np.linspace(0,1,24),progress,v) for v in values.T],axis=-1)
    return path,end


def prepare_held(output):
    """Settle on the floor while holding source joints, as robot START does."""
    import numpy as np
    import torch,yaml
    from omegaconf import OmegaConf
    from g1_deploy.deploy_diffusion import model_settings
    from motion_diffusion.behavioral.hardware.bundle import load_bundle,asset_path
    from motion_diffusion.behavioral.mujoco import flat_model,MujocoNativePlant
    from motion_diffusion.behavioral.start_state import snapshot_start
    cfg=yaml.safe_load((ROOT/'g1_deploy/configs/diffusion.yaml').read_text())
    root,manifest=load_bundle(model_settings(cfg,'r8_s25_diffusion')['bundle']);model=manifest['models']['r8_s25_diffusion']
    checkpoint=torch.load(asset_path(root,model['controller']),map_location='cpu',weights_only=True,mmap=True)
    pd=json.loads(asset_path(root,model['pd_binding']).read_text())['deployment_pd']
    robot=OmegaConf.to_container(OmegaConf.load(asset_path(root,model['controller_training_config'])).robot,resolve=True)
    plant=MujocoNativePlant(flat_model(),checkpoint['action_vae']['model_contract'],pd,robot,command_limits='bounded')
    candidates=json.loads((output/'teacher_candidates.json').read_text())['candidates'];prepared=[]
    for candidate in candidates:
        order=[candidate['dof_names'].index(n) for n in plant.names]
        goal=np.array(candidate['joints'])[order]
        limits=plant.model.jnt_range[plant.joints];goal=np.clip(goal,limits[:,0],limits[:,1])
        plant.reset(candidate['posture'],initial_joints=goal,initial_quat_xyzw=candidate['pelvis_quat_xyzw'],settle=False)
        quiet=0
        for tick in range(400):
            plant.step((goal-plant.default)/plant.scale)
            quiet=quiet+1 if np.max(abs(plant.data.qvel))<.08 else 0
            if tick>=149 and quiet>=25:break
        actual=posture(plant.observation().quat_xyzw)
        info=dict(plant.reset_info,preparation_mode='source_joint_pd_hold',settled=True,
            settle_simulated_s=(tick+1)*.02,settle_speed_before_zero=float(np.max(abs(plant.data.qvel))),
            settle_converged=quiet>=25,initial_contacts=int(plant.data.ncon),
            pd_target_joints_rad=dict(zip(plant.names,goal.tolist())),
            held_joint_error_rms_rad=float(np.sqrt(np.mean((plant.observation().q-goal)**2))),
            held_joint_error_max_rad=float(np.max(abs(plant.observation().q-goal))))
        keep=bool(actual==candidate['posture'] and quiet>=25 and plant.data.qpos[plant.root_q+2]<.35)
        item=dict(candidate,rest_preparation=info,actual_posture=actual,eligible=keep)
        if keep:
            plant.data.qvel[:]=0.;plant.data.time=0.;plant.reset_info=info
            snapshot=snapshot_start(plant,source=candidate['source'])
            path=output/'starts'/f'{candidate["id"]}.json';write_json(path,snapshot);item['start_state']=str(path.resolve())
        prepared.append(item)
        print(f'{candidate["id"]}: PD hold, posture={actual}, quiet={quiet>=25}, keep={keep}, joint RMS {info["held_joint_error_rms_rad"]:.3f}',flush=True)
    write_json(output/'prepared.json',dict(model='r8_s25_diffusion',preparation='source_joint_pd_hold',candidates=prepared))
    return prepared


def mine_ground(release,output,count):
    import numpy as np
    from scipy.spatial.transform import Rotation
    import torch
    from humanoidverse2.agents.action_vae.heading import initialize_heading
    packs=release/'packs/site_heldout_15k'
    manifest=json.loads((packs/'split_manifest.json').read_text())
    train={r['raw_file']:r for r in manifest['episodes'] if r['split']=='train'}
    audit=json.loads((packs/'repertoire_selection.json').read_text())
    rows=[r for r in audit['episodes'] if r['file'] in train and r.get('accepted') and r.get('execution')]
    output.mkdir(parents=True,exist_ok=True);cache=output/'teacher_ground_frames.npz'
    def extract(row):
        from io import BytesIO
        # One sequential remote read avoids a separate network seek for every
        # array in the zip container.
        with np.load(BytesIO((release/'work/full/raw'/row['file']).read_bytes()),allow_pickle=False) as f:
            q=f['dof_pos'].astype(float);quat=f['root_rot_xyzw'].astype(float)
            n=min(len(q),int(row['metrics']['first_success_frame'])+1)
            q=q[:n];quat=quat[:n];g=-Rotation.from_quat(quat).as_matrix()[:,2,:]
            names=f['body_names'].tolist();pelvis=names.index('pelvis') if 'pelvis' in names else 0
            height=f['link_terrain_clearance'][:n,pelvis]
            contact_names=['left_ankle_roll_link','right_ankle_roll_link','left_knee_link','right_knee_link','left_wrist_yaw_link','right_wrist_yaw_link']
            contacts=f['contacts'][:n,[names.index(k) for k in contact_names]]>0
            velocities=f['dof_vel'][:n];speeds=np.sqrt(np.mean(velocities**2,axis=1));dt=float(f['control_dt'])
            candidates=[0]
            # Look inside the realized ground phase, not just the episode reset.
            for lo in range(10,min(n-15,150),20):
                possible=[i for i in range(lo,min(lo+20,n-15)) if height[i]<.3 and abs(g[i,0])>.8]
                if possible:candidates.append(min(possible,key=lambda i:speeds[i]))
            selected=[]
            for frame in candidates:
                label=posture(quat[frame])
                if label not in ('prone','supine') or height[frame]>.3:continue
                if frame and (speeds[frame]>4. or any(np.sqrt(np.mean((a['q']-q[frame])**2))<.2 for a in selected)):continue
                path,end=ground_path(q[frame:],g[frame:],height[frame:],contacts[frame:])
                selected.append(dict(file=row['file'],frame=frame,posture=label,q=q[frame],quat=quat[frame],path=path,
                    names=f['dof_names'].tolist(),duration=end*dt,speed=float(speeds[frame])))
            return selected
    if cache.exists():
        data=np.load(cache,allow_pickle=False);bank=[]
        for i,name in enumerate(data['file'].tolist()):
            bank.append(dict(file=name,posture=str(data['posture'][i]),q=data['q'][i],quat=data['quat'][i],
                frame=int(data['frame'][i]),path=data['path'][i],names=data['names'].tolist(),duration=float(data['duration'][i]),speed=float(data['speed'][i])))
    else:
        bank=[]
        with ThreadPoolExecutor(max_workers=16) as pool:
            for i,r in enumerate(pool.map(extract,rows)):
                bank.extend(r)
                if i%250==0:print(f'Teacher ground paths: {i}/{len(rows)} read; {len(bank)} prone/supine frames',flush=True)
        np.savez_compressed(cache,**{key:np.array([r[key] for r in bank]) for key in ('file','frame','posture','q','quat','path','duration','speed')},names=bank[0]['names'])
    lookup={r['file']:r for r in rows}
    from scipy.spatial import cKDTree
    features=np.array([np.r_[r['q']/np.sqrt(29),-Rotation.from_quat(r['quat']).as_matrix()[2,:]] for r in bank])
    if len(bank)<2:raise ValueError('Fewer than two grounded teacher states')
    tree=cKDTree(features);_,neighbors=tree.query(features,k=min(24,len(bank)),workers=4)
    ranked=[]
    for i,nearby in enumerate(neighbors):
        a=bank[i];key=tuple(lookup[a['file']]['execution']['setup_identity'])
        for j in nearby:
            if j<=i:continue
            b=bank[j]
            if a['posture']!=b['posture']:continue
            if lookup[a['file']]['motion_id']==lookup[b['file']]['motion_id']:continue
            if np.sqrt(np.mean((a['q']-b['q'])**2))>.2 or np.linalg.norm(features[i,-3:]-features[j,-3:])>.15:continue
            qa,qb=a['path'],b['path']
            joint=float(np.sqrt(np.mean((qa[:,:23]-qb[:,:23])**2)))
            up=float(np.sqrt(np.mean((qa[:,23:26]-qb[:,23:26])**2)))
            contact=float(np.mean(abs(qa[:,26:]-qb[:,26:])))
            score=up/.2+joint/.35+contact/.3
            ranked.append((score,key,a,b,dict(joint=joint,up=up,contact=contact)))
    chosen=[];counts=defaultdict(int)
    for score,key,a,b,metrics in sorted(ranked,key=lambda v:v[0],reverse=True):
        category=a['posture']
        if counts[category]>=count:continue
        # Choose genuinely different initial limb arrangements across the corpus.
        if any(c['category']==category and np.sqrt(np.mean((np.array(c['joints'])-a['q'])**2))<.18 for c in chosen):continue
        heading=float(initialize_heading(torch.tensor(a['quat'],dtype=torch.float64)))
        rotation=(Rotation.from_rotvec([0,0,-heading])*Rotation.from_quat(a['quat'])).as_quat()
        row=dict(id=f'{category}_{counts[category]:02d}_anchor{key[1]}',category=category,posture=category,
            joints=a['q'].tolist(),dof_names=a['names'],pelvis_quat_xyzw=rotation.tolist(),teacher_score=score,
            source=dict(release=str(release),split='train',setup_identity=list(key),
                raw_files=[a['file'],b['file']],raw_sha256=[lookup[a['file']]['sha256'],lookup[b['file']]['sha256']],
                source_frames=[a['frame'],b['frame']],
                references=[train[a['file']]['source_motion_name'],train[b['file']]['source_motion_name']],
                ground_path_differences=metrics,ground_duration_s=[a['duration'],b['duration']],
                measured_first_joint_gap_rad=float(np.max(abs(a['q']-b['q']))),
                first_joint_speed_rms_rad_s=a['speed']))
        chosen.append(row);counts[category]+=1
        print(f'{row["id"]}: ground score {score:.2f}, {metrics}',flush=True)
    write_json(output/'teacher_candidates.json',dict(schema='hv2.teacher_multimodal_candidates.v1',
        selection='All accepted training realizations, initial and intermediate ground frames. Nearby measured states (joint RMS <=0.2 rad, body-up gap <=0.15) with different reference futures ranked by ground joint, body-up and contact paths, excluding final heading. These teacher neighbors are approximate, not identical; diffusion trials restore one exact state.',
        train_episodes=len(train),ground_episodes=len({r['file'] for r in bank}),ground_frames=len(bank),ground_pairs=len(ranked),candidates=chosen))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,default=RELEASE);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mine',action='store_true');p.add_argument('--prepare',action='store_true');p.add_argument('--prepare-held',action='store_true');p.add_argument('--search',action='store_true');p.add_argument('--analyze',action='store_true');p.add_argument('--render',action='store_true')
    p.add_argument('--candidates',type=int,default=20);p.add_argument('--ids',nargs='+')
    p.add_argument('--seeds',type=int,nargs='+',default=[101,202,303,404]);p.add_argument('--seconds',type=float,default=12.)
    p.add_argument('--export',type=Path)
    p.add_argument('--check-robot-preparation',type=Path,metavar='CATALOG')
    args=p.parse_args()
    if args.mine:mine_ground(args.release,args.output,args.candidates)
    if args.prepare:prepare(args.output,'r8_s25_diffusion')
    if args.prepare_held:prepare_held(args.output)
    if args.search:search(args.output,'r8_s25_diffusion','tensorrt',args.seeds,args.seconds,args.ids)
    if args.search or args.analyze:analyze_ground(args.output)
    if args.render:render_examples(args.output,args.ids or [])
    if args.export:export_ground(args.output,args.export,args.ids or [])
    if args.check_robot_preparation:check_robot_preparation(args.check_robot_preparation,args.output)


if __name__=='__main__':main()
