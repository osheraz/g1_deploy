"""Import the exact September 21 terrain gallery into local deployment formats.

No sensors or motors. Original assets stay byte-identical in the cache; goals
select one original batch lane without changing its costs or physical clock.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from steerable_recovery.paths import ROOT, CONFIG_ROOT
from steerable_recovery.settings import STAGE3_MODELS
from motion_diffusion.behavioral.hardware import bundle

CACHE = ROOT / 'logs/native_recovery/terrain_turn_gallery_20260921'
OUTPUT = ROOT / 'logs/native_recovery/bundles/stage3_40k_20260921'
CATALOG = CONFIG_ROOT / 'steering/terrain_gallery/catalog.json'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def extract_goal(task, target_file, index, *, provenance):
    options=deepcopy(task)
    binding=options['steering_targets']
    if binding.get('apply_guidance') is not True:
        raise ValueError('Gallery steering target must explicitly enable guidance')
    goal=dict(schema='hv2.hardware_motion_goal.v1',clock='physical_elapsed_s',
        frame='initial_pelvis_origin_causal_heading',mode=binding['mode'],
        guidance_scale=options['guidance_scale'],guidance_max_grad_norm=options['guidance_max_grad_norm'],
        position_error_scale_m=binding['position_error_scale_m'],
        yaw_error_scale_deg=binding['yaw_error_scale_deg'],
        target=deepcopy(target_file['targets'][index]),provenance=deepcopy(provenance),
        hardware_qualified=False)
    from motion_diffusion.behavioral.hardware.steering import motion_goal
    motion_goal(goal)
    return goal


def import_models(cache, output):
    source=cache/'bundle'; models=read(source/'MODELS.json')
    if (models.get('schema')!='hv2.stage3_seven_simulation.v1' or
            models.get('completed') is not True or models.get('updates')!=40000):
        raise ValueError('Expected completed seven-model 40k package')
    for name,digest in models['files'].items():
        if bundle.sha256(source/name)!=digest:raise ValueError(f'Model package hash mismatch: {name}')
    old_root,old=bundle.load_bundle(ROOT/'logs/native_recovery/bundles/s25r_40k_20260921')
    original=old['models']['s25r_diffusion']
    if original['controller']['sha256']!=models['shared']['tracker']['sha256']:
        raise ValueError('Gallery tracker differs from the verified S25R PD binding')
    config=bundle.asset_path(old_root,original['controller_training_config'])
    rows={}
    for source_name,name in STAGE3_MODELS.items():
        cp=models['models'][source_name]['checkpoint']; controller=models['shared']['tracker']
        rows[name]=dict(label=f'{source_name} diffusion40k + P42 depth tracker22k',objective='diffusion',
            planner=dict(cp,path=str(source/cp['path'])),controller=dict(controller,path=str(source/controller['path'])),
            controller_training_config=dict(path=str(config),sha256=original['controller_training_config']['sha256']),
            inference=deepcopy(original['reference_inference']),
            deployment_timing=dict(max_plan_age_ticks=10,scope='F50/K20 asynchronous reply budget: 200 ms.',
                guided=dict(max_plan_age_ticks=10,scope='F50/K20 guided planner reply budget: 200 ms; measure locally.')),
            evidence=dict(source_git=models['runner_git'],source_model=source_name,
                manifest_sha256=bundle.sha256(source/'MODELS.json'),hardware_qualified=False))
    if output.exists():
        _,existing=bundle.load_bundle(output)
        for name,row in rows.items():
            for key in ('planner','controller','controller_training_config'):
                if existing['models'][name][key]['sha256']!=row[key]['sha256']:
                    raise ValueError(f'Existing imported model differs: {name}/{key}')
                bundle.asset_path(output,existing['models'][name][key])
        return existing
    with tempfile.TemporaryDirectory(prefix='gallery_import_') as tmp:
        path=Path(tmp)/'catalog.json';write(path,dict(schema=bundle.CATALOG_SCHEMA,models=rows))
        result=bundle.prepare(path,output,list(rows),asset_mode='symlink')
    for name,row in result['models'].items():
        if row['inspected_contract']['F']!=50:raise ValueError(f'Expected F50: {name}')
        if row['pd_binding']['sha256']!=original['pd_binding']['sha256']:
            raise ValueError('Exported gallery PD differs from existing exact P42 binding')
    return result


def import_cases(cache, catalog):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from motion_diffusion.behavioral.mujoco_terrain import seed_for_tile
    manifest=read(cache/'campaign/replay_package/MANIFEST.json')
    models=read(cache/'bundle/MODELS.json')
    terrain=cache/'campaign/data/terrain_sim.npz';terrain_hash=bundle.sha256(terrain)
    records={}
    for rid,r in manifest['runs'].items():
        task=manifest['tasks'][r['task_id']];d=cache/'campaign/tasks'/r['task_id']
        for key in ('plan','options'):
            if bundle.sha256(d/(key+'.json'))!=task[key+'_sha256']:raise ValueError(f'Changed {rid}/{key}')
        plan=read(d/'plan.json');options=read(d/'options.json');targets=read(d/'targets.json');i=r['case_index']
        if bundle.sha256(d/'targets.json')!=options['steering_targets']['sha256']:raise ValueError('Targets changed')
        if terrain_hash!=targets['terrain_file_sha256']:raise ValueError('Terrain identity differs')
        if targets['recipient_controller_checkpoint_sha256']!=models['shared']['tracker']['sha256']:
            raise ValueError('Target tracker identity differs')
        if plan['initial_motion_ids'][i]!=r['motion_id'] or targets['targets'][i]['motion_id']!=r['motion_id']:
            raise ValueError('Selected lane does not match motion ID')
        record=cache/'campaign/good_runs'/rid;case=read(record/'CASE.json');metadata=read(record/'ROBOT_METADATA.json')
        root=catalog.parent/rid
        provenance=dict(run_id=rid,source_commit=manifest['source_commit'],source_model=r['model'],
            motion_id=r['motion_id'],case_index=i,source_batch_size=plan['num_envs'],
            plan_sha256=task['plan_sha256'],options_sha256=task['options_sha256'],
            targets_sha256=options['steering_targets']['sha256'],terrain_sha256=terrain_hash,
            schedule_changes='None. Exact lane, costs, physical timing and terminal behavior.',
            scope='Single-robot MuJoCo/hardware adaptation; original Isaac batch/DR is not reproduced.')
        write(root/'goal.json',extract_goal(options,targets,i,provenance=provenance))
        ep=plan['initial_episodes'][i];q=ep['replay_reset_q'];state=ep['replay_reset_root']
        tile=[case['terrain']['terrain_level'],case['terrain']['terrain_type']]
        with np.load(terrain,allow_pickle=False) as data:
            center=[*data['terrain_origins'][tuple(tile)].tolist(),float(data['terrain_origins_z'][tuple(tile)])]
            family=str(data['terrain_names'][tuple(tile)])
        if family!=case['terrain']['terrain_name']:raise ValueError('Tile family differs')
        quat=np.asarray(state[3:7]);gravity=Rotation.from_quat(quat).inv().apply([0,0,-1])
        posture='prone' if gravity[0]>0 else 'supine'
        start=dict(schema='hv2.recorded_sim_start.v1',joint_positions_rad=dict(zip(metadata['dof_names'],q)),
            joint_velocities_rad_s=dict(zip(metadata['dof_names'],ep['replay_reset_qd'])),
            root_position_m=(np.asarray(state[:3])-center).tolist(),root_quat_xyzw=quat.tolist(),
            root_linear_velocity_world_m_s=state[7:10],root_angular_velocity_world_rad_s=state[10:13],
            terrain=dict(cache=str(terrain.relative_to(ROOT)),sha256=terrain_hash,tile=tile,family=family,
                seed=seed_for_tile(tile,terrain),source_origin_xyz_m=center),source=provenance)
        write(root/'start.json',start)
        pose=dict(schema='hv2.joint_start_pose.v1',name=rid,posture=posture,
            joint_positions_rad=start['joint_positions_rad'],source=provenance,
            scope='Optional START sets joints only. Place the pelvis/body on matching physical supports yourself.')
        write(root/'robot_pose.json',pose)
        records[rid]=dict(model=STAGE3_MODELS[r['model']],source_model=r['model'],direction=r['direction'],
            motion_id=r['motion_id'],terrain=family,tile=tile,posture=posture,
            goal=f'{rid}/goal.json',start=f'{rid}/start.json',robot_pose=f'{rid}/robot_pose.json',
            seed=options['seed'],replan_hz=50/options['replan_every_ticks'],mode=r['command_type'],
            gain=options['guidance_scale'],max_ticks=task['max_ticks'])
    write(catalog,dict(schema='hv2.terrain_gallery.v1',source_commit=manifest['source_commit'],
        cache=str(cache),aliases=manifest['videos'],runs=records))
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=CACHE);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--catalog',type=Path,default=CATALOG)
    a=p.parse_args();models=import_models(a.cache.resolve(),a.output.resolve())
    cases=import_cases(a.cache.resolve(),a.catalog.resolve())
    print(f'Imported {len(models["models"])} planner pairs and {len(cases)} exact steering commands. {a.catalog}')

if __name__=='__main__':main()
