"""Import Stanley's selected lab runs without importing his simulation runtime."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from steerable_recovery.paths import ROOT, CONFIG_ROOT
from motion_diffusion.behavioral.hardware import bundle
from motion_diffusion.behavioral.hardware.steering import motion_goal

SOURCE = Path('/nfs/turbo/coe-stellayu/clhsieh/software/hv2_diffusion/custom_lab_steering_20260922/robust_set_01')
CATALOG = CONFIG_ROOT/'steering/lab_gallery/catalog.json'
CACHE = ROOT/'logs/native_recovery/custom_lab_robust_20260922'
BUNDLE = ROOT/'logs/native_recovery/bundles/stage3_40k_20260921'
PLANNER_SHA = '02c2ef96ac30a60c5d8dfea5d20b3cc08654fb10ebd6b9f198e515dc18342fcd'
TRACKER_SHA = '860784ab07a1a521e2532e326151260f5cf44451482ff91e293af526bd144703'
MANIFEST_SHA = '8234dd8910476b55eefd9b587aa5b5e6e9b9c5d1e7dd0336dfc87950807ca515'
LAUNCHER_COMMIT = '255e200f'


def read(p):
    return json.loads(Path(p).read_text())


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def verify_local_assets(catalog=CATALOG):
    """Verify the deployment dependency files without reading any Turbo paths.

    Source paths inside provenance are descriptive, never runtime dependencies.
    Local symlinks are allowed; symlinks resolving outside this checkout fail.
    This checks files, not live sensors, estimator accuracy or robot performance.
    """
    from core.config import _load_config
    from steerable_recovery.settings import model_settings
    checked={}

    def check(path, expected=None):
        path=Path(path);resolved=path.resolve()
        if not resolved.is_relative_to(ROOT):
            raise ValueError(f'Deployment asset resolves outside the local checkout: {path} -> {resolved}')
        if not resolved.is_file() or resolved.stat().st_size==0:
            raise ValueError(f'Missing or empty local deployment asset: {path}')
        key=str(path)
        if key not in checked:
            checked[key]=dict(resolved=str(resolved),bytes=resolved.stat().st_size,
                sha256=bundle.sha256(resolved))
        if expected is not None and checked[key]['sha256']!=expected:
            raise ValueError(f'Local asset checksum mismatch: {path}')
        return path

    catalog=check(Path(catalog).resolve());data=read(catalog)
    cache=Path(data['cache']);receipt=read(check(cache/'IMPORT.json'))
    if data['manifest_sha256']!=receipt['manifest_sha256']:
        raise ValueError('Catalog and import receipt refer to different source manifests')
    for name,digest in receipt['files'].items():check(cache/name,digest)
    check(cache/'replay_package/MANIFEST.json',data['manifest_sha256'])
    previews=read(check(cache/'starts/SOURCES.json'))
    for name,row in previews.items():check(cache/name,row['sha256'])
    for ids in data['aliases'].values():
        if any(rid not in data['runs'] for rid in ids):raise ValueError('Dangling gallery alias')
    for row in data['runs'].values():
        if row['model']!='s50r_diffusion':raise ValueError('Unexpected gallery model')
        for key in ('goal','start','robot_pose'):check(catalog.parent/row[key])
        goal=read(catalog.parent/row['goal']);motion_goal(goal)
        start=read(catalog.parent/row['start'])
        check(ROOT/start['terrain']['geometry'],start['terrain']['sha256'])
    # Preserve the scene's relative USD reference for local inspection.
    for name in ('lab_scene.usda','lab_terrain.usda','lab_terrain.yaml','scene_overview.png'):
        check(cache/'scene'/name)
    settings=model_settings(_load_config(check(CONFIG_ROOT/'models.yaml')),'s50r_diffusion')
    model_root=Path(settings['bundle']);model=read(check(model_root/'bundle.json'))['models']['s50r_diffusion']
    if model['planner']['sha256']!=PLANNER_SHA or model['controller']['sha256']!=TRACKER_SHA:
        raise ValueError('Local S50R/P42 weights differ from the imported commands')
    for key in ('planner','controller','controller_training_config','pd_binding'):
        asset=model[key];check(model_root/asset['path'],asset['sha256'])
    export_root=ROOT/settings['exports']['controller']
    export=read(check(export_root/'export.json'))
    if export['checkpoint_sha256']!=TRACKER_SHA:
        raise ValueError('TensorRT exports belong to a different tracker')
    for graph in export['graphs'].values():
        check(export_root/graph['file'],graph['sha256'])
        check(export_root/graph['validation']['file'],graph['validation']['sha256'])
        engine=check(graph['tensorrt_engine'])
        identity=read(check(engine.with_suffix('.json')))
        if identity['onnx_sha256']!=graph['sha256']:
            raise ValueError('Cached TensorRT engine references a different ONNX graph')
    check(CONFIG_ROOT/'lidar/mount_candidate_20260912.json')
    result=dict(schema='hv2.lab_gallery_local_assets.v1',run_count=len(data['runs']),
        launcher_commit=data['launcher_commit'],manifest_sha256=data['manifest_sha256'],
        all_files_local=True,files=checked,
        scope='Deployment weights, exports, commands, starts, calibration and scene; no robot/sensor checks or Isaac batch reproduction.')
    write(cache/'LOCAL_ASSETS.json',result)
    return result


def import_cases(source=SOURCE, catalog=CATALOG, cache=CACHE):
    import torch
    from scipy.spatial.transform import Rotation
    source, catalog, cache = (Path(p).resolve() for p in (source, catalog, cache))
    manifest_path=source/'replay_package/MANIFEST.json'
    if bundle.sha256(manifest_path)!=MANIFEST_SHA:
        raise ValueError('Gallery manifest changed; review the new recipe before importing it')
    manifest=read(manifest_path)
    sources={name:Path(path).resolve() for name,path in manifest['handoffs'].items()}
    extra_parent=Path('/nfs/turbo/coe-stellayu-extra/clhsieh/software/hv2_diffusion/custom_lab_steering_20260922')
    if sources.get(source.name)!=source or any(
            p.parent not in (source.parent,extra_parent) or p.name!=name for name,p in sources.items()):
        raise ValueError('Expected named handoff directories on the declared Turbo volumes')
    hash_sets={}
    for name,path in sources.items():
        if (path/'SHA256.json').exists():
            hash_sets[name]=read(path/'SHA256.json')
        else:
            # New singleton shipments use standard sha256sum inventories.
            hash_sets[name]={}
            for line in (path/'SHA256SUMS').read_text().splitlines():
                digest,relative=line.split(maxsplit=1);relative=relative.lstrip('*')
                if len(digest)!=64 or Path(relative).is_absolute() or '..' in Path(relative).parts:
                    raise ValueError('Invalid source hash inventory')
                hash_sets[name][relative]=digest
    copied={}

    def source_file(path):
        path=Path(path).resolve()
        for name,root in sources.items():
            if path.is_relative_to(root):
                relative=path.relative_to(root)
                cached=relative if root==source else Path(name)/relative
                return cached,hash_sets[name].get(str(relative))
        raise ValueError(f'Artifact is outside the declared handoffs: {path}')

    def copy_verified(path, expected=None):
        path=Path(path).resolve()
        try:rel,listed_hash=source_file(path)
        except ValueError:
            # The new plan references a shared seven-second start table outside
            # its shipment. Accept only this hash-bound input, copy it locally.
            if expected is None or path!=source.parent/'lab_terrain_7sec_01/source/starts48.json':raise
            rel=Path('external_inputs')/path.relative_to(source.parent);listed_hash=None
        expected=expected or listed_hash
        if not expected or bundle.sha256(path)!=expected:
            raise ValueError(f'Missing or mismatched source hash: {rel}')
        dest=cache/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or bundle.sha256(dest)!=expected:
            dest.write_bytes(path.read_bytes())
        copied[str(rel)]=expected
        return dest

    copy_verified(manifest_path,MANIFEST_SHA)
    # Human-readable guide can be newer than the producer inventory. Pin this
    # reviewed copy separately; numerical inputs still require source hashes.
    copy_verified(source/'replay_package/README.md',bundle.sha256(source/'replay_package/README.md'))
    if manifest['schema']!='hv2.custom_lab_gallery_replay.v1':
        raise ValueError('Expected custom lab replay package')
    local_root, models=bundle.load_bundle(BUNDLE)
    model=models['models']['s50r_diffusion']
    for key, digest in [('planner',PLANNER_SHA),('controller',TRACKER_SHA)]:
        if model[key]['sha256']!=digest:
            raise ValueError(f'Wrong S50R/P42 {key}')
        bundle.asset_path(local_root,model[key])
    cp=torch.load(bundle.asset_path(local_root,model['controller']),map_location='cpu',weights_only=True,mmap=True)
    names=cp['action_vae']['model_contract']['dof_names']
    del cp
    geometry=copy_verified(source/'scene/geometry.json')
    # The USD scene uses a relative reference to lab_terrain.usda.
    for name in ('lab_scene.usda','lab_terrain.usda','lab_terrain.yaml','scene_overview.png'):
        copy_verified(source/'scene'/name)
    for name in ('README.md','REVIEW.json'):
        copy_verified(source/name)
    for other in sources.values():
        if other==source:continue
        other_geometry=copy_verified(other/'scene/geometry.json')
        if bundle.sha256(other_geometry)!=bundle.sha256(geometry):
            raise ValueError('Augmented handoff uses a different terrain; import its scene explicitly')
        for name in ('README.md','REVIEW.json'):
            copy_verified(other/name)
    records={}
    for rid, row in manifest['runs'].items():
        task=manifest['tasks'][row['task_id']]
        run_source=sources[row.get('source_set',source.name)]
        # Recipe 04 reuses recipe 03's left batch at a new case index.
        # A clip shipment and its hash-bound task can belong to different sets.
        task_source=sources[task.get('source_set',source.name)]
        plan=read(copy_verified(source/'replay_package'/task['plan'],task['plan_sha256']))
        options=read(copy_verified(source/'replay_package'/task['options'],task['options_sha256']))
        binding=options['steering_targets']
        if not Path(binding['path']).resolve().is_relative_to(task_source):
            raise ValueError(f'Targets are outside the declared task handoff: {rid}')
        targets=read(copy_verified(binding['path'],binding['sha256']))
        starts=read(copy_verified(plan['custom_lab']['starts'],plan['custom_lab']['starts_sha256']))['starts']
        i=row['case_index'];start=starts[i];target=targets['targets'][i]
        exported_path=run_source/'targets'/f'{rid}.json'
        if not exported_path.exists():
            # Singles 31/32 archive their explicit direction under right.json.
            exported_path=run_source/'targets'/f"{row['direction']}.json"
        exported=read(copy_verified(exported_path))
        spatial=row.get('command_type')=='progress_xyz'
        if (task['model']!='S50R' or plan['num_envs']!=48 or
            plan['initial_motion_ids'][i]!=row['motion_id'] or
            target['motion_id']!=row['motion_id'] or start['motion_id']!=row['motion_id'] or
            start['case_index']!=i or targets['recipient_controller_checkpoint_sha256']!=TRACKER_SHA or
            target['recipient_controller_checkpoint_sha256']!=TRACKER_SHA or
            target.get('steering_active',True) is not True or
            binding['mode']!=('waypoint_xyz' if spatial else 'waypoint_pose') or
            binding['apply_guidance'] is not True):
            raise ValueError(f'Run, target, start or tracker identity mismatch: {rid}')
        if spatial:
            if (task['source_commit']!='bfa03efad91edd529271b9834a074935f9dc1968'
                    or binding.get('active_until_s') is not None or row['guidance_active_until_s'] is not None
                    or 'active_until_s' in target['pose_waypoints']
                    or target['turn_release']['direction']!=row['direction']
                    or target['turn_release']['angle_deg']!=row['release_threshold_deg']
                    or target['turn_release']!=row['release_rule']
                    or target['path_progress']!=row['progress_method']
                    or target['turn_release']!=exported['turn_release']
                    or target['path_progress']!=exported.get('selection',exported.get('path_progress'))
                    or exported['guidance_cutoff_s'] is not None
                    or exported['reference_times_s']!=target['pose_waypoints']['times_s']
                    or not exported['reference_times_are_not_deadlines']):
                raise ValueError(f'Progress recipe mismatch: {rid}')
            if 'positions_initial_frame_m' in exported:
                if (exported['positions_initial_frame_m']!=target['pose_waypoints']['positions_m']
                        or exported['guidance_scale']!=options['guidance_scale']
                        or exported['guidance_max_grad_norm']!=options['guidance_max_grad_norm']
                        or exported['position_error_scale_m']!=binding['position_error_scale_m']
                        or exported['orientation_cost'] is not False):
                    raise ValueError(f'Single-run target geometry or gains differ: {rid}')
        elif (binding['active_until_s'] not in (3.,7.) or
                row['guidance_active_until_s']!=binding['active_until_s'] or
                target['pose_waypoints']['active_until_s']!=binding['active_until_s'] or
                target['pose_waypoints']!=exported['relative_pose_waypoints']):
            raise ValueError(f'Timed recipe mismatch: {rid}')
        for key in ('replan_every_ticks','blend_ticks','seed'):
            if key in row and row[key]!=options[key]:raise ValueError(f'Wrong {key}: {rid}')
        # Some augmented manifest rows omit cadence. The hash-bound options
        # always define it; never borrow a previous run's K/blend defaults.
        bundle.timing_options(model,10,options['replan_every_ticks'],options['blend_ticks'])
        for key in ('mode','sampler','num_inference_steps','use_ema','numerical_precision','noise_policy'):
            if options[key]!=model['reference_inference'][key]:
                raise ValueError(f'Inference recipe differs: {key}')
        # Verify the exported simulator joint order rather than guessing URDF order.
        clip=Path(row['source_clip'])
        if bundle.sha256(clip)!=source_file(clip)[1]:
            raise ValueError(f'Clip hash mismatch: {rid}')
        recorded=torch.load(clip,map_location='cpu',weights_only=False)
        if recorded['robot_metadata']['dof_names']!=names:
            raise ValueError(f'Joint order mismatch: {rid}')
        del recorded
        state=start['root_state']
        if len(start['q'])!=len(names) or len(state)!=13:
            raise ValueError(f'Invalid start dimensions: {rid}')
        posture='supine' if Rotation.from_quat(state[3:7]).apply([1.,0.,0.])[2]>0 else 'prone'
        if posture!='supine':raise ValueError(f'Expected selected supine start: {rid}')
        provenance=dict(run_id=rid,source_commit=task.get('source_commit',manifest['source_commit']),source_model='S50R',
            planner_sha256=PLANNER_SHA,tracker_sha256=TRACKER_SHA,motion_id=row['motion_id'],
            case_index=i,source_batch_size=48,plan_sha256=task['plan_sha256'],
            options_sha256=task['options_sha256'],targets_sha256=binding['sha256'],
            starts_sha256=plan['custom_lab']['starts_sha256'],terrain_sha256=bundle.sha256(geometry),
            behavior_axis=row['behavior_axis'],start_id=start['start_id'],edge=start['edge'],
            scope='Single-robot adaptation with measured feedback and asynchronous plans; not exact Isaac batch/DR replay.')
        if run_source!=source:provenance['source_set']=run_source.name
        if task_source!=run_source:provenance['task_source_set']=task_source.name
        if spatial:
            runtime=manifest['runtimes'][task['source_commit']]
            archive=copy_verified(source/'replay_package'/runtime['source_archive'],runtime['source_sha256'])
            # Keep the exact controller source for offline parity/review, not an
            # external runtime dependency. Extract only the three named files.
            import tarfile
            with tarfile.open(archive) as package:
                for name in ('path_progress.py','turn_completion.py','pose_waypoints.py'):
                    dest=cache/'progress_source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
                    dest.write_bytes(package.extractfile('repo/motion_diffusion/planner/'+name).read())
                    copied[str(dest.relative_to(cache))]=bundle.sha256(dest)
        # Pose cost consumes only the sparse pose payload. Drop redundant dense
        # samples and teacher replay arrays; keep exact sparse knots and tolerances.
        goal=dict(schema='hv2.hardware_motion_goal.v1',clock='physical_elapsed_s',
            frame='initial_pelvis_origin_causal_heading',mode=binding['mode'],
            guidance_scale=options['guidance_scale'],guidance_max_grad_norm=options['guidance_max_grad_norm'],
            position_error_scale_m=binding['position_error_scale_m'],yaw_error_scale_deg=binding.get('yaw_error_scale_deg',12.),
            target=dict(pose_waypoints=deepcopy(target['pose_waypoints'])),
            execution=dict(release_after_s=binding.get('active_until_s'),ramp_s=0.,release_on_upright=False),
            provenance=provenance,hardware_qualified=False)
        if spatial:
            goal['target'].update({key:deepcopy(target[key]) for key in ('path_progress','turn_release')})
            goal['execution']=dict(release_on_upright=False)
        motion_goal(goal)
        out=catalog.parent/rid
        write(out/'goal.json',goal)
        snapshot=dict(schema='hv2.recorded_sim_start.v1',
            joint_positions_rad=dict(zip(names,start['q'])),joint_velocities_rad_s=dict.fromkeys(names,0.),
            root_position_m=state[:3],root_quat_xyzw=state[3:7],
            root_linear_velocity_world_m_s=state[7:10],root_angular_velocity_world_rad_s=state[10:13],
            terrain=dict(geometry=str(geometry.relative_to(ROOT)),sha256=bundle.sha256(geometry)),source=provenance)
        write(out/'start.json',snapshot)
        write(out/'robot_pose.json',dict(schema='hv2.joint_start_pose.v1',name=rid,posture=posture,
            joint_positions_rad=snapshot['joint_positions_rad'],source=provenance,
            scope='Optional START sets joints only; operator places the body on matching supports.'))
        records[rid]=dict(model='s50r_diffusion',source_model='S50R',direction=row['direction'],
            motion_id=row['motion_id'],terrain='lab_two_level_platform',posture=posture,
            start_id=start['start_id'],edge=start['edge'],placement=start['placement'],
            behavior_axis=row['behavior_axis'],goal=f'{rid}/goal.json',start=f'{rid}/start.json',
            robot_pose=f'{rid}/robot_pose.json',seed=options['seed'],replan_hz=50/options['replan_every_ticks'],
            blend_ticks=options['blend_ticks'],standing_hold=True,mode=binding['mode'],
            gain=options['guidance_scale'],max_ticks=task['max_ticks'])
        if run_source!=source:records[rid]['source_set']=run_source.name
        if spatial:
            records[rid].update(steering_method='progress_xyz',release_threshold_deg=row['release_threshold_deg'],
                fade_start_deg=target['turn_release'].get('fade_start_deg'))
        elif binding['active_until_s']!=3.:
            records[rid]['release_after_s']=binding['active_until_s']
    # Gallery thumbnails may be generated after SHA256.json. Record their source
    # bytes separately; numerical contracts above require producer-supplied hashes.
    images={}
    for alias,ids in manifest['videos'].items():
        if len(ids)!=1:continue
        image=source/'video_gallery/03_steering'/alias/'preview_0.jpg'
        payload=image.read_bytes()
        dest=cache/'starts'/f'{alias}.jpg';dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(payload)
        images[str(dest.relative_to(cache))]=dict(source=str(image),sha256=bundle.sha256(dest))
    write(cache/'starts/SOURCES.json',images)
    write(cache/'IMPORT.json',dict(source=str(source),handoffs=manifest['handoffs'],
        manifest_sha256=MANIFEST_SHA,launcher_commit=LAUNCHER_COMMIT,files=copied,model='s50r_diffusion',
        planner_sha256=PLANNER_SHA,tracker_sha256=TRACKER_SHA))
    write(catalog,dict(schema='hv2.terrain_gallery.v1',source_commit=manifest['source_commit'],
        cache=str(cache),launcher_commit=LAUNCHER_COMMIT,manifest_sha256=MANIFEST_SHA,
        aliases=manifest['videos'],runs=records))
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--catalog',type=Path,default=CATALOG)
    p.add_argument('--cache',type=Path,default=CACHE)
    p.add_argument('--verify-local',action='store_true',help='Check installed deployment files only; no Turbo, sensors or motors')
    a=p.parse_args()
    if not a.verify_local:
        print(f'Imported {len(import_cases(a.source,a.catalog,a.cache))} S50R lab commands: {a.catalog}')
    result=verify_local_assets(a.catalog)
    print(f"Verified {result['run_count']} runs and {len(result['files'])} local assets; model and preset files require no Turbo access.")


if __name__=='__main__':main()
