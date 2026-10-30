"""Verify retained lab deployment inputs without sensors or motor output."""
import argparse
import json
from pathlib import Path

from steerable_recovery.paths import ROOT, CONFIG_ROOT, ASSET_ROOT, recovery_asset
from motion_diffusion.inference import bundle
from motion_diffusion.inference.steering import motion_goal

CATALOG = CONFIG_ROOT/'steering/lab_gallery/catalog.json'
PLANNER_SHA = '02c2ef96ac30a60c5d8dfea5d20b3cc08654fb10ebd6b9f198e515dc18342fcd'
TRACKER_SHA = '860784ab07a1a521e2532e326151260f5cf44451482ff91e293af526bd144703'
MANIFEST_SHA = '8234dd8910476b55eefd9b587aa5b5e6e9b9c5d1e7dd0336dfc87950807ca515'
LAUNCHER_COMMIT = '255e200f'


def read(p):
    return json.loads(Path(p).read_text())


def verify_local_assets(catalog=CATALOG):
    """Verify the deployment dependency files without reading any Turbo paths.

    Source paths inside provenance are descriptive, never runtime dependencies.
    Local symlinks must resolve within the checkout or declared asset root.
    This checks files, not live sensors, estimator accuracy or robot performance.
    """
    from core.config import _load_config
    from steerable_recovery.settings import model_settings
    checked={}

    def check(path, expected=None):
        path=Path(path);resolved=path.resolve()
        if not any(resolved.is_relative_to(base) for base in (ROOT, ASSET_ROOT)):
            raise ValueError(f'Deployment asset resolves outside the declared roots: {path} -> {resolved}')
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
    cache=recovery_asset(data['cache']);receipt=read(check(cache/'IMPORT.json'))
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
        check(recovery_asset(start['terrain']['geometry']),start['terrain']['sha256'])
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
    check(CONFIG_ROOT/'lidar/mount_candidate.json')
    result=dict(schema='hv2.lab_gallery_local_assets.v1',run_count=len(data['runs']),
        launcher_commit=data['launcher_commit'],manifest_sha256=data['manifest_sha256'],
        all_files_local=True,files=checked,
        scope='Deployment weights, exports, commands, starts, calibration and scene; no robot/sensor checks or Isaac batch reproduction.')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=CATALOG)
    parser.add_argument('--output', type=Path, help='Optional new verification report; input files are read only')
    args = parser.parse_args(argv)
    result = verify_local_assets(args.catalog)
    if args.output is not None:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
