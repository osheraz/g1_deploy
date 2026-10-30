#!/usr/bin/env python3
"""Re-run recorded neural decisions and PD shaping. Never open robot/sensor I/O."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from steerable_recovery.paths import ROOT, PROJECT
sys.path.insert(0, str(ROOT))


def source_changes(sources):
    """Resolve moved source paths, retaining the original byte checks."""
    from motion_diffusion.inference.bundle import sha256
    from motion_diffusion.source_paths import SOURCE_MOVES
    mapping = {**SOURCE_MOVES, **json.loads((PROJECT/'migration_paths.json').read_text())}
    changed, relocated = [], {}
    for name, digest in sources.items():
        path = ROOT/name
        if not path.is_file() and name in mapping:
            path = ROOT/mapping[name]
            relocated[name] = mapping[name]
        if not path.is_file() or sha256(path) != digest:
            changed.append(name)
    return changed, relocated


def compare(actual, expected, *, atol, rtol):
    import numpy as np
    if hasattr(actual, 'detach'): actual = actual.detach().cpu().numpy()
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return dict(passed=False, error='shape mismatch', actual=list(actual.shape), expected=list(expected.shape))
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    return dict(passed=finite and bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
                max_abs_error=float(np.max(np.abs(actual-expected))) if finite and actual.size else None)


def replay(session, *, component='all', max_calls=0, atol=1.e-4, rtol=1.e-4, allow_code_changes=False):
    import numpy as np
    import torch
    from motion_diffusion.inference.model_trace import read_trace, engine_identity
    from motion_diffusion.inference.bundle import sha256
    from motion_diffusion.inference.execution import sample_snapshot, _ipc_tensors
    from motion_diffusion.inference.policy import load_controller
    from motion_diffusion.planner.runtime import load_frozen_planner
    from motion_diffusion.inference.engines import configure_planner
    from motion_diffusion.inference.action_limits import PDTargetLimiter
    root = Path(session).resolve()
    if root.name == 'sensors': root = root.parent
    directory = root/'model_trace'
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['schema'] not in ('hv2.model_trace.v1','hv2.model_trace.v2'): raise ValueError('Unknown model trace schema')
    summary = json.loads((directory/'summary.json').read_text())
    if not summary['complete']: raise ValueError('Trace recording is incomplete; see model_trace/summary.json')
    if sha256(directory/'source.tar.gz') != manifest['source_archive_sha256']:
        raise ValueError('Recorded source archive checksum mismatch')
    changed, relocated = source_changes(manifest['sources'])
    if changed and not allow_code_changes:
        raise ValueError(f'{len(changed)} source files differ from capture; use --allow-code-changes for an intentional comparison. '
                         f'Original source is saved in {directory}/source.tar.gz. First differences: {changed[:5]}')
    config = deepcopy(manifest['config'])
    for key, spec in manifest['weights'].items():
        if sha256(spec['path']) != spec['sha256']: raise ValueError(f'Changed model weights: {spec["path"]}')
        config[key] = spec['path']
    # Validate ALL bytes even when inference is restricted to a prefix/component.
    totals = {name: sum(1 for _ in read_trace(directory/name)) for name in ('control', 'planner')}
    depth_archive=None
    if manifest.get('depth_storage')=='metric_capture_refs_v1':
        from steerable_recovery.hardware.depth_recording import RecordedModelDepth
        depth_archive=RecordedModelDepth(root/'sensors/depth')
    def restore_depth(view,view_config):
        if view is not None and view.get('storage')=='metric_capture_refs_v1':
            if depth_archive is None:raise ValueError('Missing metric depth archive for model references')
            return depth_archive.restore(view,view_config)
        return view
    torch.set_num_threads(1)
    report = dict(schema='hv2.model_replay.v1', session=str(root), component=component,
        atol=atol, rtol=rtol, max_calls=max_calls, source_changes=changed,
        relocated_sources=relocated, records=totals,
        scope='Recorded-input neural replay and issued-target checks; no physical dynamics or translation accuracy claim',
        tracker=[], planner=[])

    def verify_engines(name, module):
        saved = json.loads((directory/name/'runtime.json').read_text())
        if engine_identity(module) != saved['engines']:
            raise ValueError(f'{name}: loaded engines differ from capture')

    if component in ('all', 'tracker'):
        controller = load_controller(config)
        from motion_diffusion.inference.standing import load_standing_reference
        standing=load_standing_reference(config,controller)
        verify_engines('control', controller)
        limiter = None
        if config.get('command_limits', {}).get('mode') == 'bounded':
            limiter = PDTargetLimiter(**config['command_limits']['parameters'])
        generation = None
        for row in read_trace(directory/'control'):
            if row['kind']=='standing_hold_start':
                if standing is None or row['controller_sha256']!=controller.checkpoint_sha256:
                    raise ValueError('Standing encoder activation lacks the bound checkpoint/config')
                inputs={k:torch.from_numpy(v).to(controller.device) if isinstance(v,np.ndarray) else v
                        for k,v in row['reference'].items()}
                standing.reset();standing.activate(**inputs)
                continue
            if row['kind'] != 'tracker' or row['preview']: continue
            if max_calls and len(report['tracker']) >= max_calls: continue
            checks = {}
            item = dict(tick=row['tick'], control_tick=row['control_tick'], generation=row['generation'],
                        outcome=row['outcome'], checks=checks)
            report['tracker'].append(item)
            if row.get('proposed') is None or row.get('raw_action') is None:
                item.update(passed=False, error='Recorded tracker call did not complete')
                continue
            try:
                execution=row.get('execution') or {}
                if execution.get('mode')=='standing_hold':
                    if standing is None or not standing.active:
                        raise ValueError('Missing recorded standing-reference activation')
                    encoded=standing(row['monotonic_s'],torch.zeros(1,3,device=controller.device),
                        torch.tensor([execution['heading_rad']],device=controller.device))
                    checks['standing_native_code']=compare(encoded,row['native_code'],atol=atol,rtol=rtol)
                depth = _ipc_tensors(restore_depth(row['depth'],controller.depth_config))
                kwargs = {} if depth is None else dict(depth_frames=depth['frames'],
                    depth_metadata=depth['metadata'], depth_available=depth['available'])
                action = controller(torch.from_numpy(row['history']), torch.from_numpy(row['native_code']), **kwargs)
                checks['raw_action'] = compare(controller.last_unclipped_action, row['raw_action'], atol=atol, rtol=rtol)
                checks['proposed'] = compare(action, row['proposed'], atol=atol, rtol=rtol)
                item['recorded_raw_abs_max'] = (float(np.max(np.abs(row['raw_action'])))
                                                if np.isfinite(row['raw_action']).all() else None)
                if row['outcome'] == 'issued':
                    proposed = row['proposed'][0]  # Independent check of shaping, not compounded model rounding.
                    if limiter is not None:
                        if generation != row['generation']: limiter.reset()
                        generation = row['generation']
                        issued, target, _ = limiter.apply(proposed, row['observation']['q'],
                            elapsed_s=(row.get('command_shaping') or {}).get('elapsed_s'))
                    else:
                        issued = proposed
                        target = controller.default_joint_pos.cpu().numpy()+np.asarray(config['pd']['training_pd']['action_scale'])*issued
                    checks['issued_request'] = compare(issued, row['issued_request'], atol=1.e-6, rtol=1.e-6)
                    checks['issued_target'] = compare(target, row['issued_target'], atol=1.e-6, rtol=1.e-6)
                item['passed'] = all(v['passed'] for v in checks.values())
            except Exception as exc:
                item.update(passed=False, error=f'{type(exc).__name__}: {exc}')
            if len(report['tracker']) % 250 == 0: print(f'Replayed {len(report["tracker"])} tracker calls', flush=True)
        del controller

    if component in ('all', 'planner'):
        options = dict(config['planner_options'])
        from motion_diffusion.inference.steering import load_hardware_planner
        runtime = load_hardware_planner(config['planner_checkpoint'], device=config.get('planner_device', 'cuda'), **options)
        configure_planner(runtime, config['planner_checkpoint'], config.get('planner_engine'))
        verify_engines('planner', runtime.model)
        for row in read_trace(directory/'planner'):
            if row['request'].get('preview'): continue
            if max_calls and len(report['planner']) >= max_calls: continue
            checks = {}
            item = dict(origin_tick=row['request']['origin_tick'], generation=row['request']['generation'], checks=checks)
            report['planner'].append(item)
            if row.get('error'):
                item.update(passed=False, error=row['error'])
                continue
            try:
                if str(runtime.generator.device) != row['generator_device']:
                    raise ValueError('Generator device differs from capture')
                runtime.generator.set_state(torch.from_numpy(row['rng_state']))
                noise = {k: torch.from_numpy(v).to(runtime.model.device) for k, v in row['noise'].items()}
                request=dict(row['request'])
                request['depth']=restore_depth(request.get('depth'),runtime.model.behavioral_config.get('depth_condition',{}))
                output = sample_snapshot(runtime, _ipc_tensors(request), noise)['prediction']
                for key, expected in row['result']['prediction'].items():
                    checks[key] = compare(output[key], expected, atol=atol, rtol=rtol)
                item['passed'] = all(v['passed'] for v in checks.values())
            except Exception as exc:
                item.update(passed=False, error=f'{type(exc).__name__}: {exc}')
            if len(report['planner']) % 25 == 0: print(f'Replayed {len(report["planner"])} planner calls', flush=True)
    selected = ('tracker', 'planner') if component == 'all' else (component,)
    report['passed'] = all(report[name] and all(row['passed'] for row in report[name]) for name in selected)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--component', choices=['all', 'tracker', 'planner'], default='all')
    parser.add_argument('--max-calls', type=int, default=0, help='Limit neural calls per component; 0 checks all')
    parser.add_argument('--atol', type=float, default=1.e-4)
    parser.add_argument('--rtol', type=float, default=1.e-4)
    parser.add_argument('--allow-code-changes', action='store_true')
    args = parser.parse_args()
    if args.max_calls < 0 or args.atol < 0 or args.rtol < 0: parser.error('Limits/tolerances must be nonnegative')
    root = args.session.resolve()
    if root.name == 'sensors': root = root.parent
    try:
        report = replay(root, component=args.component, max_calls=args.max_calls,
                        atol=args.atol, rtol=args.rtol, allow_code_changes=args.allow_code_changes)
    except Exception as exc:
        report = dict(passed=False, error=f'{type(exc).__name__}: {exc}')
    (root/'review').mkdir(exist_ok=True)
    output = root/'review/model_replay.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(('PASS' if report['passed'] else 'FAIL')+f': model replay; {output}')
    for name in ('tracker', 'planner'):
        rows = report.get(name, [])
        if rows: print(f'{name}: {sum(r["passed"] for r in rows)}/{len(rows)} calls matched')
    if report.get('error'): print(report['error'])
    if not report['passed']: raise SystemExit(1)


if __name__ == '__main__':
    main()
