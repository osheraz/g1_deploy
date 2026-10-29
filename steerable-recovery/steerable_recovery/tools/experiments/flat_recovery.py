#!/usr/bin/env python3
"""Compare Stanley's resting supine starts with the earlier flat examples.

prepare: copy the new S50R pair and settle the two source poses (no policy).
run: fixed-start unguided comparisons, all outcomes retained, no viewer.
view: live MuJoCo with depth; N cycles seeds, R repeats, no video by default.
report: score saved trajectories and draw their initial-heading pelvis paths.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from steerable_recovery.paths import ROOT, CONFIG_ROOT, PACKAGE_ROOT
sys.path.insert(0, str(ROOT))
BASE = Path('/nfs/turbo/coe-stellayu/clhsieh/software/hv2_diffusion/dependencies')
SOURCE = BASE / 'deployment_candidates'
EVIDENCE = BASE / 'deployment_evidence'
CATALOG = CONFIG_ROOT / 'multimodal/stanley_flat/catalog.json'
RESULTS = ROOT / 'logs/multimodal_recovery/stanley_flat_20260915'
BUNDLE = ROOT / 'logs/native_recovery/bundles/r8_s50r_20260915a'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    from motion_diffusion.behavioral.hardware.bundle import sha256
    return sha256(path)


def prepare():
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from scipy.spatial.transform import Rotation
    from humanoidverse2.agents.action_vae.heading import initialize_heading
    from motion_diffusion.behavioral.hardware.bundle import asset_path
    from motion_diffusion.behavioral.mujoco import flat_model, MujocoNativePlant
    from motion_diffusion.behavioral.start_state import snapshot_start

    manifest = read(SOURCE / 'bundle.json')
    row = deepcopy(manifest['models']['R8_S50R'])
    expected = 'd780bba4c5514304028be4f3c2eded891d55f006dca4839a4c413e2de06d58a1'
    if row['planner']['sha256'] != expected:
        raise ValueError('Source S50R checkpoint changed; review the new handoff')
    BUNDLE.mkdir(parents=True, exist_ok=True)
    for key in ('planner', 'controller', 'controller_training_config', 'pd_binding'):
        source = asset_path(SOURCE, row[key])
        dest = BUNDLE / row[key]['path']
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            shutil.copyfile(source, dest)
        asset_path(BUNDLE, row[key])
    manifest['models'] = {'r8_s50r_diffusion': row}
    manifest.update(asset_mode='copy', source_bundle=str(SOURCE),
                    source_manifest_sha256=digest(SOURCE / 'bundle.json'))
    write(BUNDLE / 'bundle.json', manifest)
    # The tracker defines named joint ordering and the exact plant/PD signature.
    ckpt = torch.load(asset_path(BUNDLE, row['controller']), weights_only=True,
                      map_location='cpu', mmap=True)
    contract = ckpt['action_vae']['model_contract']
    robot = OmegaConf.to_container(OmegaConf.load(asset_path(BUNDLE, row['controller_training_config'])).robot, resolve=True)
    pd = read(asset_path(BUNDLE, row['pd_binding']))['deployment_pd']
    plant = MujocoNativePlant(flat_model(), contract, pd, robot, command_limits='bounded')
    source_file = EVIDENCE / 'initial_flat_R8_S50R.json'
    source = read(source_file)
    starts = {}
    for mid in (151, 1444):
        state = source['states'][str(mid)]
        quat = np.asarray(state['root_state'][3:7])
        heading = float(initialize_heading(torch.tensor(quat, dtype=torch.float64)))
        quat = (Rotation.from_rotvec([0., 0., -heading]) * Rotation.from_quat(quat)).as_quat()
        plant.reset('supine', initial_joints=state['q'], initial_quat_xyzw=quat)
        info = plant.reset_info
        if (not info['settle_converged'] or info['passive_root_drift_100ms_m'] > .02
                or info['passive_joint_drift_100ms_rad'] > .15):
            raise ValueError(f'Source {mid} failed resting-start preparation: {info}')
        if plant.observation().gravity_body[0] > -.7:
            raise ValueError(f'Source {mid} no longer lies face up after settling')
        provenance = dict(source_file=str(source_file), source_sha256=digest(source_file),
            source_model='R8_S50R', motion_id=mid, joint_names=plant.names,
            source_linear_speed_m_s=float(np.linalg.norm(state['root_state'][7:10])),
            source_joint_speed_max_rad_s=float(np.max(np.abs(state['qdot']))),
            adaptation='Named source joints and pelvis attitude; remove world heading/translation, place on flat floor, passively settle, zero velocity. Not an exact Isaac first-decision replay.')
        snapshot = snapshot_start(plant, source=provenance)
        path = CATALOG.parent / f'supine_{mid}.json'
        write(path, snapshot)
        starts[f'supine{mid}'] = path.name
        print(f'Supine {mid}: at rest, pelvis {snapshot["root_position_m"][2]:.3f}m, '
              f'100ms passive drift {info["passive_root_drift_100ms_m"]*1000:.2f}mm', flush=True)
    cases = {}
    for mid in (151, 1444):
        for suffix in ('s50r', 's25'):
            cases[f'supine{mid}_{suffix}'] = dict(start='supine', state=starts[f'supine{mid}'],
                model=f'r8_{suffix}_diffusion', seeds=[50, 44, 42], replan_hz=5.)
    for start, model, seeds in [('supine', 'r8_s25_diffusion', [50, 44, 42]),
                                 ('prone', 'r2_f50r_flow', [49, 101, 707])]:
        for suffix, candidate in [('', model), ('_s50r', 'r8_s50r_diffusion')]:
            cases[f'earlier_{start}{suffix}'] = dict(start=start, state=f'../flat_three/{start}.json',
                model=candidate, seeds=seeds, replan_hz=5.)
    references = []
    for i in range(3):
        source = SOURCE / 'goals' / f'flat_pose_{i}.json'
        target = CATALOG.parent / 'reference_routes' / source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
        references.append(dict(path=str(target.relative_to(CATALOG.parent)), sha256=digest(source)))
    write(CATALOG, dict(schema='hv2.stanley_flat_comparison.v1', source_commit='620b34f10e97eea1d19132ce301754b0572eb2ce',
        source_bundle=str(SOURCE), source_evidence=str(EVIDENCE), sensor_seed=42,
        seconds=12., backend='tensorrt', noise_policy='fresh', cases=cases, reference_routes=references,
        scope='Unguided resting-start comparison. Nominal MuJoCo plant, 0.5s startup limiter, no joint jitter. Original 2.5Hz examples remain separate. No hardware or steering qualification.'))


def command_for(catalog, name, *, viewer=False, seeds=None, rate=None, output=None, seconds=None):
    row = catalog['cases'][name]
    cmd = [str(ROOT / '.venv/bin/python'), str(PACKAGE_ROOT / 'cli/sim.py'),
        '--model', row['model'], '--backend', catalog['backend'], '--terrain', 'flat',
        '--start', row['start'], '--start-state', str((CATALOG.parent / row['state']).resolve()),
        '--seed-list', *map(str, seeds or row['seeds']), '--sensor-seed', str(catalog['sensor_seed']),
        '--pose-seed', '42', '--noise-policy', 'fresh', '--replan-hz', str(row['replan_hz'] if rate is None else rate),
        '--command-limits', 'bounded', '--depth-startup', 'prefill', '--tracker-startup', 'repeat', '--fixed-camera']
    if viewer:
        cmd += ['--viewer']
    if seconds is not None or not viewer:
        cmd += ['--seconds', str(seconds if seconds is not None else catalog['seconds'])]
    if output is not None:
        cmd += ['--output', str(output)]
    return cmd


def run(args, catalog):
    cases = args.case or list(catalog['cases'])
    for name in cases:
        output = args.output / name
        if output.exists():
            raise FileExistsError(f'{output} already exists; choose another --output or specific --case')
        cmd = command_for(catalog, name, seeds=args.seeds, rate=args.replan_hz, output=output, seconds=args.seconds)
        args.output.mkdir(parents=True, exist_ok=True)
        print(shlex.join(cmd), flush=True)
        with (args.output / f'{name}.log').open('x') as stream:
            subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        rows = read(output / 'summary.json')
        print(name, [(r['planner_seed'], r['upright_hold_1s'], r['final_second_upright_fraction'])
                      for r in rows.values()], flush=True)


def report(output, catalog):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from itertools import combinations
    from steerable_recovery.tools.experiments.search_multimodal import describe_run

    groups = []
    for name in catalog['cases']:
        paths = sorted((output / name).glob('*/trajectory.npz'))
        if paths:
            groups.append((name, [p.parent for p in paths]))
    prior = ROOT / 'logs/multimodal_recovery/three_modes_20260915/confirmed'
    for start in ('prone', 'supine'):
        paths = sorted((prior / f'{start}_baseline').glob('*/trajectory.npz'))
        if paths:
            groups.append((f'original_{start}_2.5hz', [p.parent for p in paths]))
    if not groups:
        raise ValueError('No completed trajectories to analyze')
    fig, axes = plt.subplots((len(groups)+1)//2, 2, figsize=(11, 3.8*((len(groups)+1)//2)), squeeze=False)
    results = {}; physical_starts = {}
    colors = plt.get_cmap('tab10')
    for ax, (name, directories) in zip(axes.flat, groups):
        runs = []; endpoints = []; signatures = []
        for j, directory in enumerate(directories):
            result = describe_run(directory)
            summary = read(directory / 'summary.json')
            config = read(directory / 'config.json')
            with np.load(directory / 'first_condition.npz', allow_pickle=False) as f:
                initial = hashlib.sha256(f['initial_qpos'].tobytes()+f['initial_qvel'].tobytes()).hexdigest()
                start = config['fixed_start_snapshot']
                # Pose identity is independent of model and noise dimensions.
                key = hashlib.sha256(json.dumps(dict(q=start['joint_positions_rad'], xyz=start['root_position_m'],
                    quat=start['root_quat_xyzw']), sort_keys=True).encode()).hexdigest()
                if key in physical_starts and physical_starts[key] != initial:
                    raise AssertionError(f'Physical initialization changed across models: {name}')
                physical_starts[key] = initial
            result.update(model=config['model'], replan_hz=config['actual_replan_hz'],
                final_second_upright_fraction=summary['final_second_upright_fraction'],
                initial_speed=summary['reset']['initial_speed'])
            with np.load(directory / 'trajectory.npz', allow_pickle=False) as f:
                heading = float(f['causal_heading_rad'][0]); c, s = np.cos(heading), np.sin(heading)
                path = (f['root_position'][:,:2]-f['root_position'][0,:2]) @ np.array([[c,-s],[s,c]])
                height = f['root_position'][:,2]-f['terrain_height_m']
                # First sustained-upright entry, same physical time for each run.
                end = max(0, round((summary['first_upright_hold_s'] or summary['simulated_seconds'])/.02)-50)
                end = min(end, len(path)-1)
                result['upright_entry_xy_m'] = path[end].tolist()
                result['initial_height_m'] = float(height[0])
                result['ground_path_length_m'] = float(np.linalg.norm(np.diff(path, axis=0), axis=1)[height[:-1]<.3].sum())
                limited = (f['command_position_limited_joints']>0)|(f['command_rate_limited_joints']>0)
                if np.any(limited & (f['time']>=.5)):
                    raise AssertionError(f'Limiter remained active after 0.5s: {directory}')
                result['limited_after_half_second'] = False
                color = colors(j%10)
                label = f'seed {config["seed"]}: ' + ('quiet' if result['quiet_completion'] else 'not quiet')
                ax.plot(path[:,0], path[:,1], color=color, alpha=.65, label=label)
                low = np.where((height<.3)[:,None], path, np.nan)
                ax.plot(low[:,0], low[:,1], color=color, lw=3)
                ax.scatter(*path[end], color=color, marker='o' if result['quiet_completion'] else 'x')
                if result['quiet_completion']:
                    endpoints.append((config['seed'], path[end]))
            signatures.append(result['first_condition_sha256'])
            for key in ('body', 'route'):
                result.pop(key, None)
            runs.append(result)
        if len(set(signatures)) != 1:
            raise AssertionError(f'{name}: initial observations changed between seeds')
        pairs = [dict(seeds=[a,b], upright_entry_gap_m=float(np.linalg.norm(pa-pb)))
                 for (a,pa),(b,pb) in combinations(endpoints,2)]
        results[name] = dict(runs=runs, tested=len(runs), quiet=sum(r['quiet_completion'] for r in runs),
            same_first_condition=True, distinct_first_noise=len({r['first_noise_sha256'] for r in runs})==len(runs),
            pairwise=pairs)
        ax.set_title(f'{name}: {results[name]["quiet"]}/{len(runs)} quiet')
        ax.scatter(0,0,c='black',marker='+'); ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('X from start (m)'); ax.set_ylabel('Y from start (m)'); ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    fig.suptitle('Unguided MuJoCo at rest; thick path = pelvis below 30cm; dot = sustained-upright entry')
    fig.tight_layout(); fig.savefig(output/'comparison.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,6))
    for i, reference in enumerate(catalog['reference_routes']):
        path = CATALOG.parent / reference['path']
        if digest(path) != reference['sha256']:
            raise ValueError('Stanley reference route changed')
        target = read(path)['target']; position = np.asarray(target['positions_m'])
        ax.plot(position[:,0],position[:,1], label=f'Stanley route {i} / seed {target["episode_id"].split(":")[-1]}')
    ax.set_aspect('equal'); ax.legend(); ax.grid(alpha=.25)
    ax.set(xlabel='Initial-heading X (m)', ylabel='Initial-heading Y (m)',
           title='Stanley supine 151: recorded route targets\nReference only; no steering applied in this comparison')
    fig.tight_layout(); fig.savefig(output/'stanley_reference_routes.png', dpi=160); plt.close(fig)
    scope = ('Selected unguided examples, not a population success estimate. Quiet: >=1s upright, '
             '>=90% of final second upright, final joint-speed RMS <0.5 rad/s. '
             'Nominal MuJoCo, actual Warp depth, TensorRT FP32, 0.5s limiter, rest at t=0. '
             'Stanley reference routes used different Isaac initialization and PyTorch TF32; no exact trajectory parity claimed.')
    write(output/'comparison.json', dict(scope=scope, same_physics_across_models=True, cases=results))
    lines = ['# Flat resting-start comparison', '', scope, '',
             '|Case|Quiet completions|Median upright hold (s)|Pairwise upright-entry separation (m)|',
             '|---|---|---|---|']
    for name, row in results.items():
        times = [r['recovery_hold_s'] for r in row['runs'] if r['quiet_completion']]
        gaps = [p['upright_entry_gap_m'] for p in row['pairwise']]
        times_text = f'{np.median(times):.2f}' if times else '—'
        gap_text = f'{min(gaps):.2f}–{max(gaps):.2f}' if gaps else '—'
        lines.append(f'|{name}|{row["quiet"]}/{row["tested"]}|{times_text}|{gap_text}|')
    lines += ['', 'All completed runs, including failures, are retained. All per-case initial conditions matched; only planner noise varied.',
              '', '![Comparison](comparison.png)', '', '![Stanley reference routes](stanley_reference_routes.png)', '']
    (output/'README.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:6+len(results)]), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('prepare', 'run', 'view', 'report'))
    p.add_argument('--case', nargs='+', help='Case names in the comparison catalog; view requires one')
    p.add_argument('--seeds', type=int, nargs='+', help='Override the fixed comparison seeds')
    p.add_argument('--replan-hz', type=float, help='Override cadence; the primary comparison is 5 Hz')
    p.add_argument('--seconds', type=float, help='Default: 12 seconds headless, unlimited live')
    p.add_argument('--output', type=Path, default=RESULTS)
    p.add_argument('--print-command', action='store_true')
    args = p.parse_args(argv)
    args.output = args.output.expanduser().resolve()
    if args.command == 'prepare':
        prepare()
        return
    catalog = read(CATALOG)
    for name in args.case or []:
        if name not in catalog['cases']:
            p.error(f'Unknown case {name}; choose from {", ".join(catalog["cases"])}')
    if args.command == 'view':
        if not args.case or len(args.case) != 1:
            p.error('view requires exactly one --case')
        output = args.output / f'live_{args.case[0]}_{time.time_ns()}'
        cmd = command_for(catalog, args.case[0], viewer=True, seeds=args.seeds,
                          rate=args.replan_hz, seconds=args.seconds, output=output)
        print(shlex.join(cmd), flush=True)
        if not args.print_command:
            import os
            os.execv(cmd[0], cmd)
    elif args.command == 'run':
        if args.print_command:
            for name in args.case or catalog['cases']:
                print(shlex.join(command_for(catalog, name, seeds=args.seeds, rate=args.replan_hz,
                    seconds=args.seconds, output=args.output/name)))
        else:
            run(args, catalog)
    else:
        report(args.output, catalog)


if __name__ == '__main__':
    main()
