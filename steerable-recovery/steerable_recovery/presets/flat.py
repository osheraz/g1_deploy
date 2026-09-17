#!/usr/bin/env python3
"""Selected flat-ground recoveries from one prone and one supine start.

Simulation opens the live viewer, with depth, and records no video.
--robot runs the normal deployment entry point with its recording options.
"""
from __future__ import annotations
from steerable_recovery.paths import RESULTS_ROOT, ROOT, CONFIG_ROOT, PACKAGE_ROOT

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import time

DEFAULT_CATALOG = CONFIG_ROOT / 'multimodal/flat_three/catalog.json'
from steerable_recovery.settings import DIFFUSION_RATES


def command_for(args, catalog):
    row = catalog['starts'][args.start]
    selected = row['options'][args.option - 1]
    model = args.model or row['model']
    rate = args.replan_hz if args.replan_hz is not None else (
        DIFFUSION_RATES[model] if args.model else selected['replan_hz'])
    seed = str(args.seed if args.seed is not None else selected['seed'])
    folder = args.catalog.resolve().parent
    common = ['--model', model, '--backend', 'tensorrt',
              '--noise-policy', 'fresh', '--replan-hz', str(rate)]
    if args.robot:
        session = args.output or RESULTS_ROOT / (
            (f'{model}_{args.start}_' if args.model else f'three_{args.start}_option{args.option}_') +
            time.strftime('%Y%m%d_%H%M%S'))
        cmd = [sys.executable, str(PACKAGE_ROOT / 'cli/deploy.py'), *common,
               '--start-pose', str(folder / row['robot_pose']), '--seed', seed,
               '--actuate', '--release-controller', '--duration-s', str(60 if args.seconds is None else args.seconds),
               '--record-sensors', '--record-depth', '--record-model',
               '--record-label', f'flat_{args.start}_{model}_option{args.option}', '--output', str(session)]
        if args.live_odometry:
            cmd += ['--live-odometry']
        if args.external_camera:
            cmd += ['--external-camera', '--camera-size', '1920x1080', '--camera-fps', '30']
    else:
        cmd = [sys.executable, str(PACKAGE_ROOT / 'cli/sim.py'), *common,
               '--terrain', 'flat', '--start', args.start,
               '--start-state', str(folder / row['sim_state']),
               '--sensor-seed', str(catalog['sensor_seed']), '--pose-seed', '42',
               '--command-limits', 'bounded', '--depth-startup', 'prefill',
               '--tracker-startup', 'repeat', '--fixed-camera', '--viewer']
        # Cycle the selected examples only when their sampling rates agree.
        if args.seed is None and (args.replan_hz is not None or
                                 len({o['replan_hz'] for o in row['options']}) == 1):
            options = row['options'][args.option - 1:] + row['options'][:args.option - 1]
            cmd += ['--seed-list', *[str(o['seed']) for o in options]]
        else:
            cmd += ['--seed', seed]
        if args.seconds is not None:
            cmd += ['--seconds', str(args.seconds)]
        if args.output is not None:
            cmd += ['--output', str(args.output)]
    if args.start_pose_jitter_deg:
        cmd += ['--start-pose-jitter-deg', str(args.start_pose_jitter_deg)]
        if args.start_pose_seed is not None:
            cmd += ['--start-pose-seed', str(args.start_pose_seed)]
    return cmd


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start', choices=('prone', 'supine'), required=True)
    p.add_argument('--model', choices=tuple(DIFFUSION_RATES),
                   help='Compare a diffusion model on the same saved joints; S25 defaults to 2.5 Hz, S50R to 5 Hz')
    p.add_argument('--option', type=int, choices=(1, 2, 3), default=1)
    p.add_argument('--robot', action='store_true', help='Enable normal robot deployment; default is simulation')
    p.add_argument('--catalog', type=Path, default=DEFAULT_CATALOG)
    p.add_argument('--seed', help='Override selected example: integer or random')
    p.add_argument('--replan-hz', type=float, help='Override the evaluated rate; changes the result')
    p.add_argument('--start-pose-jitter-deg', type=float, default=0., help='Optional nearby-start experiment, up to +/-5 degrees')
    p.add_argument('--start-pose-seed', help='Independent joint-perturbation seed')
    p.add_argument('--seconds', type=float, help='Default: unlimited viewer; 60 seconds on robot')
    p.add_argument('--output', type=Path)
    p.add_argument('--external-camera', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--live-odometry', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--print-command', action='store_true', help='Print only; no models, sensors or motor commands')
    args = p.parse_args(argv)
    if not 0 <= args.start_pose_jitter_deg <= 5:
        p.error('--start-pose-jitter-deg must be between 0 and 5')
    if args.start_pose_seed is not None and not args.start_pose_jitter_deg:
        p.error('--start-pose-seed requires nonzero joint jitter')
    catalog = json.loads(args.catalog.read_text())
    if catalog.get('schema') != 'hv2.flat_recovery_options.v1':
        p.error('Unknown recovery-options catalog schema')
    cmd = command_for(args, catalog)
    print(shlex.join(cmd), flush=True)
    if args.print_command:
        return
    if args.model or args.seed is not None or args.replan_hz is not None or args.start_pose_jitter_deg:
        print('Exploration override: the selected example may change.', flush=True)
    # Reuse the existing model/recipe check, then the normal launcher verifies
    # asset bytes and applies its usual initialization and operator controls.
    sys.path.insert(0, str(ROOT))
    from steerable_recovery.presets.multimodal import configured_bundle
    row = catalog['starts'][args.start]
    if args.model and args.model != row['model']:
        # This is a model comparison, not a reproduction of the catalog's
        # selected motion. Each normal launcher validates its own paired assets.
        import yaml
        from steerable_recovery.settings import model_settings
        from motion_diffusion.behavioral.hardware.bundle import load_bundle
        settings=model_settings(yaml.safe_load((CONFIG_ROOT / 'models.yaml').read_text()),args.model)
        bundle,manifest=load_bundle(settings['bundle'])
        if args.model not in manifest['models']:raise ValueError('Selected model is absent from its configured bundle')
    else:bundle = configured_bundle(row)
    cmd += ['--bundle', str(bundle)] if not args.robot else []
    # Replace this wrapper: Ctrl+C and recorder cleanup belong to the existing
    # entry point, with no second process waiting or intercepting interrupts.
    os.chdir(ROOT)
    os.execv(sys.executable, cmd)


if __name__ == '__main__':
    main()
