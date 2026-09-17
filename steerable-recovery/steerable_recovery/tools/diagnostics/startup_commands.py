#!/usr/bin/env python3
"""Headless, matched comparisons of startup command shaping.

Uses sim_diffusion.py model/terrain/seed options. Example:
  .venv/bin/python g1_deploy/compare_startup_commands.py --model r8_s25_diffusion \
    --backend tensorrt --terrain fork3 --start prone --seconds 10

All arms run the planner normally, observe actual motion and feed back the
issued action. This is an experiment, not a change to hardware deployment.
"""
from __future__ import annotations

import argparse
from copy import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def startup_fraction(tick, mode, seconds, dt=.02):
    """Scale target error about the measured pose; retain the original damping."""
    if mode == 'vigor':
        # deploy_vigor.py increments act_count before testing act_count < 50.
        return .5 if tick + 1 < 50 else 1.
    if mode == 'smooth':
        u = min(1., (tick + 1) * dt / seconds)
        return u * u * (3. - 2. * u)
    if mode == 'gentle':
        u = min(1., tick * dt / seconds)
        return .5 + .5 * u * u * (3. - 2. * u)
    return 1.


def shape_startup(action, q, default, scale, fraction):
    import numpy as np
    action = np.asarray(action)
    if fraction == 1.:
        return action.copy()
    target = default + scale * action
    return ((q + fraction * (target - q) - default) / scale).astype(np.float32)


def restore_recorded_start(plant, session):
    """Match recorded joints/orientation; infer only floor clearance from geometry.

    This cannot reconstruct missing depth, unknown base translation/velocity,
    or real contacts. It deliberately does not resettle into a different pose.
    """
    import gzip
    import mujoco
    import numpy as np
    path = Path(session)
    if path.is_dir(): path = path / 'telemetry.jsonl.gz'
    with gzip.open(path, 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    contract = next(row for row in rows if row.get('event') == 'session_contract')
    row = next(row for row in rows if row.get('event') == 'control')
    if contract['planner_contract']['dof_names'] != plant.names:
        raise ValueError('Recorded joint names differ from simulation')
    if any(plant.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE for g in plant.terrain_geoms):
        raise ValueError('Recorded-start comparison currently requires flat terrain')
    mujoco.mj_resetData(plant.model, plant.data)
    plant.data.qpos[plant.qadr] = row['q']
    plant.data.qvel[plant.vadr] = row['dq']
    plant.data.qvel[3:6] = row['gyro_body']
    root = plant.root_q
    plant.data.qpos[root:root+3] = [0., 0., 1.]
    plant.data.qpos[root+3:root+7] = np.asarray(row['pelvis_quat_xyzw'])[[3,0,1,2]]
    mujoco.mj_forward(plant.model, plant.data)
    bottom = min(mujoco.mj_geomDistance(plant.model,plant.data,plant.floor,int(g),10.,None)
                 for g in plant.collisions)
    plant.data.qpos[root+2] -= bottom
    mujoco.mj_forward(plant.model, plant.data)
    obs = plant.observation()
    for key, value in [('q',obs.q),('dq',obs.dq),('pelvis_quat_xyzw',obs.quat_xyzw),('gyro_body',obs.gyro_body)]:
        np.testing.assert_allclose(value,row[key],atol=2e-6)
    if plant.command_limiter:plant.command_limiter.reset()
    plant.last_command_limit = None
    plant.reset_info = dict(recorded_start=str(path.resolve()),inferred_pelvis_z_m=float(plant.data.qpos[root+2]),
        posture='recorded',settled=False,initial_contacts=int(plant.data.ncon),
        limitation='Measured joints/orientation/gyro matched; base XYZ/linear velocity and visual inputs are not reconstructed')
    return rows


def compare(run, own, policy, plant, capture, args, output, seed):
    import mujoco
    import numpy as np
    import torch
    from motion_diffusion.behavioral import mujoco as plant_module
    from motion_diffusion.behavioral import deploy as policy_module

    base_config = json.loads((output / 'config.json').read_text())
    # Record source snapshots and compiled physics alongside the measurements.
    for name, source in [('plant', plant_module), ('policy', policy_module)]:
        text = inspect.getsource(source)
        (output / f'{name}_source.py').write_text(text)
        base_config[f'{name}_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    mujoco.mj_saveModel(plant.model, str(output / 'physics.mjb'), None)
    original_step = plant.step
    original_reset = plant.reset
    limiter = plant.command_limiter
    original_mode = plant.command_limit_mode
    controller = policy.controller
    original_forward = controller.forward
    depth_encoder = controller.depth_encoder
    original_depth = depth_encoder.forward if depth_encoder is not None else None
    results = {}
    first = {}
    if own.start_session:
        plant.reset = lambda *a, **kw: restore_recorded_start(plant, own.start_session)
    try:
        for arm in own.arms:
            directory = output / arm
            directory.mkdir()
            selected = copy(args)
            if own.n_start:
                from g1_deploy.sim_diffusion import sample_start_offset
                selected.start_offset = sample_start_offset(args.pose_seed,
                    xy_m=args.n_xy_m, tilt_deg=args.n_tilt_deg, yaw_deg=args.n_yaw_deg)
                selected.start_jitter = args.n_joint_jitter
            mode = 'bounded' if arm.startswith('bounded') else 'training'
            soften = arm.removeprefix('bounded_')
            plant.command_limiter = limiter if mode == 'bounded' else None
            plant.command_limit_mode = mode
            tick = 0
            trace = []
            embedding = None

            def encode(*a, **kw):
                nonlocal embedding
                embedding = original_depth(*a, **kw)
                return embedding

            def forward(history, code, **kw):
                result = original_forward(history, code, **kw)
                if tick < round(own.trace_seconds / plant.control_dt):
                    if getattr(policy.planner, '_request_history', None) is not None:
                        torch.testing.assert_close(policy.planner._request_history[:, -1],
                            history.reshape(1, 10, -1)[:, -1, -29:], rtol=0, atol=0)
                        torch.testing.assert_close(policy.planner._measured_history[:, -1],
                            history.reshape(1, 10, -1)[:, -1, :64], rtol=0, atol=0)
                    pieces = [history.clamp(-controller.observation_clip, controller.observation_clip), code]
                    if embedding is not None:
                        pieces.append(embedding)
                    trace.append(dict(features=torch.cat(pieces, -1)[0].detach().cpu().numpy().copy(),
                        raw=controller.last_unclipped_action[0].cpu().numpy().copy(),
                        measured_history=history[0].detach().cpu().numpy().copy()))
                return result

            def step(action):
                nonlocal tick
                fraction = startup_fraction(tick, soften, own.startup_seconds, plant.control_dt)
                shaped = shape_startup(action, plant.data.qpos[plant.qadr], plant.default, plant.scale, fraction)
                result = original_step(shaped)
                if tick < len(trace):
                    trace[tick]['startup_fraction'] = fraction
                    trace[tick]['min_contact_distance'] = min((c.dist for c in plant.data.contact), default=0.)
                tick += 1
                return result

            controller.forward = forward
            if depth_encoder is not None:
                depth_encoder.forward = encode
            plant.step = step
            (directory / 'config.json').write_text(json.dumps(dict(base_config,
                comparison_arm=arm, startup_seconds=own.startup_seconds, command_limit_mode=mode,
                n_start=own.n_start, start_offset=getattr(selected, 'start_offset', {})), indent=2))
            print(f'\nStartup comparison: {arm}', flush=True)
            results[arm] = run(policy, plant, capture, selected, directory, seed)
            if trace:
                np.savez_compressed(directory / 'tracker_trace.npz',
                    **{key: np.asarray([v[key] for v in trace]) for key in trace[0]})
            with np.load(directory / 'first_condition.npz') as data:
                first[arm] = {key: data[key].copy() for key in data.files}
            with np.load(directory / 'trajectory.npz') as data:
                results[arm]['max_dq'] = float(np.abs(data['dq']).max())
                results[arm]['max_target_rad'] = float(np.abs(data['target']).max())
                results[arm]['first_raw_above_100_s'] = next(
                    (float(t) for t, a in zip(data['time'], data['raw_action_max']) if a > 100.), None)
                # This is the history actually consumed by the tracker, not a reconstructed guess.
                if len(trace) > 1:
                    rows = np.asarray([v['measured_history'] for v in trace]).reshape(len(trace), 10, -1)
                    np.testing.assert_array_equal(rows[1:, -1, -29:], data['action'][:len(trace)-1])
                np.testing.assert_allclose(plant.default + plant.scale * data['action'], data['target'], atol=2e-6)
        reference = first[own.arms[0]]
        identical = all(set(values) == set(reference) and all(np.array_equal(reference[k], values[k], equal_nan=True)
                        for k in reference) for values in first.values())
        report = dict(initial_conditions_identical=identical, actual_issued_feedback_verified=True,
            interpretation='Normal replanning in every arm; only command shaping changes. Later states, depth and plans can diverge.',
            results=results)
        (output / 'comparison.json').write_text(json.dumps(report, indent=2))
        if not identical:
            raise AssertionError('Comparison initial state/history/depth/noise mismatch')
        print('Matched conditions and actual issued-action feedback verified.', flush=True)
        return dict(status='duration_reached', comparison=report)
    finally:
        plant.step = original_step
        plant.reset = original_reset
        plant.command_limiter = limiter
        plant.command_limit_mode = original_mode
        controller.forward = original_forward
        if depth_encoder is not None:
            depth_encoder.forward = original_depth


def main():
    extra = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    extra.add_argument('--arms', nargs='+', choices=('training', 'bounded', 'vigor', 'smooth', 'bounded_smooth', 'bounded_vigor', 'gentle', 'bounded_gentle'),
                       default=['training', 'vigor', 'smooth', 'bounded', 'bounded_smooth'])
    extra.add_argument('--startup-seconds', type=float, default=.5)
    extra.add_argument('--trace-seconds', type=float, default=2.)
    extra.add_argument('--n-start', action='store_true', help='Use the same nearby pose sampling as N, at --pose-seed')
    extra.add_argument('--start-session',type=Path,help='Use recorded starting joints/orientation on flat ground, without resettling')
    if '-h' in sys.argv or '--help' in sys.argv:
        extra.print_help()
        print('\nAlso accepts sim_diffusion.py options. Headless only; no video or robot connection.')
        return
    own, remaining = extra.parse_known_args()
    import math
    if not all(math.isfinite(v) and v > 0 for v in (own.startup_seconds, own.trace_seconds)):
        extra.error('Startup/trace durations must be positive and finite')
    if len(set(own.arms)) != len(own.arms):
        extra.error('Comparison arms must be unique')
    sys.argv = [sys.argv[0], *remaining]
    from g1_deploy import sim_diffusion as sim
    original_arguments, original_run = sim.arguments, sim.run_episode
    def arguments():
        args = original_arguments()
        if args.viewer or args.video or args.seconds is None or args.seconds < .02:
            raise ValueError('Comparison needs a finite headless run of at least .02 s; no video')
        args.command_limits = 'bounded'  # Construct bounds once; each arm selects whether to use them.
        return args
    sim.arguments = arguments
    sim.run_episode = lambda policy, plant, capture, args, output, seed, *unused, **kw: compare(
        original_run, own, policy, plant, capture, args, output, seed)
    try:
        sim.main()
    finally:
        sim.arguments, sim.run_episode = original_arguments, original_run


if __name__ == '__main__':
    main()
