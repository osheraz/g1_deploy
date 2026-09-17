#!/usr/bin/env python3
"""Isolate tracker pre-start history with matched physics and identical latent actions.

Accepts sim_diffusion.py arguments; headless only. Example:
  .venv/bin/python g1_deploy/compare_tracker_startup.py --model r8_s25_diffusion \
    --backend tensorrt --terrain fork3 --start prone --seconds 2

Three arms: normal planning with repeated tracker history; an identical-code
replay with repeated history (control); identical-code replay with measured
tracker prehistory. All arms receive the same passive pre-start interval.
The replay arms do not replan in response to their changed trajectories.
"""
from __future__ import annotations

import argparse
from copy import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class RecordedPlanner:
    """Replay the exact z_t consumed in a reference run; no planner inference."""
    def __init__(self,source,directory,device):
        import numpy as np
        import torch
        self.model=source.model;self.contract=source.contract
        self.relative_attitude=source.relative_attitude
        self.device=device
        with np.load(directory/'trajectory.npz') as data:
            self.codes=torch.tensor(data['native_code'],device=device)
        self.plans={row['tick']:row for row in json.loads((directory/'plans.json').read_text())}
        # These are the reference planner conditions, deliberately held fixed.
        with np.load(directory/'first_condition.npz') as data:
            self.hist={key[8:]:torch.tensor(data[key],device=device) for key in data.files if key.startswith('history_')}
            self.state_noise=torch.tensor(data['state_noise'],device=device)
            self.action_noise=torch.tensor(data['code_noise'],device=device)

    def predict_code(self,measured,request,previous,*,control_tick,**kwargs):
        import torch
        tick=int(control_tick[0])
        if not 0<=tick<len(self.codes):raise ValueError('Reference latent sequence is exhausted')
        row=self.plans[max(t for t in self.plans if t<=tick)]
        self.last_predicted_states=torch.tensor([row['state']],device=self.device)
        self.last_predicted_codes=torch.tensor([row['native_code']],device=self.device)
        self.last_replanned_mask=torch.tensor([tick in self.plans],device=self.device)
        return self.codes[tick:tick+1]


def compare_episode(run,initial_path,policy,plant,capture,args,output,seed):
    import mujoco
    import numpy as np
    from motion_diffusion.behavioral.deploy import NativeRecoveryPolicy
    base_config=json.loads((output/'config.json').read_text())
    reset=plant.reset
    if initial_path:
        with np.load(initial_path) as data:
            qpos=data['initial_qpos'].copy();qvel=data['initial_qvel'].copy()
        def restore(*a,**kw):
            reset(*a,**kw)
            if qpos.shape!=plant.data.qpos.shape or qvel.shape!=plant.data.qvel.shape:
                raise ValueError('Saved initial state does not match the plant')
            mujoco.mj_resetData(plant.model,plant.data)
            plant.data.qpos[:]=qpos;plant.data.qvel[:]=qvel
            mujoco.mj_forward(plant.model,plant.data)
            plant.reset_info['initial_condition_source']=str(initial_path)
        plant.reset=restore
    results={}
    try:
        for arm in ('reference','replay_repeat','replay_measured'):
            directory=output/arm;directory.mkdir()
            selected=copy(args)
            selected.tracker_startup='measured' if arm=='replay_measured' else 'repeat'
            chosen=policy
            if arm!='reference':
                planner=RecordedPlanner(policy.planner,output/'reference',policy.controller.device)
                chosen=NativeRecoveryPolicy(policy.controller,planner)
                selected.seconds=len(planner.codes)*plant.control_dt
            (directory/'config.json').write_text(json.dumps(dict(base_config,
                tracker_startup=selected.tracker_startup,seconds=selected.seconds,
                planner_execution='live' if arm=='reference' else 'recorded_z_sequence',
                initial_condition_source=str(initial_path) if initial_path else None,
                plant_source_sha256=hashlib.sha256((ROOT/'motion_diffusion/behavioral/mujoco.py').read_bytes()).hexdigest()),indent=2))
            print(f'\nTracker history comparison: {arm}',flush=True)
            results[arm]=run(chosen,plant,capture,selected,directory,seed)
    finally:plant.reset=reset
    arrays={arm:dict(np.load(output/arm/'trajectory.npz')) for arm in results}
    first={arm:dict(np.load(output/arm/'first_condition.npz')) for arm in results}
    reference=arrays['reference']
    def identical_initial(arm):
        keys=('initial_qpos','initial_qvel','state_noise','code_noise','depth_frames',
              'depth_metadata','depth_available','tracker_depth_frames','tracker_depth_metadata','tracker_depth_available')
        return all(np.array_equal(first['reference'][k],first[arm][k],equal_nan=True)
                   for k in keys if k in first['reference'])
    control=arrays['replay_repeat']
    control_matches=(len(reference['qpos'])==len(control['qpos']) and all(
        np.array_equal(reference[k],control[k]) for k in ('qpos','action','native_code','raw_action_max')))
    checks=dict(identical_initial_states_and_camera=all(identical_initial(arm) for arm in results),
        identical_latent_sequences=all(np.array_equal(arrays[arm]['native_code'],reference['native_code'][:len(arrays[arm]['native_code'])]) for arm in results),
        repeated_history_replay_exactly_matches_reference=control_matches,
        tracker_history_initial_max_difference=float(np.max(np.abs(
            first['replay_repeat']['tracker_history']-first['replay_measured']['tracker_history']))))
    metric_keys=('status','frames','simulated_seconds','max_raw_action','saturated_action_ticks',
                 'upright_hold_1s','first_upright_hold_s','final_second_upright_fraction','physics_warnings')
    report=dict(seed=seed,tracker_preroll_s=args.tracker_preroll_s,checks=checks,
        interpretation='Only tracker initialization differs at takeover. Replays use the same z_t; subsequent physics and depth may diverge. This is not a replanning-policy success benchmark.',
        arms={arm:{k:row[k] for k in metric_keys} for arm,row in results.items()})
    for arm,data in arrays.items():
        report['arms'][arm].update(max_joint_speed_rad_s=float(np.max(np.abs(data['dq']))),
            max_torque_nm=float(np.max(data['torque_peak'])),
            motor_saturated_substeps=int(np.sum(data['torque_clipped'])))
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not all(checks[k] for k in ('identical_initial_states_and_camera','identical_latent_sequences',
                                  'repeated_history_replay_exactly_matches_reference')):
        raise RuntimeError('Tracker comparison control check failed; see comparison.json')
    return dict(status='duration_reached',comparison=report)


def main():
    extra=argparse.ArgumentParser(add_help=False)
    extra.add_argument('--initial-condition',type=Path,
                       help='Optional saved first_condition.npz; restore its qpos/qvel before common passive prehistory')
    if any(flag in sys.argv[1:] for flag in ('-h','--help')):
        print(__doc__)
        extra.print_help()
        print('\nShared model/terrain/seed options: sim_diffusion.py --help.\n'
              'Defaults: 10 s per arm, .2 s passive prehistory. No viewer or video.\n'
              'Both tracker initialization modes run automatically; --tracker-preroll-s changes their common pre-start interval.\n'
              'Command limits default to training here, preserving the original comparison; override explicitly if needed.')
        return
    own,remaining=extra.parse_known_args()
    if own.initial_condition:own.initial_condition=own.initial_condition.resolve()
    # Preserve the unmodified execution used by the history-isolation experiment.
    if not any(value=='--command-limits' or value.startswith('--command-limits=') for value in remaining):
        remaining.extend(['--command-limits','training'])
    sys.argv=[sys.argv[0],*remaining]
    from g1_deploy import sim_diffusion as sim
    arguments=sim.arguments;run=sim.run_episode
    def comparison_arguments():
        args=arguments()
        if args.viewer or args.video:raise ValueError('Tracker comparison is headless; use sim_diffusion.py for viewing')
        if args.seconds is None or round(args.seconds/.02)<1:
            raise ValueError('Tracker comparison needs a finite duration of at least one control tick')
        if args.tracker_preroll_s==0:args.tracker_preroll_s=.2
        return args
    def comparison_run(policy,plant,capture,args,output,seed,viewer=None,controls=None):
        return compare_episode(run,own.initial_condition,policy,plant,capture,args,output,seed)
    sim.arguments=comparison_arguments;sim.run_episode=comparison_run
    try:sim.main()
    finally:sim.arguments=arguments;sim.run_episode=run


if __name__=='__main__':main()
