"""Model selection and command settings shared by hardware and simulation."""
import argparse
from pathlib import Path
from .paths import ROOT

MODEL_NAMES = ('ours', 'joint', 'depth_blind', 'blind_blind')

def sampling_seed(value):
    if value == 'random':
        import secrets
        return secrets.randbits(63)
    try:
        seed=int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Seed must be an integer or random') from exc
    if not 0<=seed<2**63:
        raise argparse.ArgumentTypeError('Seed must be in [0, 2**63)')
    return seed


def model_settings(settings, model):
    """Resolve one published pair; never substitute a retired model alias."""
    from copy import deepcopy
    if model not in MODEL_NAMES:
        raise ValueError(f'Unsupported recovery model: {model}')
    selected=deepcopy(settings)
    selected.update(selected.get('models',{}).get(model,{}))
    path=Path(selected['bundle'])
    selected['bundle']=str(path if path.is_absolute() else ROOT/path)
    return selected


def hardware_settings(config, contract):
    from motion_diffusion.behavioral.hardware.sensors import motor_indices
    indices=motor_indices(contract['dof_names'])
    configured=config['action_joint2motor_idx']
    if len(configured)!=29 or set(configured)!=set(range(29)):
        raise ValueError('Diffusion deployment requires all 29 physical joints')
    order=[configured.index(i) for i in indices]
    return dict(tracker_device=config['tracker_device'],planner_device=config['planner_device'],
        robot=dict(interface=config['network_interface'],domain_id=0,tick_period_s=.001,
            tick_clock=None,max_command_age_s=config['max_command_age_s'],require_remote_start=True),
        camera=config['camera'],pd=dict(training_pd=None,mode_machine=config['mode_machine'],
            lower_joint_limits=[config['dof_lower'][i] for i in order],
            upper_joint_limits=[config['dof_upper'][i] for i in order],
            max_joint_velocity_rad_s=config['max_joint_velocity'],
            max_abs_requested_action=contract['action_clip']))
