"""Model selection and command settings shared by hardware and simulation."""
import argparse
import os
from pathlib import Path
from .paths import recovery_asset

PUBLISHED_MODELS = ('ours', 'joint', 'depth_blind', 'blind_blind')
RETAINED_MODELS = ('s25r_diffusion', 's50r_diffusion', 'r8_s50r_diffusion', 'r8_s25_diffusion', 'r2_f50r_flow')
MODEL_NAMES = PUBLISHED_MODELS + RETAINED_MODELS
DIFFUSION_RATES = {'r8_s25_diffusion': 2.5, 'r8_s50r_diffusion': 5.,
                   's25r_diffusion': 2.5, 's50r_diffusion': 2.5}

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


def model_settings(settings, model, *, bundle=None):
    """Explicit bundle wins; configured pairs precede the environment fallback."""
    from copy import deepcopy
    if model not in MODEL_NAMES:
        raise ValueError(f'Unsupported recovery model: {model}')
    selected=deepcopy(settings)
    selected.update(selected.get('models',{}).get(model,{}))
    if bundle is not None:
        selected['bundle']=str(Path(bundle).expanduser().resolve())
    elif selected.get('bundle'):
        selected['bundle']=str(recovery_asset(selected['bundle']))
    elif os.environ.get('HV2_RECOVERY_BUNDLE'):
        selected['bundle']=str(Path(os.environ['HV2_RECOVERY_BUNDLE']).expanduser().resolve())
    else:
        raise ValueError(f'No bundle configured for {model}. Prepare its matching bundle with '
                         'scripts/tools/recovery.py deployment-catalog and deploy prepare, then pass '
                         '--bundle PATH or set HV2_RECOVERY_BUNDLE; see docs/guides/deployment.md')
    selected['exports']={name: str(recovery_asset(path))
                         for name, path in selected.get('exports', {}).items()}
    return selected


def hardware_settings(config, contract):
    from motion_diffusion.inference.sensors import motor_indices
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
