"""Workspace paths for recovery tools; importing this module opens no devices."""
from pathlib import Path
import os
import sys

PROJECT = Path(__file__).resolve().parents[1]
DEPLOY_ROOT = PROJECT.parent
ROOT = next((p for p in PROJECT.parents if (p / 'motion_diffusion').is_dir()), None)
if ROOT is None:
    raise RuntimeError('Steerable Recovery requires the hv2 checkout with its g1_deploy submodule')
# Asset binding changes inputs only; new sessions remain in this checkout.
ASSET_ROOT = Path(os.environ.get('HV2_RECOVERY_ASSET_ROOT', str(ROOT))).expanduser().resolve()


def recovery_asset(path):
    """Resolve a workspace-relative retained input against the declared asset root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else ASSET_ROOT / path


PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = PROJECT / 'config'
RESULTS_ROOT = ROOT / 'logs/real_world'
VALIDATION_ROOT = ROOT / 'logs/deployment_validation'
PAPER_RESULTS_ROOT = ROOT / 'paper/real_world'
HIGHLIGHTS_ROOT = ROOT / 'logs/real_world_hightlight'
PAPER_HIGHLIGHTS_ROOT = ROOT / 'paper/real_world_hightlight'
ABORTED_ROOT = ROOT / 'logs/aborted_deployments'

# The model and shared deployment packages belong to this same hv2 checkout.
for directory in (ROOT, DEPLOY_ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
