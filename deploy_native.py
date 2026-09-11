"""Current diffusion/flow recovery: use hv2's shared sim/robot inference runtime.

Example: python deploy_native.py run --model diffusion --bundle ... --hardware ... --output ...
No motor publisher is created without --actuate and a commissioning receipt.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / 'motion_diffusion/behavioral/hardware/bundle.py').is_file():
    raise RuntimeError('Run g1_deploy inside the hv2 checkout; the native runtime is shared, not copied here')
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    from motion_diffusion.behavioral.hardware.bundle import main
    main()
