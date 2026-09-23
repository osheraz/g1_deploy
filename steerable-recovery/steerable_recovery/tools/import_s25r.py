"""Import Stanley's completed S25R simulation handoff into a native bundle.

Copies the exact planner/tracker/config, verifies every delivered file, and
derives the existing PD binding. Opens no sensors and issues no motor commands.
"""
import argparse
import json
from pathlib import Path
import tempfile

from steerable_recovery.paths import ROOT
from motion_diffusion.behavioral.hardware.bundle import prepare, sha256

MODEL = 's25r_diffusion'


def import_handoff(source, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Bundle already exists: {output}')
    handoff = json.loads((source / 'HANDOFF.json').read_text())
    if (handoff.get('schema') != 'hv2.s25r_simulation_handoff.v1'
            or handoff.get('completed') is not True
            or handoff.get('model') != 'S25R_s42'
            or handoff.get('updates') != 40000):
        raise ValueError('Expected the completed S25R_s42 40k handoff')
    required = ('models/planner.pt', 'models/tracker.pt', 'config/controller_runtime.yaml')
    if not all(name in handoff.get('files', {}) for name in required):
        raise ValueError('Handoff is missing the planner, tracker or PD configuration')
    for name, digest in handoff['files'].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or sha256(path) != digest:
            raise ValueError(f'Handoff asset identity differs: {name}')
    inference = dict(handoff['inference'])
    if inference.pop('delay_ms', 0) != 0 or inference.pop('guidance', 'none') != 'none':
        raise ValueError('Expected the nominal unguided execution reference')
    inference.pop('seed', None)  # The deployment/simulator selects its episode seed.
    def asset(name):
        return dict(path=str(source / name), sha256=handoff['files'][name])
    row = dict(label='S25R_s42 diffusion40k + exact P42_s42 depth tracker22k',
        objective='diffusion', planner=asset(required[0]), controller=asset(required[1]),
        controller_training_config=asset(required[2]), inference=inference,
        evidence=dict(handoff=str(source), handoff_sha256=sha256(source / 'HANDOFF.json'),
            source_git=handoff['packaging_source'], hardware_qualified=False),
        deployment_timing=dict(max_plan_age_ticks=5,
            scope='F25/K20 leaves 5 ticks for asynchronous delivery; measure on the deployment GPU.',
            guided=dict(max_plan_age_ticks=5,
                scope='Experimental guidance; 100 ms reply budget, not a hardware qualification.')))
    catalog = dict(schema='hv2.robot_candidate_catalog.v1', models={MODEL: row})
    with tempfile.TemporaryDirectory(prefix='s25r_import_') as temporary:
        path = Path(temporary) / 'catalog.json'
        path.write_text(json.dumps(catalog, indent=2) + '\n')
        result = prepare(path, output, [MODEL], asset_mode='copy')
    (output / 'handoff.source.json').write_text(json.dumps(handoff, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff', required=True, type=Path)
    parser.add_argument('--output', type=Path,
        default=ROOT / 'logs/native_recovery/bundles/s25r_40k_20260921')
    args = parser.parse_args()
    result = import_handoff(args.handoff, args.output)
    print(f'Imported {MODEL}: {args.output}')
    contract = result['models'][MODEL]['inspected_contract']
    print(json.dumps({key: contract[key] for key in ('F', 'H', 'H_z', 'latent_dim')}))


if __name__ == '__main__':
    main()
