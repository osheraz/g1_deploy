"""Public commands; named examples reuse the same deployment/simulation runtime."""
import argparse
import sys


def run(mode, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument('--preset', choices=('prone', 'supine'))
    choices.add_argument('--route', type=int, choices=(1, 2, 3))
    choices.add_argument('--gallery', help='Selected lab run, e.g. robust_01_minimal; list shows available runs')
    selected, remaining = parser.parse_known_args(argv)
    if mode == 'sim' and ('--actuate' in remaining or
                         ((selected.preset or selected.route or selected.gallery) and '--robot' in remaining)):
        parser.error('The sim entry point cannot enable robot deployment')
    if selected.preset:
        from .presets.flat import main
        return main(['--start', selected.preset, *(['--robot'] if mode == 'deploy' else []), *remaining])
    if selected.route:
        from .presets.steering import main
        return main(['--option', str(selected.route), *(['--robot'] if mode == 'deploy' else []), *remaining])
    if selected.gallery:
        from .presets.gallery import main
        return main(['--example', selected.gallery, *(['--robot'] if mode == 'deploy' else []), *remaining])
    if '-h' in argv or '--help' in argv:
        print('Named modes: --preset prone|supine (flat starts); --route 1|2|3 (pelvis steering); --gallery NAME (terrain steering).\n'
              'Use MODE --help for its options. Without either, the full runtime options follow.\n')
    if mode == 'sim':
        from .cli.sim import main
        return main(remaining)
    from .cli.deploy import main
    return main(remaining)
