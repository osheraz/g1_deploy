"""Public commands for the maintained deployment and simulation runtimes."""
import argparse
import sys


def run(mode, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if mode not in ('sim', 'deploy'):
        raise ValueError(f'Unknown recovery entry point: {mode}')
    parser = argparse.ArgumentParser(add_help=False)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--preset', choices=('prone', 'supine'))
    modes.add_argument('--gallery', help='Retained Osher lab run; list prints the catalog')
    selected, remaining = parser.parse_known_args(argv)
    if mode == 'sim' and ('--actuate' in remaining or
            ((selected.preset or selected.gallery) and '--robot' in remaining)):
        parser.error('The sim entry point cannot enable robot deployment')
    if selected.preset:
        from .presets.flat import main
        return main(['--start', selected.preset, *(['--robot'] if mode == 'deploy' else []), *remaining])
    if selected.gallery:
        from .presets.gallery import main
        return main(['--example', selected.gallery, *(['--robot'] if mode == 'deploy' else []), *remaining])
    if '-h' in argv or '--help' in argv:
        print('Retained commands: --preset prone|supine; --gallery NAME|list.\n')
    if mode == 'sim':
        from .cli.sim import main
        return main(remaining)
    from .cli.deploy import main
    return main(remaining)
