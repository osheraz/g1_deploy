"""Public commands for the maintained deployment and simulation runtimes."""
import argparse
import sys


def run(mode, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if mode not in ('sim', 'deploy'):
        raise ValueError(f'Unknown recovery entry point: {mode}')
    if mode == 'sim' and '--actuate' in argv:
        argparse.ArgumentParser().error('The sim entry point cannot enable robot deployment')
    if mode == 'sim':
        from .cli.sim import main
        return main(argv)
    from .cli.deploy import main
    return main(argv)
