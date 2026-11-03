"""Named starting poses and lab recipes for the deployment commands."""

import argparse
import sys


def dispatch(mode, argv=None):
    """Run a selected recipe; return False when ordinary command arguments remain."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument("--preset", choices=("prone", "supine"))
    choices.add_argument("--gallery")
    selected, remaining = parser.parse_known_args(argv)
    if mode == "sim" and (
        "--actuate" in remaining
        or ((selected.preset or selected.gallery) and "--robot" in remaining)
    ):
        parser.error("Simulation cannot enable robot deployment")
    robot = ["--robot"] if mode == "deploy" else []
    if selected.preset:
        from .flat import main

        main(["--start", selected.preset, *robot, *remaining])
        return True
    if selected.gallery:
        from .gallery import main

        main(["--example", selected.gallery, *robot, *remaining])
        return True
    return False


def add_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preset", choices=("prone", "supine"), help="Saved flat-ground start")
    group.add_argument("--gallery", metavar="NAME", help="Lab recipe; list prints available names")
