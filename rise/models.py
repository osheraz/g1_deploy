"""Model selection and command settings shared by hardware and simulation."""

import argparse
import os
from pathlib import Path
from rise.paths import recovery_asset

PUBLISHED_MODELS = ("ours", "joint", "depth_blind", "blind_blind")
RETAINED_MODELS = (
    "s25r_diffusion",
    "s50r_diffusion",
    "r8_s50r_diffusion",
    "r8_s25_diffusion",
    "r2_f50r_flow",
)
MODEL_NAMES = PUBLISHED_MODELS + RETAINED_MODELS
DIFFUSION_RATES = {
    "r8_s25_diffusion": 2.5,
    "r8_s50r_diffusion": 5.0,
    "s25r_diffusion": 2.5,
    "s50r_diffusion": 2.5,
}


def sampling_seed(value):
    if value == "random":
        import secrets

        return secrets.randbits(63)
    try:
        seed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seed must be an integer or random") from exc
    if not 0 <= seed < 2**63:
        raise argparse.ArgumentTypeError("Seed must be in [0, 2**63)")
    return seed


def model_settings(settings, model, *, bundle=None):
    """Explicit bundle wins; configured pairs precede the environment fallback."""
    from copy import deepcopy

    if model not in MODEL_NAMES:
        raise ValueError(f"Unsupported recovery model: {model}")
    selected = deepcopy(settings)
    selected.update(selected.get("models", {}).get(model, {}))
    if bundle is not None:
        selected["bundle"] = str(Path(bundle).expanduser().resolve())
    elif selected.get("bundle"):
        selected["bundle"] = str(recovery_asset(selected["bundle"]))
    elif os.environ.get("HV2_RECOVERY_BUNDLE"):
        selected["bundle"] = str(Path(os.environ["HV2_RECOVERY_BUNDLE"]).expanduser().resolve())
    else:
        raise ValueError(
            f"No bundle configured for {model}. Pass --bundle PATH or set "
            "HV2_RECOVERY_BUNDLE; see docs/guides/deployment.md for bundle preparation."
        )
    selected["exports"] = {
        name: str(recovery_asset(path)) for name, path in selected.get("exports", {}).items()
    }
    return selected


def hardware_settings(config, contract):
    from humanoidverse2.observations.streams import motor_indices

    indices = motor_indices(contract["dof_names"])
    configured = config["action_joint2motor_idx"]
    if len(configured) != 29 or set(configured) != set(range(29)):
        raise ValueError("Diffusion deployment requires all 29 physical joints")
    order = [configured.index(i) for i in indices]
    return dict(
        tracker_device=config["tracker_device"],
        planner_device=config["planner_device"],
        robot=dict(
            interface=config["network_interface"],
            domain_id=0,
            tick_period_s=0.001,
            tick_clock=None,
            max_command_age_s=config["max_command_age_s"],
            require_remote_start=True,
        ),
        camera=config["camera"],
        pd=dict(
            training_pd=None,
            mode_machine=config["mode_machine"],
            lower_joint_limits=[config["dof_lower"][i] for i in order],
            upper_joint_limits=[config["dof_upper"][i] for i in order],
            max_joint_velocity_rad_s=config["max_joint_velocity"],
            max_abs_requested_action=contract["action_clip"],
        ),
    )


def main(argv=None):
    """Prepare or inspect a model bundle without opening sensors."""
    import json
    from motion_diffusion.inference.bundle import prepare, load_bundle, asset_path, inspect_pair

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Assemble selected model pairs")
    prepare_parser.add_argument("--catalog", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--models", nargs="+", required=True)
    prepare_parser.add_argument("--asset-mode", choices=("copy", "symlink"), default="symlink")
    check_parser = commands.add_parser("check", help="Check bundle compatibility and assets")
    check_parser.add_argument(
        "--bundle",
        default=os.environ.get("HV2_RECOVERY_BUNDLE"),
        required=not os.environ.get("HV2_RECOVERY_BUNDLE"),
    )
    check_parser.add_argument("--model", default="all")
    check_parser.add_argument("--sample", action="store_true", help="Also run synthetic inference")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = prepare(args.catalog, args.output, args.models, asset_mode=args.asset_mode)
        print(json.dumps(dict(bundle=args.output, models=list(value["models"]))))
        return
    root, manifest = load_bundle(args.bundle)
    names = list(manifest["models"]) if args.model == "all" else [args.model]
    for name in names:
        row = manifest["models"][name]
        assets = {
            key: asset_path(root, row[key])
            for key in ("planner", "controller", "pd_binding", "controller_training_config")
        }
        value = inspect_pair(
            assets["planner"],
            assets["controller"],
            row["reference_inference"],
            row["objective"],
            sample=args.sample,
        )
        print(json.dumps(dict(model=name, verified=value), allow_nan=False))


if __name__ == "__main__":
    main()
