"""Recorded-input standing/tracker attribution. Never connects to the robot.

See docs/rise/recording.md for replay and diagnostics.
Every intervention retains recorded measurements and previous-issued actions.
These are pointwise sensitivity probes, NOT alternative physical rollouts.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

from rise.paths import ROOT
import numpy as np
import torch

from humanoidverse2.agents.action_vae.modules import ReconstructionHead
from hv2_common.artifacts import file_sha256
from motion_diffusion.inference.policy import load_controller
from rise.recording.depth import RecordedModelDepth
from motion_diffusion.inference.model_trace import read_trace, load_trace_manifest
from motion_diffusion.inference.standing import StandingReference


def factorial_attribution(evaluate, names):
    """Exact Shapley decomposition of one finite before/after input contrast.

    Includes all interactions by averaging all orders of substitution. This is
    an algebraic decomposition, not identification of a physical cause.
    """
    n = len(names)
    values = {
        bits: np.asarray(evaluate(bits), dtype=np.float64)
        for bits in itertools.product((0, 1), repeat=n)
    }
    contributions = {}
    for i, name in enumerate(names):
        total = np.zeros_like(values[(0,) * n])
        for bits, value in values.items():
            if bits[i]:
                continue
            k = sum(bits)
            weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            changed = list(bits)
            changed[i] = 1
            total += weight * (values[tuple(changed)] - value)
        contributions[name] = total
    difference = values[(1,) * n] - values[(0,) * n]
    if not np.allclose(sum(contributions.values()), difference, atol=1.0e-9):
        raise AssertionError("Attribution does not sum to the observed contrast")
    return values, contributions


def summarize_targets(targets):
    delta = np.diff(np.asarray(targets), axis=0)
    return dict(
        max_adjacent_target_change_rad=float(np.abs(delta).max()),
        p95_adjacent_target_change_rad=float(np.quantile(np.abs(delta).max(-1), 0.95)),
    )


@torch.no_grad()
def investigate(session, events, interval, output, device="cuda"):
    root = Path(session).resolve()
    directory = root / "model_trace"
    manifest = load_trace_manifest(directory)
    config = dict(manifest["config"])
    for key, spec in manifest["weights"].items():
        if file_sha256(spec["path"]) != spec["sha256"]:
            raise ValueError(f'Changed checkpoint: {spec["path"]}')
        config[key] = spec["path"]
    # Retain captured TensorRT settings for the baseline, then test native FP32.
    config["tracker_device"] = device
    controller = load_controller(config)
    payload = torch.load(config["controller_checkpoint"], map_location="cpu", weights_only=True)
    standing = StandingReference(payload, controller, frame="local")
    av = payload["action_vae"]
    contract = av["model_contract"]
    if contract["history"]["row_dim"] != 93 or contract["history"]["frames"] != 10:
        raise ValueError("This field attribution expects the recorded row93 history contract")
    recon = (
        ReconstructionHead(
            contract["proprio_dim"],
            av["latent_dim"],
            av["num_tokens"],
            hidden_dims=contract["recon_hidden_dims"],
            proprio_conditioned=contract["recon_proprio_conditioned"],
            activation=contract["recon_activation"],
        )
        .to(device)
        .eval()
    )
    recon.load_state_dict(av["recon_state_dict"], strict=True)
    if recon.proprio_conditioned:
        raise ValueError("Latent-only intent diagnosis needs the unconditioned reconstruction head")

    # Exhaust both streams to verify their digest/count/complete checks.
    records = list(read_trace(directory / "control"))
    plans = list(read_trace(directory / "planner"))
    rows = {
        r["control_tick"]: r
        for r in records
        if r["kind"] == "tracker" and not r["preview"] and r["outcome"] == "issued"
    }
    depth_archive = RecordedModelDepth(root / "sensors/depth")
    depth_cache = {}

    def tensor(value):
        return torch.as_tensor(value, device=device)

    def depth(row):
        key = row["control_tick"]
        if key not in depth_cache:
            view = row["depth"]
            if view.get("storage") in ("metric_capture_refs_v1", "image_capture_refs"):
                view = depth_archive.restore(view, controller.depth_config)
            depth_cache[key] = {
                k: tensor(v) for k, v in view.items() if k in ("frames", "metadata", "available")
            }
        return depth_cache[key]

    scale = np.asarray(config["pd"]["training_pd"]["action_scale"])
    default = controller.default_joint_pos.cpu().numpy()

    def target(history, code, d):
        action = controller(
            tensor(history),
            tensor(code),
            depth_frames=d["frames"],
            depth_metadata=d.get("metadata"),
            depth_available=d.get("available"),
        )
        return default + scale * action.cpu().numpy()[0]

    def reconstruct(code):
        return (
            (
                recon(tensor(code)) * standing.normalizer.var.sqrt().clamp_min(1.0e-4)
                + standing.normalizer.mean
            )
            .cpu()
            .numpy()[0]
        )

    # A fully settled reference: five valid stationary states, zero translation,
    # upright orientation in current heading. No random/zero latent substitute.
    zero = torch.zeros(1, 3, device=device)
    heading = torch.zeros(1, device=device)
    standing.activate(
        0.0,
        controller.default_joint_pos[None],
        torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device),
        zero,
        heading,
    )
    stand_code = standing(1.0, zero, heading)
    selected = sorted(
        k
        for k, r in rows.items()
        if interval[0] <= r["command_shaping"]["elapsed_s"] <= interval[1]
    )
    if len(selected) < 2 or np.any(np.diff(selected) != 1):
        raise ValueError("Choose a contiguous interval with >=2 issued tracker calls")
    report = dict(
        schema="hv2.standing_cause.v1",
        session=str(root),
        scope="Pointwise recorded-input probes, not closed-loop rollouts or a physical causality proof",
        controller_sha256=controller.checkpoint_sha256,
        planner_sha256=manifest["weights"]["planner_checkpoint"]["sha256"],
        interval_s=interval,
        backend=config.get("controller_engine"),
        histories="Recorded actual history; no reset, no counterfactual-action feedback",
        events=[],
    )
    report["standing_encoder_reconstruction"] = dict(
        q_error_max_rad=float(np.abs(reconstruct(stand_code)[:, :29]).max()),
        dq_max_rad_s=float(np.abs(reconstruct(stand_code)[:, 29:58]).max()),
    )
    history_error = 0.0
    checked_history_rows = 0
    for k in selected:
        h = rows[k]["history"].reshape(10, 93)
        for i, t in enumerate(range(k - 9, k + 1)):
            if t not in rows or t - 1 not in rows:
                continue
            o = rows[t]["observation"]
            expected = np.concatenate(
                (
                    o["q"] - default,
                    o["dq"],
                    o["gyro_body"],
                    o["gravity_body"],
                    rows[t - 1]["issued_request"],
                )
            )
            history_error = max(history_error, float(np.abs(h[i] - expected).max()))
            checked_history_rows += 1
    if not checked_history_rows or history_error > 1.0e-6:
        raise ValueError(
            f"History audit failed: {history_error}; checked {checked_history_rows} rows"
        )
    report["history_audit"] = dict(
        checked_rows=checked_history_rows,
        max_error=history_error,
        contract="Chronological measurement rows with preceding actually issued action",
    )
    current, fixed = [], []
    replay_error = []
    for k in selected:
        r = rows[k]
        d = depth(r)
        actual = target(r["history"], r["native_code"], d)
        current.append(actual)
        fixed.append(target(r["history"], stand_code, d))
        replay_error.append(float(np.abs(actual - r["issued_target"]).max()))
    report["window"] = dict(
        n=len(selected),
        actual=summarize_targets(current),
        standing=summarize_targets(fixed),
        replay_max_error_rad=max(replay_error),
    )
    if max(replay_error) > 1.0e-4:
        raise ValueError(
            f"Baseline target replay failed: {max(replay_error)} rad; do not interpret swaps"
        )

    names = controller.contract["dof_names"]
    fields = dict(
        q=slice(0, 29),
        dq=slice(29, 58),
        gyro=slice(58, 61),
        gravity=slice(61, 64),
        prior_issued_action=slice(64, 93),
    )
    for k in events:
        a, b = rows[k - 1], rows[k]
        if a.get("reset_history") or b.get("reset_history"):
            raise ValueError("Event crosses a reset; analyze separately")
        histories = [a["history"], b["history"]]
        codes = [a["native_code"], b["native_code"]]
        depths = [depth(a), depth(b)]
        values, contribution = factorial_attribution(
            lambda bits: target(histories[bits[0]], codes[bits[1]], depths[bits[2]]),
            ("history", "native_code", "depth"),
        )
        delta = values[(1, 1, 1)] - values[(0, 0, 0)]
        j = int(np.abs(delta).argmax())
        h0, h1 = (h.reshape(1, 10, 93) for h in histories)

        def field_swap(bits):
            h = h0.copy()
            for enabled, region in zip(bits, fields.values()):
                if enabled:
                    h[..., region] = h1[..., region]
            return target(h.reshape(1, -1), codes[0], depths[0])

        _, field_contribution = factorial_attribution(field_swap, tuple(fields))

        def lag_swap(bits):
            h = h0.copy()
            for lag, enabled in enumerate(bits):
                if enabled:
                    h[:, -1 - lag, 64:] = h1[:, -1 - lag, 64:]
            return target(h.reshape(1, -1), codes[0], depths[0])

        _, lag_contribution = factorial_attribution(
            lag_swap, tuple(f"prior_command_{lag+1}_ticks_ago" for lag in range(10))
        )
        latest_joint_swaps = []
        for source_j, source_name in enumerate(names):
            h = h0.copy()
            h[:, -1, 64 + source_j] = h1[:, -1, 64 + source_j]
            response = target(h.reshape(1, -1), codes[0], depths[0])[j] - values[(0, 0, 0)][j]
            latest_joint_swaps.append(
                dict(
                    joint=source_name,
                    preceding_target_change_rad=float(
                        (h1[0, -1, 64 + source_j] - h0[0, -1, 64 + source_j]) * scale[source_j]
                    ),
                    output_ankle_change_rad=float(response),
                )
            )
        latest_joint_swaps.sort(key=lambda r: abs(r["output_ankle_change_rad"]), reverse=True)
        stand_pair = [target(h, stand_code, d) for h, d in zip(histories, depths)]
        frozen_pair = [target(h, codes[0], d) for h, d in zip(histories, depths)]
        local_gains = {}
        # Perturb one previous issued target by a specified number of radians,
        # converting to native normalized action with the pinned training scale.
        for i, label in enumerate(("before", "after")):
            for mode, code in (("recorded_code", codes[i]), ("standing_code", stand_code)):
                gains = {}
                for epsilon in (0.001, 0.01):
                    plus = histories[i].reshape(1, 10, 93).copy()
                    minus = plus.copy()
                    plus[:, -1, 64 + j] += epsilon / scale[j]
                    minus[:, -1, 64 + j] -= epsilon / scale[j]
                    gain = (
                        target(plus.reshape(1, -1), code, depths[i])[j]
                        - target(minus.reshape(1, -1), code, depths[i])[j]
                    ) / (2 * epsilon)
                    gains[str(epsilon)] = float(gain)
                local_gains[f"{label}_{mode}"] = gains
        fp32 = []
        saved_backend = controller.inference_backend
        saved_depth_backend = controller.depth_encoder.inference_backend
        saved_precision = controller.precision
        try:
            controller.inference_backend = None
            controller.precision = "fp32"
            controller.depth_encoder.inference_backend = None
            for r in (a, b):
                fp32.append(target(r["history"], r["native_code"], depth(r)))
        finally:
            controller.inference_backend = saved_backend
            controller.precision = saved_precision
            controller.depth_encoder.inference_backend = saved_depth_backend
        decoded = [reconstruct(z) for z in codes]
        # q/dq are anchor-independent. Do not compare XYZ from different anchors.
        plan = next(
            (
                p
                for p in plans
                if not p.get("error")
                and not p["request"].get("preview")
                and int(p["request"]["origin_tick"]) == int(b["execution"]["origin_tick"][0])
                and int(p["request"]["generation"]) == int(b["execution"]["generation"][0])
            ),
            None,
        )
        plan_info = None
        if plan is not None:
            state = plan["result"]["prediction"]["state"][0]
            cursor = int(b["execution"]["cursor"][0])
            offsets = [int(round(t / 0.02)) for t in av["future_times_s"]]
            # S starts at t+1; z starts at t. Compare positive encoder offsets.
            pairs = [
                (i, cursor + offset - 1)
                for i, offset in enumerate(offsets)
                if offset > 0 and cursor + offset - 1 < len(state)
            ]
            errs = np.array([decoded[1][i, :29] - state[t, :29] for i, t in pairs])
            plan_info = dict(
                origin_tick=int(b["execution"]["origin_tick"][0]),
                cursor=cursor,
                request_guidance_enabled=plan["request"].get("guidance_enabled", False),
                q_reconstruction_vs_predicted_state_rmse_rad=float(np.sqrt(np.mean(errs**2))),
                comparison="q only, positive 0.1/0.2/0.3/0.4 s offsets; S[t+1], z[t]",
            )
        report["events"].append(
            dict(
                tick=k,
                time_s=b["command_shaping"]["elapsed_s"],
                joint=names[j],
                joint_index=j,
                same_plan=bool(
                    np.array_equal(a["execution"]["origin_tick"], b["execution"]["origin_tick"])
                ),
                actual_jump_rad=float(delta[j]),
                factorial_targets_rad={
                    "".join(map(str, bits)): float(v[j]) for bits, v in values.items()
                },
                input_shapley_rad={name: float(v[j]) for name, v in contribution.items()},
                history_field_shapley_rad={
                    name: float(v[j]) for name, v in field_contribution.items()
                },
                action_history_lag_shapley_rad={
                    name: float(v[j]) for name, v in lag_contribution.items()
                },
                lag_scope="Only prior-action rows changed; measured history, code and depth fixed at earlier tick",
                latest_prior_action_single_joint_swaps=latest_joint_swaps,
                joint_swap_scope="Each joint changed alone in newest prior-action row; NOT additive",
                field_scope="Earlier code and depth fixed; field interactions distributed by Shapley",
                fixed_previous_code_jump_rad=float(frozen_pair[1][j] - frozen_pair[0][j]),
                fixed_standing_code_jump_rad=float(stand_pair[1][j] - stand_pair[0][j]),
                fixed_standing_targets_rad=[float(p[j]) for p in stand_pair],
                previous_ankle_command_local_gain=local_gains,
                gain_scope="rad target response per rad perturbation of preceding issued ankle target; "
                "all other inputs fixed, NOT full robot closed-loop gain",
                fp32_target_error_rad=max(
                    float(np.abs(fp32[i] - values[(i, i, i)]).max()) for i in (0, 1)
                ),
                reconstructed_q_change_max_rad=float(
                    np.abs(decoded[1][:, :29] - decoded[0][:, :29]).max()
                ),
                reconstructed_ankle_q_change_max_rad=float(
                    np.abs(decoded[1][:, j] - decoded[0][:, j]).max()
                ),
                plan=plan_info,
            )
        )

    # Save compact, directly inspectable vectors for all window probes.
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "probes.npz",
        ticks=selected,
        time_s=[rows[k]["command_shaping"]["elapsed_s"] for k in selected],
        actual_target=current,
        standing_target=fixed,
        measured_q=[rows[k]["observation"]["q"] for k in selected],
        measured_dq=[rows[k]["observation"]["dq"] for k in selected],
        standing_code=stand_code.cpu().numpy(),
    )
    report["diagnostic_source_sha256"] = file_sha256(__file__)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("session", type=Path)
    p.add_argument(
        "--events",
        type=int,
        nargs="+",
        required=True,
        help="Later control tick of each adjacent pair",
    )
    p.add_argument("--interval", type=float, nargs=2, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_num_threads(1)
    investigate(args.session, args.events, args.interval, args.output, args.device)


if __name__ == "__main__":
    main()
