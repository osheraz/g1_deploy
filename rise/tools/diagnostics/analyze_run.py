"""Analyze a saved hardware run without connecting to sensors or running models."""

import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

from rise.paths import ROOT

sys.path.insert(0, str(ROOT))


def analyze(session):
    from humanoidverse2.observations.streams import motor_indices

    with gzip.open(session / "telemetry.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    contract = next(r for r in rows if r.get("event") == "session_contract")
    control = [r for r in rows if r.get("event") == "control"]
    names = contract["planner_contract"]["dof_names"]
    pd = contract["deployment_config"]["pd"]["training_pd"]
    t = np.array([r["monotonic_s"] - control[0]["monotonic_s"] for r in control])
    q = np.array([r["q"] for r in control])
    dq = np.array([r["dq"] for r in control])
    issued = np.array([r["issued_request"] is not None for r in control])
    first = int(np.flatnonzero(issued)[0])
    requests = np.array(
        [
            r["issued_request"] if r["issued_request"] is not None else r["previous_issued_request"]
            for r in control
        ]
    )
    target = (
        np.array(contract["planner_contract"]["default_joint_pos"])
        + np.array(pd["action_scale"]) * requests
    )
    delta = target[first] - q[first]
    early = t < 0.5
    rank = np.argsort(np.max(abs(dq[early]), axis=0))[::-1]
    joints = []
    for j in rank[:8]:
        vel = dq[early, j]
        flips = (vel[1:] * vel[:-1] < 0) & (abs(vel[1:]) > 1) & (abs(vel[:-1]) > 1)
        joints.append(
            dict(
                joint=names[j],
                speed_peak_rad_s=float(abs(vel).max()),
                velocity_reversals_above_1_rad_s=int(flips.sum()),
                first_target_delta_rad=float(delta[j]),
                target_range_rad=[float(target[early, j].min()), float(target[early, j].max())],
                measured_range_rad=[float(q[early, j].min()), float(q[early, j].max())],
            )
        )
    pairing = None
    if (session / "sensors/raw/sensors.npz").is_file():
        with np.load(session / "sensors/raw/sensors.npz") as raw:
            # DDS can deliver distinct samples bearing the same millisecond tick.
            # The policy retains the first sample at a given tick, as does this check.
            lookup = {}
            for i, tick in enumerate(raw["state_tick_ms"]):
                lookup.setdefault(int(tick), i)
            ids = np.array([lookup.get(int(r["observation_sequence"]), -1) for r in control])
            good = ids >= 0
            rq = raw["q"][ids[good]][:, motor_indices(names)]
            pairing = dict(
                matched_ticks=int(good.sum()),
                max_joint_difference_rad=float(abs(rq - q[good]).max()),
            )
    out = session / "review/startup_analysis"
    out.mkdir(parents=True, exist_ok=True)
    health = [r for r in rows if r.get("event") == "health"]
    last = health[-1] if health else {}
    timing = {
        key: last.get(key)
        for key in (
            "timing_warning_ticks",
            "publication_late_ticks",
            "timing_hold_ticks",
            "rejected_stale_robot_packets",
            "command_limit_violations",
        )
    }
    # Targets are requested PD setpoints, not measured joint positions or torques.
    limits = contract["deployment_config"]["pd"]
    outside = np.maximum(
        np.maximum(
            np.array(limits["lower_joint_limits"]) - target,
            target - np.array(limits["upper_joint_limits"]),
        ),
        0.0,
    )
    result = dict(
        session=str(session),
        duration_s=float(t[-1]),
        first_learned_command_s=float(t[first]),
        first_target_max_delta_rad=float(abs(delta).max()),
        first_target_max_joint=names[int(abs(delta).argmax())],
        peak_issued_normalized_request=float(abs(requests[issued]).max()),
        early_target_outside_mechanical_limits_max_rad=float(outside[early].max()),
        first_half_second=joints,
        timing=timing,
        recorded_joint_order_check=pairing,
        depth_recorded=(session / "sensors/depth/frames.jsonl").is_file(),
        live_xyz_recorded=any(r.get("pelvis_odometry") is not None for r in control),
        normalization="requested joint target = saved default + saved per-joint action_scale * issued_request",
        limitation=(
            "Use replay_model.py for exact model-output verification; replay does not reproduce the physical plant."
            if (session / "model_trace/manifest.json").is_file()
            else "No exact model trace: telemetry alone cannot reproduce all tracker/planner inputs."
        ),
    )
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for ax, j in zip(
        axes[:3],
        [names.index(n) for n in ("left_wrist_roll_joint", "left_elbow_joint", "left_knee_joint")],
    ):
        ax.plot(t, target[:, j], label="requested PD target", lw=1)
        ax.plot(t, q[:, j], label="measured joint", lw=1.4)
        ax.axhline(limits["lower_joint_limits"][j], color="gray", ls=":", lw=0.7)
        ax.axhline(limits["upper_joint_limits"][j], color="gray", ls=":", lw=0.7)
        ax.set_ylabel(names[j].replace("_joint", "").replace("_", " ") + "\nrad")
        ax.legend(loc="upper right")
    axes[-1].plot(t, [r["control_wall_s"] * 1000 for r in control], label="control computation")
    axes[-1].axhline(20, color="red", ls="--", label="20 ms control period")
    axes[-1].set(xlabel="Seconds since first control tick", ylabel="ms")
    axes[-1].legend()
    for ax in axes:
        ax.axvline(t[first], color="black", ls="--", lw=0.7)
        ax.grid(alpha=0.2)
        ax.set_xlim(0, 2.0)
    fig.suptitle("Measured startup oscillation and requested targets — " + session.name)
    fig.tight_layout()
    fig.savefig(out / "startup.png", dpi=150)
    plt.close(fig)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("session", type=Path)
    args = p.parse_args()
    analyze(args.session.expanduser().resolve())
