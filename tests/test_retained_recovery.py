"""Retained consumers must preserve their inputs and safety command boundaries."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rise import sim, models
from rise.paths import CONFIG_ROOT
from rise.presets import gallery


def options(name, **changes):
    values = dict(
        catalog=gallery.CATALOG,
        example=name,
        seed=None,
        replan_hz=None,
        guidance_scale=None,
        blend_ticks=None,
        standing_hold=None,
        check=False,
        robot=False,
        sensor_only=False,
        output=None,
        feedback="estimated",
        seconds=1.0,
        depth_processing="vigor_legacy",
        external_camera=True,
        position_start=False,
        odometry_clock=None,
        headless=True,
        standing_hold_on_upright=False,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_all_retained_lab_recipes_have_paired_inputs_and_preserve_operator_y():
    catalog = json.loads(gallery.CATALOG.read_text())
    aliases = {name: ids[0] for name, ids in catalog["aliases"].items() if len(ids) == 1}
    assert len(catalog["runs"]) == len(aliases) == 32
    for name, run_id in aliases.items():
        row = catalog["runs"][run_id]
        assert row["model"] == "s50r_diffusion"
        for key in ("goal", "start", "robot_pose"):
            assert (gallery.CATALOG.parent / row[key]).is_file()
        command = gallery.command_for(options(name))
        assert "--actuate" not in command and "--viewer" not in command
        assert command[command.index("--recorded-start") + 1].endswith(row["start"])
        assert ("--standing-hold" in command) == row.get("standing_hold", False)
        assert command[command.index("--replan-hz") + 1] == str(row["replan_hz"])
        assert command[command.index("--blend-ticks") + 1] == str(row["blend_ticks"])
        sensors = gallery.command_for(options(name, robot=True, sensor_only=True))
        assert "--actuate" not in sensors and "--start-pose" not in sensors
        motors = gallery.command_for(options(name, robot=True))
        assert "--actuate" in motors and "--start-pose" not in motors


def test_model_selection_preserves_named_pairs_and_rejects_retired_aliases(monkeypatch, tmp_path):
    profile = yaml.safe_load((CONFIG_ROOT / "models.yaml").read_text())
    monkeypatch.setenv("HV2_RECOVERY_BUNDLE", str(tmp_path / "published"))
    for name in models.RETAINED_MODELS:
        selected = models.model_settings(profile, name)
        assert Path(selected["bundle"]).name == Path(profile["models"][name]["bundle"]).name
        assert selected["camera"]["allow_intrinsics_mismatch"] is False
    assert models.model_settings(profile, "ours")["default_backend"] == "pytorch"
    for retired in ("diffusion", "flow", "p11_diffusion", "p05_diffusion"):
        with pytest.raises(ValueError, match="Unsupported"):
            models.model_settings(profile, retired)


def test_explicit_bundle_overrides_environment_and_configured_pair(monkeypatch, tmp_path):
    profile = yaml.safe_load((CONFIG_ROOT / "models.yaml").read_text())
    environment = tmp_path / "environment"
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("HV2_RECOVERY_BUNDLE", str(environment))
    assert models.model_settings(profile, "ours")["bundle"] == str(environment)
    for name in ("ours", models.RETAINED_MODELS[0]):
        assert models.model_settings(profile, name, bundle=explicit)["bundle"] == str(explicit)


@pytest.mark.parametrize("name", models.PUBLISHED_MODELS)
def test_unbound_model_requires_a_prepared_bundle(monkeypatch, name):
    monkeypatch.delenv("HV2_RECOVERY_BUNDLE", raising=False)
    profile = yaml.safe_load((CONFIG_ROOT / "models.yaml").read_text())
    with pytest.raises(ValueError, match="--bundle PATH.*HV2_RECOVERY_BUNDLE"):
        models.model_settings(profile, name)


def test_missing_deployment_bundle_fails_before_loading_or_opening_devices(monkeypatch, capsys):
    from rise import deploy
    from motion_diffusion.inference import bundle

    monkeypatch.delenv("HV2_RECOVERY_BUNDLE", raising=False)

    def unexpected(*args, **kwargs):
        pytest.fail("Unbound deployment must fail before reading weights or opening devices")

    monkeypatch.setattr(bundle, "load_bundle", unexpected)
    monkeypatch.setattr(deploy, "hardware_settings", unexpected)
    with pytest.raises(SystemExit) as error:
        deploy.main(["--model", "ours", "--check"])
    assert error.value.code == 2
    assert "No bundle configured for ours" in capsys.readouterr().err


def test_missing_simulation_bundle_creates_no_output(monkeypatch, tmp_path):
    from rise import sim

    monkeypatch.delenv("HV2_RECOVERY_BUNDLE", raising=False)
    output = tmp_path / "not_created"
    with pytest.raises(ValueError, match="No bundle configured"):
        sim.main(["--model", "ours", "--output", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("selector", [["--gallery", "robust_29_left"], ["--preset", "supine"]])
@pytest.mark.parametrize("motor_flag", ["--robot", "--actuate"])
def test_named_simulation_rejects_motor_enable(selector, motor_flag):
    with pytest.raises(SystemExit) as error:
        sim.main([*selector, motor_flag, "--print-command"])
    assert error.value.code == 2


def test_recording_revision_check_detects_changed_or_dirty_checkout(monkeypatch):
    from hv2_common import artifacts
    from rise.review import model

    code = dict(commit="a" * 40, branch="proposal", dirty=False)
    manifest = dict(schema="hv2.model_trace.v3", code=dict(hv2=code, g1_deploy=code))
    current = dict(code)
    monkeypatch.setattr(artifacts, "git_metadata", lambda root: dict(current))
    assert model.source_changes(manifest) == []
    current["commit"] = "b" * 40
    assert model.source_changes(manifest) == ["hv2", "g1_deploy"]
    current.update(commit=code["commit"], dirty=True)
    assert model.source_changes(manifest) == ["hv2", "g1_deploy"]
    current.update(dirty=False)
    code["dirty"] = True
    assert model.source_changes(manifest) == ["hv2", "g1_deploy"]
