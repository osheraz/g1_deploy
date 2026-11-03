"""Resolve a SONIC encoder, decoder and observation profile as one matched set."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

DEPLOY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = DEPLOY_ROOT / "configs/sonic_models.yaml"


def _profiles() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text())


def model_profile_names() -> tuple[str, ...]:
    return tuple(_profiles())


def canonical_model_profile(name: str) -> str:
    token = str(name).strip().lower()
    for canonical, profile in _profiles().items():
        if token in (canonical, *profile["aliases"]):
            return canonical
    raise ValueError(
        f"unknown SONIC model profile {name!r}; choose {', '.join(model_profile_names())}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_python_model_profile(name: str) -> dict:
    canonical = canonical_model_profile(name)
    profile = _profiles()[canonical]
    directory = (DEPLOY_ROOT / profile["directory"]).resolve()
    resolved = dict(
        name=canonical, variant=profile["variant"], tensor_contract=dict(profile["tensor_contract"])
    )
    for role, key, filename in (
        ("encoder", "encoder", "model_encoder.onnx"),
        ("decoder", "decoder", "model_decoder.onnx"),
        ("observation", "observation_config", "observation_config.yaml"),
    ):
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"SONIC {canonical} {role} is not installed: {path}")
        actual = _sha256(path)
        expected = profile["sha256"][role]
        if actual != expected:
            raise ValueError(
                f"SONIC {canonical} {role} hash is {actual}, expected {expected}: {path}"
            )
        resolved[key] = path
    return resolved
