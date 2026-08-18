"""Python view of the repository-wide SONIC deployment profile registry."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "config/sonic_deployment_profiles.json"


def _manifest() -> dict:
    with MANIFEST_PATH.open() as stream:
        value = json.load(stream)
    if value.get("schema") != "e_skin.sonic_deployment_profiles.v1":
        raise ValueError(f"unsupported SONIC deployment manifest: {value.get('schema')!r}")
    return value


def model_profile_names() -> tuple[str, ...]:
    return tuple(_manifest()["models"])


def canonical_model_profile(name: str) -> str:
    token = str(name).strip().lower()
    for canonical, record in _manifest()["models"].items():
        aliases = {str(value).lower() for value in record.get("aliases", [])}
        if token == canonical.lower() or token in aliases:
            return canonical
    raise ValueError(
        f"unknown SONIC model profile {name!r}; choose {', '.join(model_profile_names())}"
    )


def _expand(value: str) -> Path:
    return Path(
        value.format(repo=REPO_ROOT, user=os.environ.get("USER", "unknown"))
    ).expanduser()


def _first_file(values: list[str], label: str) -> Path:
    candidates = [_expand(value) for value in values]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"SONIC {label} is not installed; tried " + ", ".join(map(str, candidates))
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_python_model_profile(name: str) -> dict:
    canonical = canonical_model_profile(name)
    record = _manifest()["models"][canonical]
    resolved = {
        "name": canonical,
        "variant": record["python_variant"],
        "encoder": _first_file(record["encoder_candidates"], "encoder"),
        "decoder": _first_file(record["decoder_candidates"], "decoder"),
        "observation_config": _first_file(
            record["observation_candidates"], "observation config"
        ),
        "tensor_contract": dict(record["tensor_contract"]),
    }
    for role, key in (
        ("encoder", "encoder"),
        ("decoder", "decoder"),
        ("observation", "observation_config"),
    ):
        actual = _sha256(resolved[key])
        expected = record["sha256"][role]
        if actual != expected:
            raise ValueError(
                f"SONIC {canonical} {role} hash is {actual}, expected "
                f"{expected}: {resolved[key]}"
            )
    return resolved
