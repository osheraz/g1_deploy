"""Verified Git-LFS binary overlay for Unitree's Python CRC helper."""
from __future__ import annotations

import ctypes
import hashlib
import os
import platform
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lfs_object(pointer: bytes) -> tuple[str, int] | None:
    try:
        lines = pointer.decode("ascii").splitlines()
    except UnicodeDecodeError:
        return None
    if not lines or lines[0] != "version https://git-lfs.github.com/spec/v1":
        return None
    values = dict(line.split(" ", 1) for line in lines[1:] if " " in line)
    oid = values.get("oid", "")
    size = values.get("size", "")
    if not oid.startswith("sha256:") or not size.isdigit():
        raise RuntimeError("malformed Git LFS pointer for Unitree CRC library")
    return oid.removeprefix("sha256:"), int(size)


def configure_materialized_unitree_crc(
    crc_module,
    *,
    materialized_sdk_root: str | Path | None = None,
) -> Path:
    """Load-test the CRC ELF, resolving a reviewed Git LFS pointer if needed.

    The imported Python/IDL module remains the reviewed checkout. If its native
    library is an LFS pointer, only the module's library-relative ``__file__``
    anchor is redirected to a byte-identical materialized SDK tree.
    """

    machine = platform.machine()
    if machine == "x86_64":
        library_name = "crc_amd64.so"
    elif machine == "aarch64":
        library_name = "crc_aarch64.so"
    else:
        raise RuntimeError(f"unsupported Unitree CRC architecture: {machine}")

    reviewed_crc_py = Path(crc_module.__file__).resolve()
    reviewed_library = reviewed_crc_py.parent / "lib" / library_name
    if not reviewed_library.is_file():
        raise RuntimeError(f"Unitree CRC library is missing: {reviewed_library}")

    prefix = reviewed_library.read_bytes()[:256]
    if prefix.startswith(b"\x7fELF"):
        try:
            ctypes.CDLL(str(reviewed_library))
        except OSError as exc:
            raise RuntimeError(
                f"Unitree CRC ELF cannot be loaded: {reviewed_library}: {exc}"
            ) from exc
        return reviewed_library

    lfs = _lfs_object(prefix)
    if lfs is None:
        raise RuntimeError(
            f"Unitree CRC library is neither ELF nor a Git LFS pointer: "
            f"{reviewed_library}"
        )
    expected_sha256, expected_size = lfs

    if materialized_sdk_root is None:
        materialized_sdk_root = os.environ.get(
            "E_SKIN_MATERIALIZED_UNITREE_SDK",
            str(
                Path.home()
                / "GR00T-WholeBodyControl"
                / "external_dependencies"
                / "unitree_sdk2_python"
            ),
        )
    materialized_utils = (
        Path(materialized_sdk_root).expanduser().resolve()
        / "unitree_sdk2py"
        / "utils"
    )
    materialized_crc_py = materialized_utils / "crc.py"
    materialized_library = materialized_utils / "lib" / library_name
    if not materialized_crc_py.is_file() or not materialized_library.is_file():
        raise RuntimeError(
            "materialized Unitree CRC overlay is unavailable: "
            f"{materialized_library}"
        )
    if reviewed_crc_py.read_bytes() != materialized_crc_py.read_bytes():
        raise RuntimeError(
            "materialized Unitree crc.py differs from the reviewed checkout"
        )
    actual_size = materialized_library.stat().st_size
    actual_sha256 = _sha256(materialized_library)
    if actual_size != expected_size or actual_sha256 != expected_sha256:
        raise RuntimeError(
            "materialized Unitree CRC binary does not match its reviewed Git "
            f"LFS object: expected {expected_size} bytes/{expected_sha256}, got "
            f"{actual_size} bytes/{actual_sha256}"
        )
    if materialized_library.read_bytes()[:4] != b"\x7fELF":
        raise RuntimeError(
            f"materialized Unitree CRC object is not ELF: {materialized_library}"
        )
    try:
        ctypes.CDLL(str(materialized_library))
    except OSError as exc:
        raise RuntimeError(
            f"materialized Unitree CRC ELF cannot be loaded: "
            f"{materialized_library}: {exc}"
        ) from exc

    # CRC.__init__ derives its library directory from the module-global
    # ``__file__``. The class itself and every IDL definition remain those
    # already imported from the reviewed checkout.
    crc_module.__file__ = str(materialized_crc_py)
    return materialized_library
