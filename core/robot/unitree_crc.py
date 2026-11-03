"""Unitree CRC with the SDK's packet format and a zero-copy word buffer."""

import ctypes
import platform
from pathlib import Path
import sys

from unitree_sdk2py.utils.crc import CRC as _SDKCRC


def check_crc_library(crc_module):
    """Check the installed SDK; incomplete LFS downloads need fixing at installation."""
    machine = platform.machine()
    names = {"x86_64": "crc_amd64.so", "aarch64": "crc_aarch64.so"}
    if machine not in names:
        raise RuntimeError(f"Unsupported Unitree CRC architecture: {machine}")
    library = Path(crc_module.__file__).resolve().parent / "lib" / names[machine]
    if not library.is_file():
        raise RuntimeError(f"Unitree CRC library is missing: {library}")
    with library.open("rb") as stream:
        prefix = stream.read(128)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/"):
        raise RuntimeError(
            f"Unitree SDK has an unmaterialized LFS object: {library}. Run git lfs pull in the SDK checkout."
        )
    if not prefix.startswith(b"\x7fELF"):
        raise RuntimeError(f"Invalid Unitree CRC library: {library}")
    ctypes.CDLL(str(library))
    return library


class CRC(_SDKCRC):
    def __new__(cls):
        # Avoid the SDK singleton returning a previously created base instance.
        return object.__new__(cls)

    def _CRC__Trans(self, packed):
        if sys.byteorder != "little" or self.platform != "Linux":
            return super()._CRC__Trans(packed)
        # The final native uint32 is the CRC itself, excluded by the SDK too.
        return (ctypes.c_uint32 * (len(packed) // 4 - 1)).from_buffer_copy(packed)

    def _crc_ctypes(self, words):
        if not isinstance(words, ctypes.Array):
            return super()._crc_ctypes(words)
        return self.crc_lib.crc32_core(words, len(words))
