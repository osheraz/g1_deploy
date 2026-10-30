"""Keep the SDK's exact packet packing and C CRC; remove Python word rebuilding.

The installed SDK packs the native message with struct, then reconstructs every
uint32 in Python and copies that list back to ctypes. A typed buffer supplies the
same words directly. This does not change the polynomial, coverage, packing or
CRC checks. The SDK implementation remains untouched.
"""
import ctypes
import sys

from unitree_sdk2py.utils.crc import CRC as _SDKCRC


class CRC(_SDKCRC):
    def __new__(cls):
        # Avoid the SDK singleton returning a previously created base instance.
        return object.__new__(cls)

    def _CRC__Trans(self, packed):
        if sys.byteorder!='little' or self.platform!='Linux':
            return super()._CRC__Trans(packed)
        # The final native uint32 is the CRC itself, excluded by the SDK too.
        return (ctypes.c_uint32*(len(packed)//4-1)).from_buffer_copy(packed)

    def _crc_ctypes(self, words):
        if not isinstance(words,ctypes.Array):return super()._crc_ctypes(words)
        return self.crc_lib.crc32_core(words,len(words))
