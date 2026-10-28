"""Compact array codecs for CDS labels. Parquet stores bytes, not JSON arrays."""

from __future__ import annotations

import numpy as np


def pack_bits(values) -> bytes:
    array = np.asarray(values, dtype=np.uint8).reshape(-1)
    return np.packbits(array).tobytes()


def unpack_bits(blob: bytes, length: int) -> np.ndarray:
    packed = np.frombuffer(blob, dtype=np.uint8)
    bits = np.unpackbits(packed, count=int(length))
    return bits.astype(np.float32)


def pack_float16(values) -> bytes:
    array = np.asarray(values, dtype=np.float16).reshape(-1)
    return array.tobytes()


def unpack_float16(blob: bytes, length: int) -> np.ndarray:
    array = np.frombuffer(blob, dtype=np.float16, count=int(length))
    return np.array(array, dtype=np.float32, copy=True)
