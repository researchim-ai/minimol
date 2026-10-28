"""Non-overlapping 6-mer coordinates.

Carbon-A's published card uses non-overlapping 6-mers. This module does not
assume that token logits are already per-base. `repeat` is an explicit,
tested expansion: each token value is copied onto its six bases, and a tail
shorter than 6 is masked. It stays unverified against the upstream checkpoint
until a pinned smoke test says otherwise.
"""

from __future__ import annotations

import numpy as np

from minimol.errors import AlignmentError

VOCAB = 4**6
UNK = VOCAB


def sixmer_token_id(kmer: str) -> int:
    if len(kmer) != 6 or any(char not in "ACGT" for char in kmer):
        return UNK
    acc = 0
    for char in kmer:
        acc = acc * 4 + "ACGT".index(char)
    return acc


def id_to_sixmer(index: int) -> str:
    if index < 0 or index > UNK:
        raise AlignmentError(f"token id out of range: {index}")
    if index == UNK:
        return "NNNNNN"
    chars = []
    value = index
    for _ in range(6):
        chars.append("ACGT"[value % 4])
        value //= 4
    return "".join(reversed(chars))


def tokenize_sixmers(sequence: str) -> list[int]:
    usable = len(sequence) // 6
    return [sixmer_token_id(sequence[index * 6 : (index + 1) * 6]) for index in range(usable)]


def expand_token_values_to_bases(token_values, length: int, mode: str = "repeat"):
    if mode != "repeat":
        raise AlignmentError(f"unsupported expansion {mode}")
    values = np.asarray(token_values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    expected = length // 6
    if values.shape[0] != expected:
        raise AlignmentError(f"expected {expected} tokens for length {length}, got {values.shape[0]}")
    bases = np.full((length, values.shape[1]), np.nan, dtype=np.float64)
    mask = np.zeros(length, dtype=bool)
    for index in range(expected):
        start = index * 6
        bases[start : start + 6] = values[index]
        mask[start : start + 6] = True
    return bases, mask


def assert_base_alignment(sequence_length: int, n_tokens: int, base_values, mask) -> None:
    expected = sequence_length // 6
    if n_tokens != expected:
        raise AlignmentError("token count does not match non-overlapping 6-mers")
    if len(base_values) != sequence_length or len(mask) != sequence_length:
        raise AlignmentError("base axis does not match the sequence")
    if int(mask.sum()) != expected * 6:
        raise AlignmentError("valid-base count does not match complete 6-mers")
