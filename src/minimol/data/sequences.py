"""DNA and protein sequence helpers. Alphabet is ACGTN; U is normalized to T."""

from __future__ import annotations

import hashlib

import torch

DNA_ALPHABET = "ACGTN"
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")
_COMPLEMENT_INDEX = (3, 2, 1, 0, 4)


def normalize_sequence(seq: str) -> str:
    if seq is None:
        raise ValueError("sequence is empty")
    text = str(seq).strip().upper().replace("U", "T")
    if not text:
        raise ValueError("sequence is empty")
    bad = sorted({char for char in text if char not in DNA_ALPHABET})
    if bad:
        raise ValueError(f"invalid DNA bases: {''.join(bad)}")
    return text


def reverse_complement(seq: str) -> str:
    text = normalize_sequence(seq)
    return text.translate(_COMPLEMENT)[::-1]


def sequence_hash(seq: str) -> str:
    return hashlib.sha256(normalize_sequence(seq).encode()).hexdigest()


def sample_id(*parts: str) -> str:
    raw = "\t".join(parts)
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


def gc_fraction(seq: str) -> float:
    text = normalize_sequence(seq)
    return (text.count("G") + text.count("C")) / len(text)


def n_fraction(seq: str) -> float:
    text = normalize_sequence(seq)
    return text.count("N") / len(text)


def one_hot_sequence(seq: str) -> torch.Tensor:
    text = normalize_sequence(seq)
    index = torch.tensor([DNA_ALPHABET.index(char) for char in text], dtype=torch.long)
    encoded = torch.zeros(len(DNA_ALPHABET), len(text), dtype=torch.float32)
    encoded.scatter_(0, index.unsqueeze(0), 1.0)
    return encoded


def reverse_complement_one_hot(encoded: torch.Tensor) -> torch.Tensor:
    if encoded.dim() == 2:
        return torch.flip(encoded[_COMPLEMENT_INDEX, :], dims=(-1,))
    if encoded.dim() == 3:
        return torch.flip(encoded[:, _COMPLEMENT_INDEX, :], dims=(-1,))
    raise ValueError("one-hot tensor must have shape [5, L] or [B, 5, L]")


def one_hot_protein(seq: str) -> torch.Tensor:
    text = str(seq).strip().upper()
    if not text:
        raise ValueError("protein sequence is empty")
    bad = sorted({char for char in text if char not in AA_ALPHABET})
    if bad:
        raise ValueError(f"invalid amino acids: {''.join(bad)}")
    index = torch.tensor([AA_ALPHABET.index(char) for char in text], dtype=torch.long)
    encoded = torch.zeros(len(AA_ALPHABET), len(text), dtype=torch.float32)
    encoded.scatter_(0, index.unsqueeze(0), 1.0)
    return encoded


def maybe_reverse_complement_batch(
    encoded: torch.Tensor,
    enabled: bool,
    coin: torch.Tensor | None = None,
    pad_mask: torch.Tensor | None = None,
):
    """Training-time RC augmentation. Labels are unchanged; default is off.

    Regulatory assays are often stranded. Enable this only when the assay
    definition says orientation does not matter. When a pad mask is passed,
    it is flipped with the sequence so padded coordinates stay padded.
    """

    if not enabled:
        return (encoded, pad_mask) if pad_mask is not None else encoded
    if coin is None:
        coin = torch.rand(encoded.shape[0], device=encoded.device)
    chosen = coin < 0.5
    if not bool(chosen.any()):
        return (encoded, pad_mask) if pad_mask is not None else encoded
    flipped = reverse_complement_one_hot(encoded)
    chosen_b = chosen.view(-1, 1, 1)
    encoded = torch.where(chosen_b, flipped, encoded)
    if pad_mask is None:
        return encoded
    flipped_mask = torch.flip(pad_mask, dims=(-1,))
    return encoded, torch.where(chosen.view(-1, 1), flipped_mask, pad_mask)
