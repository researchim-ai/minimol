"""Carbon-A adapter.

The loader refuses trust_remote_code until a human pins a revision and confirms
the remote code was reviewed. Local smoke checks only the coordinate contract.
"""

from __future__ import annotations

import numpy as np

from minimol.data.sequences import normalize_sequence
from minimol.errors import MinimolError, RemoteCodeNotReviewed
from minimol.teachers.alignment import (
    expand_token_values_to_bases,
    id_to_sixmer,
    tokenize_sixmers,
)

CARBON_A_REPO = "HuggingFaceBio/Carbon-A-1.2B"


def assert_carbon_a_loadable(revision: str, reviewed_remote_code: bool) -> None:
    if not reviewed_remote_code:
        raise RemoteCodeNotReviewed(
            "Refusing trust_remote_code until the pinned Carbon-A Python is reviewed."
        )
    text = "" if revision is None else str(revision)
    if not text or "UNVERIFIED" in text or "REPLACE" in text:
        raise MinimolError("Carbon-A revision is not a verified commit.")


def load_carbon_a(revision: str, reviewed_remote_code: bool = False):
    assert_carbon_a_loadable(revision, reviewed_remote_code)
    try:
        from transformers import AutoModelForTokenClassification
    except ImportError as exc:
        raise MinimolError("transformers is not installed") from exc
    return AutoModelForTokenClassification.from_pretrained(
        CARBON_A_REPO,
        trust_remote_code=True,
        revision=revision,
    )


def local_coordinate_smoke(sequence: str) -> dict:
    """Deterministic stand-in used to test 6-mer alignment. Not Carbon-A output."""

    text = normalize_sequence(sequence)
    tokens = tokenize_sixmers(text)
    logits = []
    for token in tokens:
        kmer = id_to_sixmer(token)
        plus = 2.0 if "ATG" in kmer else -2.0
        minus = 2.0 if "CTA" in kmer else -2.0
        logits.append((plus, minus))
    array = np.asarray(logits, dtype=np.float64) if logits else np.zeros((0, 2))
    bases, mask = expand_token_values_to_bases(array, len(text))
    return {
        "length": len(text),
        "n_tokens": len(tokens),
        "expansion": "repeat",
        "verified_against_upstream": False,
        "upstream_weights": "not_loaded",
        "valid_bases": int(mask.sum()),
        "base_logits": bases,
        "valid_mask": mask,
    }
