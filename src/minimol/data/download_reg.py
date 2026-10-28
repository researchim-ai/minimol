"""Pinned Malinois/MPRA download. Refuses unreviewed revisions and over-budget files."""

from __future__ import annotations

from pathlib import Path

from minimol.data.download import assert_download_allowed, huggingface_download
from minimol.errors import MinimolError

MALINOIS_REPO = "HuggingFaceBio/malinois-mpra-regression"


def download_reg_dataset(
    revision: str,
    dest: Path,
    max_bytes: int,
    decision: str,
    license_name: str,
    reviewed_card: bool = False,
) -> list[str]:
    if not reviewed_card:
        raise MinimolError("Refusing the MPRA download until the dataset card and license are reviewed.")
    assert_download_allowed(decision, revision, license_name)
    return huggingface_download(
        MALINOIS_REPO,
        revision,
        dest,
        max_bytes,
        decision,
        license_name,
    )
