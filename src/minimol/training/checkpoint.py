"""Checkpoint payload. Files are trusted local artifacts, not remote code."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, metadata: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": None if optimizer is None else optimizer.state_dict(),
            "metadata": metadata,
            "torch_rng": torch.get_rng_state(),
        },
        destination,
    )


def load_checkpoint(path, model, optimizer=None, map_location: str = "cpu"):
    blob = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(blob["model_state"])
    if optimizer is not None and blob.get("optimizer_state") is not None:
        optimizer.load_state_dict(blob["optimizer_state"])
    return blob
