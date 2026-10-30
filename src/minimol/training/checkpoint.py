"""Atomic local checkpoints and the RNG state needed for epoch-boundary resume."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def rng_state(include_cuda: bool = True) -> dict:
    return {
        'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if include_cuda and torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None:
        if not torch.cuda.is_available():
            raise ValueError('CUDA RNG state requires a CUDA resume device')
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])


def save_checkpoint(path, model, optimizer, metadata: dict, training_state: dict | None = None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = rng_state(include_cuda=any(parameter.is_cuda for parameter in model.parameters()))
    payload = {
        'model_state': model.state_dict(),
        'optimizer_state': None if optimizer is None else optimizer.state_dict(),
        'metadata': metadata, 'torch_rng': state['torch'], 'rng_state': state,
        'training_state': training_state,
    }
    fd, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, model, optimizer=None, map_location: str = 'cpu'):
    blob = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(blob['model_state'])
    if optimizer is not None and blob.get('optimizer_state') is not None:
        optimizer.load_state_dict(blob['optimizer_state'])
    return blob
