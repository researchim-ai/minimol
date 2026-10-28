"""Uncertainty helpers that stay separate from typed probabilities."""

from __future__ import annotations

import torch
import torch.nn as nn

from minimol.training.losses import gaussian_nll


class HeteroscedasticHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.mean = nn.Linear(hidden, 1)
        self.log_var = nn.Linear(hidden, 1)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mean(hidden).squeeze(-1), self.log_var(hidden).squeeze(-1)

    def loss(self, hidden: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mean, log_var = self.forward(hidden)
        return gaussian_nll(mean, log_var, target, mask)


def ensemble_variance(predictions) -> dict:
    """Variance across seeds is an epistemic proxy, not a calibrated interval."""

    import numpy as np

    stack = np.stack([np.asarray(item, dtype=np.float64) for item in predictions], axis=0)
    return {
        "mean": stack.mean(axis=0),
        "variance": stack.var(axis=0),
        "n_seeds": int(stack.shape[0]),
        "calibrated": False,
    }
