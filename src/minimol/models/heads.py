"""Typed decision heads.

choice: softmax over a fully specified option set.
noul: Bernoulli probability of an explicitly defined event.
score: ordinal bins with fixed boundaries, not an arbitrary mark out of ten.
regression: a measured quantity, never reported as a probability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChoiceHead(nn.Module):
    def __init__(self, hidden: int, n_options: int):
        super().__init__()
        self.logit = nn.Linear(hidden, n_options)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.logit(hidden)

    def distribution(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(hidden), dim=-1)


class NoulHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.logit = nn.Linear(hidden, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.logit(hidden).squeeze(-1)


class RegressionHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.mean = nn.Linear(hidden, 1)
        self.log_var = nn.Linear(hidden, 1)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mean(hidden).squeeze(-1), self.log_var(hidden).squeeze(-1)


class OrdinalHead(nn.Module):
    """CORAL-style cumulative logits. Thresholds are constrained to increase."""

    def __init__(self, hidden: int, n_classes: int):
        super().__init__()
        if n_classes < 2:
            raise ValueError("ordinal score needs at least two bins")
        self.n_classes = n_classes
        self.score = nn.Linear(hidden, 1)
        self.raw_thresholds = nn.Parameter(torch.zeros(n_classes - 1))

    def thresholds(self) -> torch.Tensor:
        return torch.cumsum(F.softplus(self.raw_thresholds), dim=0)

    def cumulative_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.score(hidden) - self.thresholds()

    def class_probabilities(self, hidden: torch.Tensor) -> torch.Tensor:
        greater = torch.sigmoid(self.cumulative_logits(hidden))
        ones = torch.ones(hidden.shape[0], 1, device=hidden.device, dtype=hidden.dtype)
        zeros = torch.zeros_like(ones)
        upper = torch.cat([ones, greater], dim=1)
        lower = torch.cat([greater, zeros], dim=1)
        probs = (upper - lower).clamp(min=0)
        return probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)


def ordinal_targets(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    levels = torch.arange(n_classes - 1, device=labels.device)
    return (labels.unsqueeze(1) > levels).to(dtype=torch.float32)


def ordinal_loss(logits: torch.Tensor, labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    target = ordinal_targets(labels, n_classes)
    return F.binary_cross_entropy_with_logits(logits, target)


class PairwiseRanker(nn.Module):
    """Scores whether candidate A should rank above candidate B."""

    def __init__(self, fp_dim: int = 2048, hidden: int = 640, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(fp_dim * 3, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(depth):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        self.fp_dim = fp_dim

    def forward(self, fp_a: torch.Tensor, fp_b: torch.Tensor) -> torch.Tensor:
        features = torch.cat([fp_a, fp_b, fp_a - fp_b], dim=-1)
        return self.net(features).squeeze(-1)
