"""Small, query-conditioned distribution head over a frozen DNA encoder.

The learned conditional activity distribution is trained on typed answers.
Threshold and ordinal outputs share a monotone logistic CDF. Choices share
candidate scores, enforcing permutation equivariance and pair antisymmetry.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RegDecisionHead(nn.Module):
    def __init__(self, feature_dim: int, n_assays: int = 3, hidden: int = 128):
        super().__init__()
        self.feature_dim, self.n_assays, self.hidden = feature_dim, n_assays, hidden
        self.register_buffer('feature_mean', torch.zeros(feature_dim))
        self.register_buffer('feature_std', torch.ones(feature_dim))
        self.project = nn.Sequential(nn.Linear(feature_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.assay_embedding = nn.Embedding(n_assays, hidden)
        self.answer = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.answer[-1].weight)
        nn.init.zeros_(self.answer[-1].bias)
        self.answer[-1].bias.data[1] = math.log(math.expm1(.35))

    def forward(self, features: torch.Tensor, base_mu: torch.Tensor):
        z = (features.float() - self.feature_mean) / self.feature_std
        context = self.project(z).unsqueeze(1) + self.assay_embedding.weight.unsqueeze(0)
        raw = self.answer(context)
        return base_mu.float() + 1.0 * torch.tanh(raw[..., 0]), .05 + F.softplus(raw[..., 1])

    @staticmethod
    def event_logits(mu, scale, thresholds, temperature=1.0):
        return (mu.unsqueeze(-1) - thresholds) / (scale.unsqueeze(-1) * temperature)

    @staticmethod
    def bin_probabilities(mu, scale, boundaries, temperature=1.0):
        below = torch.sigmoid(-RegDecisionHead.event_logits(mu, scale, boundaries, temperature))
        return torch.cat([below[..., :1], below[..., 1:] - below[..., :-1], 1 - below[..., -1:]], dim=-1).clamp(min=1e-8)

    @staticmethod
    def assay_logits(mu, scale, direction=1, temperature=1.0):
        return direction * mu / (scale.mean(-1, keepdim=True) * temperature)

    @staticmethod
    def pair_logits(mu_a, scale_a, mu_b, scale_b, temperature=1.0):
        return (mu_a - mu_b) / (torch.sqrt(scale_a.square() + scale_b.square()) * temperature)
