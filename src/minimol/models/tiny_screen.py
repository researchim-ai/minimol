"""Tiny Reg screen models. Parameter counts are measured, not taken from the preset name."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from minimol.models.blocks import (
    AttentionPool,
    ConvStem,
    ResBlock,
    make_dilations,
    masked_group_norm,
    sequence_mask,
)

BIN_EDGES = (0.0, 0.5, 1.0, 2.0)
BIN_CENTERS = (-0.25, 0.25, 0.75, 1.5, 3.0)


class _AssayHead(nn.Module):
    def __init__(self, channels: int, n_assays: int, outputs: int):
        super().__init__()
        self.assay = nn.Embedding(n_assays, channels)
        self.hidden = nn.Sequential(nn.GELU(), nn.Linear(channels, channels), nn.GELU())
        self.out = nn.Linear(channels, outputs)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        conditioned = pooled.unsqueeze(1) + self.assay.weight.unsqueeze(0)
        return self.out(self.hidden(conditioned))


class _GlobalEncoder(nn.Module):
    def __init__(self, channels: int, n_blocks: int, dropout: float, depthwise: bool = False):
        super().__init__()
        self.stem = ConvStem(5, channels, dropout)
        dilations = make_dilations(n_blocks)
        block = _DepthwiseBlock if depthwise else ResBlock
        self.blocks = nn.ModuleList(block(channels, dilation, dropout) for dilation in dilations)
        self.pool = AttentionPool(channels)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.stem(x, pad_mask)
        for block in self.blocks:
            hidden = block(hidden, pad_mask)
        return self.pool(hidden, pad_mask)


class _DepthwiseBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(1 if channels < 8 else 8, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.pointwise(self.depthwise(sequence_mask(x, pad_mask)))
        hidden = sequence_mask(self.drop(F.gelu(masked_group_norm(self.norm, hidden, pad_mask))), pad_mask)
        return sequence_mask(x + hidden, pad_mask)


class _BinnedEncoder(_GlobalEncoder):
    def __init__(self, channels: int, n_blocks: int, dropout: float, n_bins: int = 4):
        super().__init__(channels, n_blocks, dropout, depthwise=False)
        self.n_bins = n_bins
        self.combine = nn.Linear(channels * n_bins, channels)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.stem(x, pad_mask)
        for block in self.blocks:
            hidden = block(hidden, pad_mask)
        batch, _channels, length = hidden.shape
        if pad_mask is None:
            valid = torch.ones(batch, length, dtype=torch.bool, device=hidden.device)
        else:
            valid = pad_mask.to(device=hidden.device, dtype=torch.bool)
        lengths = valid.sum(dim=-1).clamp(min=1)
        positions = torch.arange(length, device=hidden.device).unsqueeze(0)
        bin_id = torch.div(positions * self.n_bins, lengths.unsqueeze(1), rounding_mode="floor")
        bin_id = bin_id.clamp(max=self.n_bins - 1)
        pieces = []
        for index in range(self.n_bins):
            in_bin = valid & (bin_id == index)
            if bool(in_bin.any()):
                pooled = torch.nan_to_num(self.pool(hidden, in_bin), nan=0.0)
                pooled = pooled.masked_fill(~in_bin.any(dim=-1).unsqueeze(-1), 0.0)
            else:
                pooled = hidden.new_zeros(batch, hidden.shape[1])
            pieces.append(pooled)
        return self.combine(torch.cat(pieces, dim=-1))


class TinyScreen(nn.Module):
    """One of the stage-B recipes. `predict` always returns location and scale."""

    def __init__(self, recipe: str, n_assays: int = 3, dropout: float = 0.1):
        super().__init__()
        known = {"huber", "huber_matched", "distribution", "staged", "positional", "positional_huber", "ordinal", "legnet"}
        if recipe not in known:
            raise ValueError(f"unknown tiny recipe: {recipe}")
        self.recipe = recipe
        if recipe == "legnet":
            channels, blocks, depthwise = 160, 6, True
        elif recipe == "huber_matched":
            channels, blocks, depthwise = 136, 6, False
        else:
            channels, blocks, depthwise = 128, 6, False
        if recipe.startswith("positional"):
            self.encoder = _BinnedEncoder(channels, blocks, dropout)
        else:
            self.encoder = _GlobalEncoder(channels, blocks, dropout, depthwise=depthwise)
        outputs = 5 if recipe == "ordinal" else 1
        self.head = _AssayHead(channels, n_assays, outputs)
        self.scale_head = None if recipe in {"huber", "huber_matched", "positional_huber", "ordinal"} else _AssayHead(channels, n_assays, 1)
        self.n_assays = n_assays

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None):
        pooled = self.encoder(x, pad_mask)
        if self.recipe == "ordinal":
            return F.softmax(self.head(pooled), dim=-1)
        mu = self.head(pooled).squeeze(-1)
        if self.scale_head is None:
            return mu
        scale = 0.05 + F.softplus(self.scale_head(pooled).squeeze(-1))
        return mu, scale


def event_probability(prediction, thresholds: torch.Tensor, scale: torch.Tensor | None = None, temperature: float = 1.0):
    """P(y > t) for a location-scale pair or an ordinal bin distribution."""

    if torch.is_tensor(prediction) and prediction.dim() == 3:
        left = prediction.new_tensor([-1.0e6, 0.0, 0.5, 1.0, 2.0])
        right = prediction.new_tensor([0.0, 0.5, 1.0, 2.0, 4.0])
        width = (right - left).clamp(min=1e-6)
        masses = []
        for threshold in thresholds:
            above = (left >= threshold).to(prediction.dtype)
            partial = ((left < threshold) & (right > threshold)).to(prediction.dtype)
            fraction = partial * (right - threshold) / width
            masses.append((prediction * (above + fraction)).sum(-1))
        return torch.stack(masses, dim=-1)
    mu = prediction if scale is None else prediction
    sigma = scale if scale is not None else torch.ones_like(mu)
    return torch.sigmoid((mu.unsqueeze(-1) - thresholds) / (sigma.unsqueeze(-1) * temperature))
