"""Shared convolutional blocks. GroupNorm is used so batch size 1 is valid."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def group_norm(channels: int) -> nn.GroupNorm:
    groups = 1
    for candidate in (32, 16, 8, 4, 2):
        if channels % candidate == 0:
            groups = candidate
            break
    return nn.GroupNorm(groups, channels)


def make_dilations(n_blocks: int) -> list[int]:
    cycle = (1, 2, 4, 8, 16, 32)
    return [cycle[index % len(cycle)] for index in range(n_blocks)]


def conv_receptive_field(
    dilations: list[int],
    stem_kernel: int = 7,
    block_kernel: int = 3,
) -> int:
    """Maximum receptive field in base pairs of stem + two k=3 convs per block."""

    field = 1 + (stem_kernel - 1)
    for dilation in dilations:
        field += 2 * (block_kernel - 1) * int(dilation)
    return int(field)


def sequence_mask(hidden: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
    """Zero padded positions. pad_mask True marks real bases and has shape [B, L] or [L]."""

    if pad_mask is None:
        return hidden
    mask = pad_mask.to(dtype=hidden.dtype)
    if hidden.dim() == 3:
        mask = mask.unsqueeze(1)
    return hidden * mask


def masked_group_norm(norm: nn.GroupNorm, hidden: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
    """GroupNorm over real positions. A full mask matches nn.GroupNorm."""

    if pad_mask is None or bool(pad_mask.all()):
        return norm(hidden)
    batch, channels, length = hidden.shape
    groups = norm.num_groups
    per_group = channels // groups
    grouped = hidden.reshape(batch, groups, per_group, length)
    valid = pad_mask.to(dtype=hidden.dtype).reshape(batch, 1, 1, length)
    count = (valid.sum(dim=-1, keepdim=True) * per_group).clamp(min=1.0)
    mean = (grouped * valid).sum(dim=(2, 3), keepdim=True) / count
    centered = grouped - mean
    var = (centered.square() * valid).sum(dim=(2, 3), keepdim=True) / count
    normalized = (centered * torch.rsqrt(var + norm.eps)).reshape(batch, channels, length)
    if norm.affine:
        normalized = normalized * norm.weight.view(1, -1, 1) + norm.bias.view(1, -1, 1)
    return normalized


class ConvStem(nn.Module):
    def __init__(self, in_channels: int, channels: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, channels, kernel_size=7, padding=3)
        self.norm = group_norm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.conv(sequence_mask(x, pad_mask))
        hidden = masked_group_norm(self.norm, hidden, pad_mask)
        return sequence_mask(self.drop(F.gelu(hidden)), pad_mask)


class ResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation)
        self.norm1 = group_norm(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation)
        self.norm2 = group_norm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.conv1(sequence_mask(x, pad_mask))
        hidden = sequence_mask(self.drop(F.gelu(masked_group_norm(self.norm1, hidden, pad_mask))), pad_mask)
        hidden = self.conv2(hidden)
        hidden = sequence_mask(self.drop(F.gelu(masked_group_norm(self.norm2, hidden, pad_mask))), pad_mask)
        return sequence_mask(x + hidden, pad_mask)


class AttentionPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, C, L]. pad_mask True marks real bases.
        tokens = x.transpose(1, 2)
        scores = self.score(tokens).squeeze(-1)
        if pad_mask is not None:
            scores = scores.masked_fill(~pad_mask, -1.0e4)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return torch.einsum("bl,blc->bc", weights, tokens)


def sinusoidal_position(length: int, dim: int, device, dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
    )
    encoding = torch.zeros(length, dim, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(position * div)
    encoding[:, 1::2] = torch.cos(position * div[: encoding[:, 1::2].shape[1]])
    return encoding


def build_transformer(
    dim: int,
    layers: int,
    heads: int,
    ffn_mult: int,
    dropout: float,
) -> nn.TransformerEncoder:
    if dim % heads != 0:
        raise ValueError(f"transformer dim {dim} is not divisible by {heads} heads")
    layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=heads,
        dim_feedforward=dim * ffn_mult,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)


def pool_mean_and_attention(
    hidden: torch.Tensor,
    batch: torch.Tensor,
    n_graphs: int,
    gate: nn.Linear,
) -> torch.Tensor:
    """Sum of mean pooling and attention pooling. Output width equals hidden width."""

    # Pool in float32. Autocast can make the gate bf16 while the accumulator stays fp32.
    hidden_f = hidden.float()
    summed = hidden_f.new_zeros(n_graphs, hidden_f.size(-1))
    summed.index_add_(0, batch, hidden_f)
    ones = hidden_f.new_ones(hidden_f.size(0))
    counts = hidden_f.new_zeros(n_graphs).index_add_(0, batch, ones).clamp(min=1.0).unsqueeze(-1)
    mean = summed / counts
    scores = gate(hidden).float().squeeze(-1)
    maxes = torch.full((n_graphs,), -1.0e9, device=hidden.device, dtype=scores.dtype)
    maxes.scatter_reduce_(0, batch, scores, reduce="amax", include_self=True)
    weights = torch.exp(scores - maxes[batch])
    denom = hidden_f.new_zeros(n_graphs).index_add_(0, batch, weights).clamp(min=1.0e-9)
    alpha = weights / denom[batch]
    attended = hidden_f.new_zeros(n_graphs, hidden_f.size(-1))
    attended.index_add_(0, batch, hidden_f * alpha.unsqueeze(-1))
    return (mean + attended).to(dtype=hidden.dtype)
