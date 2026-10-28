"""Minimol-CDS: per-base coding probabilities on both strands.

The two outputs are independent. Coding regions may overlap on opposite strands,
so there is no mutual-exclusivity constraint.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimol.models.blocks import (
    ConvStem,
    ResBlock,
    build_transformer,
    conv_receptive_field,
    group_norm,
    make_dilations,
    sinusoidal_position,
)
from minimol.models.presets import CDS_PRESETS, resolve_spec


class _Stage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm1 = group_norm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm2 = group_norm(out_channels)
        self.drop = nn.Dropout(dropout)
        self.skip = (
            nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.drop(F.gelu(self.norm1(self.conv1(x))))
        hidden = self.drop(F.gelu(self.norm2(self.conv2(hidden))))
        return hidden + self.skip(x)


class DilatedCDS(nn.Module):
    def __init__(self, channels: int, n_blocks: int, dropout: float):
        super().__init__()
        dilations = make_dilations(n_blocks)
        self.dilations = dilations
        self.receptive_field_bp = conv_receptive_field(dilations)
        self.attention_window = None
        self.stem = ConvStem(5, channels, dropout)
        self.blocks = nn.ModuleList(ResBlock(channels, dilation, dropout) for dilation in dilations)
        self.head = nn.Conv1d(channels, 2, kernel_size=1)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        del pad_mask
        hidden = self.stem(x)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden)


class UNetCDS(nn.Module):
    def __init__(self, widths: tuple[int, ...] | list[int], dropout: float):
        super().__init__()
        widths = [int(width) for width in widths]
        self.widths = widths
        self.stem = _Stage(5, widths[0], dropout)
        self.downs = nn.ModuleList()
        self.encoders = nn.ModuleList()
        for left, right in zip(widths[:-1], widths[1:], strict=True):
            self.downs.append(nn.Conv1d(left, right, kernel_size=4, stride=2, padding=1))
            self.encoders.append(_Stage(right, right, dropout))
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        descending = list(reversed(widths))
        for deep, shallow in zip(descending[:-1], descending[1:], strict=True):
            self.ups.append(nn.ConvTranspose1d(deep, shallow, kernel_size=4, stride=2, padding=1))
            self.decoders.append(_Stage(shallow * 2, shallow, dropout))
        self.head = nn.Conv1d(widths[0], 2, kernel_size=1)
        self.multiple = 2 ** (len(widths) - 1)
        # Downsampling stacks local filters, so the field is larger than the 5M stack.
        self.receptive_field_bp = conv_receptive_field(make_dilations(4)) * self.multiple
        self.attention_window = None

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        del pad_mask
        original = x.shape[-1]
        remainder = (self.multiple - original % self.multiple) % self.multiple
        if remainder:
            x = F.pad(x, (0, remainder))
        skips = [self.stem(x)]
        hidden = skips[0]
        for down, encoder in zip(self.downs, self.encoders, strict=True):
            hidden = encoder(F.gelu(down(hidden)))
            skips.append(hidden)
        hidden = skips.pop()
        for up, decoder in zip(self.ups, self.decoders, strict=True):
            hidden = up(hidden)
            skip = skips.pop()
            if hidden.shape[-1] != skip.shape[-1]:
                length = min(hidden.shape[-1], skip.shape[-1])
                hidden = hidden[..., :length]
                skip = skip[..., :length]
            hidden = decoder(torch.cat([hidden, skip], dim=1))
        return self.head(hidden)[..., :original]


class HybridCDS(nn.Module):
    def __init__(
        self,
        channels: int,
        n_blocks: int,
        d_model: int,
        layers: int,
        heads: int,
        ffn_mult: int,
        window: int,
        dropout: float,
    ):
        super().__init__()
        dilations = make_dilations(n_blocks)
        self.dilations = dilations
        self.conv_receptive_field_bp = conv_receptive_field(dilations)
        self.attention_window = int(window)
        # Upper bound: each window token already mixes conv_receptive_field bases.
        self.receptive_field_bp = self.conv_receptive_field_bp + self.attention_window
        self.stem = ConvStem(5, channels, dropout)
        self.blocks = nn.ModuleList(ResBlock(channels, dilation, dropout) for dilation in dilations)
        self.proj = nn.Conv1d(channels, d_model, kernel_size=1)
        self.transformer = build_transformer(d_model, layers, heads, ffn_mult, dropout)
        self.head = nn.Conv1d(d_model, 2, kernel_size=1)

    def _attend(
        self,
        hidden: torch.Tensor,
        pad_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        tokens = hidden.transpose(1, 2)
        batch, length, dim = tokens.shape
        window = self.attention_window
        if length <= window:
            positions = sinusoidal_position(length, dim, tokens.device, tokens.dtype)
            tokens = tokens + positions.unsqueeze(0)
            key_padding = None
            if pad_mask is not None and not bool(pad_mask.all()):
                key_padding = ~pad_mask
            tokens = self.transformer(tokens, src_key_padding_mask=key_padding)
            return tokens.transpose(1, 2)
        extra = (window - length % window) % window
        if extra:
            tokens = F.pad(tokens, (0, 0, 0, extra))
            if pad_mask is not None:
                pad_mask = F.pad(pad_mask, (0, extra), value=False)
        padded = tokens.shape[1]
        windows = padded // window
        tokens = tokens.reshape(batch * windows, window, dim)
        positions = sinusoidal_position(window, dim, tokens.device, tokens.dtype)
        tokens = tokens + positions.unsqueeze(0)
        key_padding = None
        if pad_mask is not None:
            window_mask = pad_mask.reshape(batch * windows, window)
            if not bool(window_mask.all()):
                key_padding = ~window_mask
        tokens = self.transformer(tokens, src_key_padding_mask=key_padding)
        tokens = tokens.reshape(batch, padded, dim)[:, :length, :]
        return tokens.transpose(1, 2)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.stem(x)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.proj(hidden)
        hidden = self._attend(hidden, pad_mask)
        return self.head(hidden)


class MinimolCDS(nn.Module):
    def __init__(self, preset: str = "5m", **overrides):
        super().__init__()
        spec = resolve_spec(CDS_PRESETS, preset, overrides)
        self.preset = preset
        self.kind = spec["kind"]
        dropout = float(spec.get("dropout", 0.0))
        if spec["kind"] == "dilated":
            self.body = DilatedCDS(int(spec["channels"]), int(spec["n_blocks"]), dropout)
        elif spec["kind"] == "unet":
            self.body = UNetCDS(spec["widths"], dropout)
        elif spec["kind"] == "hybrid":
            self.body = HybridCDS(
                int(spec["channels"]),
                int(spec["n_blocks"]),
                int(spec["transformer_dim"]),
                int(spec["transformer_layers"]),
                int(spec["heads"]),
                int(spec["ffn_mult"]),
                int(spec["window"]),
                dropout,
            )
        else:
            raise KeyError(spec["kind"])

    @property
    def receptive_field_bp(self) -> int:
        return int(self.body.receptive_field_bp)

    @property
    def attention_window(self) -> int | None:
        return self.body.attention_window

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.body(x, pad_mask)
