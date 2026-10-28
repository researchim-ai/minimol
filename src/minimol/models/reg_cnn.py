"""Minimol-Reg: dilated CNN and a short-sequence hybrid for MPRA log2FC.

Cell-type embeddings condition a shared pooled representation. The head emits
one regression output per assay. Outputs are not probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from minimol.models.blocks import (
    AttentionPool,
    ConvStem,
    ResBlock,
    build_transformer,
    conv_receptive_field,
    make_dilations,
    sinusoidal_position,
)
from minimol.models.presets import REG_PRESETS, resolve_spec


class MinimolReg(nn.Module):
    def __init__(self, preset: str = "5m", n_assays: int = 3, **overrides):
        super().__init__()
        spec = resolve_spec(REG_PRESETS, preset, overrides)
        channels = int(spec["channels"])
        dropout = float(spec.get("dropout", 0.0))
        n_blocks = int(spec["n_blocks"])
        dilations = make_dilations(n_blocks)
        self.preset = preset
        self.kind = spec["kind"]
        self.dilations = dilations
        self.receptive_field_bp = conv_receptive_field(dilations)
        self.n_assays = int(n_assays)
        self.stem = ConvStem(5, channels, dropout)
        self.blocks = nn.ModuleList(ResBlock(channels, dilation, dropout) for dilation in dilations)
        pooled_dim = channels
        if spec["kind"] == "hybrid":
            d_model = int(spec["transformer_dim"])
            self.proj = nn.Conv1d(channels, d_model, kernel_size=1)
            self.transformer = build_transformer(
                d_model,
                int(spec["transformer_layers"]),
                int(spec["heads"]),
                int(spec["ffn_mult"]),
                dropout,
            )
            pooled_dim = d_model
        else:
            self.proj = None
            self.transformer = None
        self.pool = AttentionPool(pooled_dim)
        self.assay_emb = nn.Embedding(self.n_assays, pooled_dim)
        self.out = nn.Linear(pooled_dim, 1)
        self.pooled_dim = pooled_dim

    def encode(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.stem(x)
        for block in self.blocks:
            hidden = block(hidden)
        if self.transformer is not None:
            hidden = self.proj(hidden)
            tokens = hidden.transpose(1, 2)
            positions = sinusoidal_position(tokens.shape[1], tokens.shape[2], tokens.device, tokens.dtype)
            tokens = tokens + positions.unsqueeze(0)
            key_padding = None
            if pad_mask is not None and not bool(pad_mask.all()):
                key_padding = ~pad_mask
            tokens = self.transformer(tokens, src_key_padding_mask=key_padding)
            hidden = tokens.transpose(1, 2)
        return self.pool(hidden, pad_mask)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        pooled = self.encode(x, pad_mask)
        conditioned = pooled.unsqueeze(1) + self.assay_emb.weight.unsqueeze(0)
        return self.out(conditioned).squeeze(-1)

    def forward_assay(
        self,
        x: torch.Tensor,
        assay_index: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = self.encode(x, pad_mask)
        conditioned = pooled + self.assay_emb(assay_index)
        return self.out(conditioned).squeeze(-1)
