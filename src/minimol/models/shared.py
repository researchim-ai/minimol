"""Optional low-level DNA stem.

v1 trainers do not share this stem. A shared encoder is only a candidate if an
ablation shows no regression against the independent models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimol.models.blocks import ResBlock, conv_receptive_field, group_norm, make_dilations


class SharedStem(nn.Module):
    def __init__(self, channels: int = 128, n_blocks: int = 4, dropout: float = 0.0):
        super().__init__()
        dilations = make_dilations(n_blocks)
        self.receptive_field_bp = conv_receptive_field(dilations)
        self.conv = nn.Conv1d(5, channels, kernel_size=7, padding=3)
        self.norm = group_norm(channels)
        self.blocks = nn.ModuleList(ResBlock(channels, dilation, dropout) for dilation in dilations)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.norm(self.conv(x)))
        for block in self.blocks:
            hidden = block(hidden)
        return hidden
