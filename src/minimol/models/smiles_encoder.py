"""Character CNN over SMILES. This is an optional encoder, not the ADMET default."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SMILES_CHARSET = " #<>=+-.()[]0123456789@%BCNOPSFIclnopsbr*"


class SmilesCharCNN(nn.Module):
    def __init__(self, hidden: int = 128, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.charset = SMILES_CHARSET
        self.embedding = nn.Embedding(len(self.charset), hidden, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1) for _ in range(n_layers)
        )
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden

    def tokenize(self, smiles: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        ids = []
        for text in smiles:
            row = [self.charset.index(char) if char in self.charset else self.charset.index("*") for char in text]
            ids.append(row or [0])
        width = max(len(row) for row in ids)
        tokens = torch.zeros(len(ids), width, dtype=torch.long)
        mask = torch.zeros(len(ids), width, dtype=torch.bool)
        for index, row in enumerate(ids):
            tokens[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[index, : len(row)] = True
        return tokens, mask

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(tokens).transpose(1, 2)
        for conv in self.convs:
            hidden = self.drop(F.gelu(conv(hidden)))
        tokens_h = hidden.transpose(1, 2)
        weights = mask.to(dtype=tokens_h.dtype).unsqueeze(-1)
        pooled = (tokens_h * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return pooled
