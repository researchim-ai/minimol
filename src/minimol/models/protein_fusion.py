"""Binding and matched-mutation research models.

Runtime parameters include the molecular encoder and the protein CNN. There is
no frozen ESM tower in this package. A numeric output is an assay-specific
synthetic or experimental endpoint, not a universal affinity constant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimol.data.sequences import AA_ALPHABET
from minimol.models.blocks import AttentionPool, ConvStem, ResBlock, make_dilations
from minimol.models.gnn import build_gine
from minimol.models.heads import ChoiceHead


class ProteinCNN(nn.Module):
    def __init__(self, channels: int = 384, n_blocks: int = 8, dropout: float = 0.1):
        super().__init__()
        dilations = make_dilations(n_blocks)
        self.stem = ConvStem(len(AA_ALPHABET), channels, dropout)
        self.blocks = nn.ModuleList(ResBlock(channels, dilation, dropout) for dilation in dilations)
        self.pool = AttentionPool(channels)
        self.out_dim = channels

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.stem(x)
        for block in self.blocks:
            hidden = block(hidden)
        return self.pool(hidden, pad_mask)


class FusionMLP(nn.Module):
    def __init__(self, mol_dim: int, prot_dim: int, hidden: int, depth: int, out_dim: int):
        super().__init__()
        self.mol_proj = nn.Linear(mol_dim, hidden)
        self.prot_proj = nn.Linear(prot_dim, hidden)
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU()])
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.out = nn.Linear(hidden, out_dim)

    def forward(
        self,
        mol: torch.Tensor,
        prot: torch.Tensor,
        prot_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        left = self.mol_proj(mol)
        if prot_b is None:
            right = self.prot_proj(prot)
        else:
            right = self.prot_proj(prot_b) - self.prot_proj(prot)
        return self.out(self.mlp(F.gelu(left * right)))


class MinimolBind(nn.Module):
    """Molecule plus one protein sequence. Endpoint must name Ki, Kd, or IC50."""

    def __init__(self, atom_dim: int, edge_dim: int):
        super().__init__()
        admet = build_gine(atom_dim, edge_dim, preset="30m", task_names=["unused"], n_assays=0)
        self.mol = admet.encoder
        self.protein = ProteinCNN(channels=448, n_blocks=8, dropout=0.1)
        self.fusion = FusionMLP(self.mol.out_dim, self.protein.out_dim, hidden=768, depth=2, out_dim=1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        n_graphs: int,
        protein: torch.Tensor,
        protein_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mol = self.mol(x, edge_index, edge_attr, batch, n_graphs)
        prot = self.protein(protein, protein_mask)
        return self.fusion(mol, prot).squeeze(-1)


class MinimolPrecision(nn.Module):
    """Predict a measured activity change between WT and mutant in one assay.

    The protein encoder is shared. Direction logits are decrease / no_change / increase
    for thresholds declared by the dataset, not a clinical interpretation.
    """

    def __init__(self, atom_dim: int, edge_dim: int):
        super().__init__()
        admet = build_gine(atom_dim, edge_dim, preset="30m", task_names=["unused"], n_assays=0)
        self.mol = admet.encoder
        self.protein = ProteinCNN(channels=384, n_blocks=8, dropout=0.1)
        self.fusion = FusionMLP(self.mol.out_dim, self.protein.out_dim, hidden=2048, depth=4, out_dim=1)
        direction_in = self.mol.out_dim + self.protein.out_dim * 3
        self.direction = ChoiceHead(direction_in, n_options=3)
        self.direction_proj = nn.Linear(direction_in, direction_in)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        n_graphs: int,
        protein_wt: torch.Tensor,
        protein_mut: torch.Tensor,
        mask_wt: torch.Tensor | None = None,
        mask_mut: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mol = self.mol(x, edge_index, edge_attr, batch, n_graphs)
        wild = self.protein(protein_wt, mask_wt)
        mutant = self.protein(protein_mut, mask_mut)
        delta = self.fusion(mol, wild, mutant).squeeze(-1)
        features = torch.cat([mol, wild, mutant, mutant - wild], dim=-1)
        direction_logits = self.direction(F.gelu(self.direction_proj(features)))
        return delta, direction_logits
