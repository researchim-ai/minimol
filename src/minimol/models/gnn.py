"""Graph encoders for Minimol-ADMET.

GINE is the tiny-model family. DirectedMPNN is the Chemprop-like baseline:
a learned message passer trained from scratch, not a wrapper around Chemprop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimol.models.blocks import pool_mean_and_attention
from minimol.models.presets import GNN_PRESETS, resolve_spec


class GINEConv(nn.Module):
    def __init__(self, hidden: int, edge_dim: int, expansion: int, dropout: float):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.edge_proj = nn.Linear(edge_dim, hidden)
        mid = hidden * expansion
        self.lin1 = nn.Linear(hidden, mid)
        self.lin2 = nn.Linear(mid, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        agg = torch.zeros_like(x)
        if src.numel():
            message = F.relu(x[src] + self.edge_proj(edge_attr))
            agg.index_add_(0, dst, message)
        hidden = (1.0 + self.eps) * x + agg
        hidden = self.lin2(self.drop(F.relu(self.lin1(hidden))))
        return self.norm(x + hidden)


class GINEEncoder(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        edge_dim: int,
        hidden: int,
        depth: int,
        expansion: int,
        dropout: float,
    ):
        super().__init__()
        self.out_dim = hidden
        self.atom_encoder = nn.Linear(atom_dim, hidden)
        self.layers = nn.ModuleList(
            GINEConv(hidden, edge_dim, expansion, dropout) for _ in range(depth)
        )
        self.gate = nn.Linear(hidden, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        hidden = F.relu(self.atom_encoder(x))
        for layer in self.layers:
            hidden = layer(hidden, edge_index, edge_attr)
        return pool_mean_and_attention(hidden, batch, n_graphs, self.gate)


class DirectedMPNN(nn.Module):
    """Directed bond messages. Reverse edges are excluded from the neighbor sum."""

    def __init__(self, atom_dim: int, bond_dim: int, hidden: int = 128, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        self.out_dim = hidden
        self.depth = depth
        self.input = nn.Linear(atom_dim + bond_dim, hidden)
        self.message = nn.Linear(hidden, hidden)
        self.readout = nn.Linear(atom_dim + hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        rev_index: torch.Tensor,
        batch: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        if src.numel() == 0:
            atom_message = x.new_zeros(x.size(0), self.out_dim)
        else:
            seed = self.input(torch.cat([x[src], edge_attr], dim=-1))
            message = F.relu(seed)
            for _ in range(self.depth):
                agg = message.new_zeros(x.size(0), message.size(-1))
                agg.index_add_(0, dst, message)
                incoming = agg[src] - message[rev_index]
                message = F.relu(seed + self.message(self.drop(incoming)))
            atom_message = message.new_zeros(x.size(0), message.size(-1))
            atom_message.index_add_(0, dst, message)
        atoms = F.relu(self.readout(torch.cat([x, atom_message], dim=-1)))
        return pool_mean_and_attention(atoms, batch, n_graphs, self.gate)


class MinimolADMET(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        task_names: list[str],
        n_assays: int = 0,
    ):
        super().__init__()
        self.encoder = encoder
        self.task_names = list(task_names)
        hidden = int(encoder.out_dim)
        self.heads = nn.ModuleDict({name: nn.Linear(hidden, 1) for name in self.task_names})
        self.assay_emb = nn.Embedding(n_assays, hidden) if n_assays else None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        n_graphs: int,
        assay_index: torch.Tensor | None = None,
        rev_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if isinstance(self.encoder, DirectedMPNN):
            if rev_index is None:
                raise ValueError("DirectedMPNN requires rev_index")
            hidden = self.encoder(x, edge_index, edge_attr, rev_index, batch, n_graphs)
        else:
            hidden = self.encoder(x, edge_index, edge_attr, batch, n_graphs)
        if self.assay_emb is not None:
            if assay_index is None:
                raise ValueError("assay_index is required when assay embeddings are enabled")
            hidden = hidden + self.assay_emb(assay_index)
        return {name: head(hidden).squeeze(-1) for name, head in self.heads.items()}


def build_gine(
    atom_dim: int,
    edge_dim: int,
    preset: str = "5m",
    task_names: list[str] | None = None,
    n_assays: int = 4,
    **overrides,
) -> MinimolADMET:
    spec = resolve_spec(GNN_PRESETS, preset, overrides)
    encoder = GINEEncoder(
        atom_dim,
        edge_dim,
        int(spec["hidden"]),
        int(spec["depth"]),
        int(spec["expansion"]),
        float(spec["dropout"]),
    )
    names = list(task_names or ["logp_rdkit", "tpsa_rdkit", "qed_rdkit", "high_logp_proxy"])
    return MinimolADMET(encoder, names, n_assays=n_assays)
