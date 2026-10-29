"""Matched WT/mutant delta trainer.

`preset: tiny` is a smoke network. `preset: precision_100m` builds the research
fusion model. Direction classes use the dataset thresholds, not a clinical scale.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from minimol.data.featurize import atom_feature_dim, bond_feature_dim, collate_graphs, mol_to_graph
from minimol.data.normalize import LabelScaler, canonicalize_split
from minimol.data.sequences import AA_ALPHABET, one_hot_protein
from minimol.errors import MinimolError
from minimol.jsonutil import write_json
from minimol.models.blocks import count_parameters
from minimol.models.gnn import GINEEncoder
from minimol.models.protein_fusion import FusionMLP, MinimolPrecision, ProteinCNN
from minimol.seeding import set_seed
from minimol.tracking import git_sha, sha256_file
from minimol.training.checkpoint import load_checkpoint
from minimol.training.loop import fit, move_batch, resolve_device
from minimol.training.losses import masked_huber

DIRECTIONS = ("decrease", "no_change", "increase")


class TinyPrecision(nn.Module):
    def __init__(self, atom_dim: int, edge_dim: int):
        super().__init__()
        self.mol = GINEEncoder(atom_dim, edge_dim, hidden=32, depth=2, expansion=1, dropout=0.0)
        self.protein = ProteinCNN(channels=32, n_blocks=2, dropout=0.0)
        self.fusion = FusionMLP(self.mol.out_dim, self.protein.out_dim, hidden=32, depth=1, out_dim=1)
        self.direction = nn.Linear(self.mol.out_dim + self.protein.out_dim * 3, 3)

    def forward(self, x, edge_index, edge_attr, batch, n_graphs, protein_wt, protein_mut, mask_wt=None, mask_mut=None):
        mol = self.mol(x, edge_index, edge_attr, batch, n_graphs)
        wild = self.protein(protein_wt, mask_wt)
        mutant = self.protein(protein_mut, mask_mut)
        delta = self.fusion(mol, wild, mutant).squeeze(-1)
        logits = self.direction(torch.cat([mol, wild, mutant, mutant - wild], dim=-1))
        return delta, logits


class PrecisionDataset(Dataset):
    def __init__(self, rows, scaler: LabelScaler):
        self.rows = rows
        self.scaler = scaler

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        raw = np.array([row["y"]], dtype=np.float32)
        scaled = self.scaler.transform(raw[None, :])[0]
        return {
            "graph": row["graph"],
            "wt": one_hot_protein(row["wt"]),
            "mut": one_hot_protein(row["mut"]),
            "y": torch.tensor(scaled, dtype=torch.float32),
            "y_raw": torch.tensor(raw, dtype=torch.float32),
            "direction": row["direction"],
        }


def _pad_protein(protein: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    length = protein.shape[-1]
    if length < width:
        pad = torch.zeros(len(AA_ALPHABET), width - length)
        pad[-1] = 1.0
        protein = torch.cat([protein, pad], dim=-1)
        mask = torch.cat([torch.ones(length), torch.zeros(width - length)]).bool()
    else:
        mask = torch.ones(width, dtype=torch.bool)
    return protein, mask


def _collate(batch):
    payload = collate_graphs([item["graph"] for item in batch])
    width = max(max(item["wt"].shape[-1], item["mut"].shape[-1]) for item in batch)
    wild, wild_mask, mutant, mutant_mask = [], [], [], []
    for item in batch:
        wt, wt_mask = _pad_protein(item["wt"], width)
        mut, mut_mask = _pad_protein(item["mut"], width)
        wild.append(wt)
        wild_mask.append(wt_mask)
        mutant.append(mut)
        mutant_mask.append(mut_mask)
    payload["protein_wt"] = torch.stack(wild)
    payload["mask_wt"] = torch.stack(wild_mask)
    payload["protein_mut"] = torch.stack(mutant)
    payload["mask_mut"] = torch.stack(mutant_mask)
    payload["y"] = torch.stack([item["y"] for item in batch]).view(-1)
    payload["y_raw"] = torch.stack([item["y_raw"] for item in batch]).view(-1)
    payload["y_mask"] = torch.ones(len(batch))
    payload["direction"] = torch.tensor([item["direction"] for item in batch], dtype=torch.long)
    return payload


def _rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for record in frame.itertuples(index=False):
        graph = mol_to_graph(record.smiles_canonical, self_loops=True)
        if graph is None:
            continue
        if record.direction not in DIRECTIONS:
            raise MinimolError(f"unknown direction {record.direction}")
        rows.append(
            {
                "graph": graph,
                "wt": record.sequence_wt,
                "mut": record.sequence_mut,
                "y": float(record.measurement),
                "direction": DIRECTIONS.index(record.direction),
                "split": canonicalize_split(record.split),
                "protein_id": str(record.protein_id),
            }
        )
    return rows


def train_precision(config: dict, data_path: Path, out_dir: Path) -> dict:
    frame = pd.read_parquet(data_path)
    rows = _rows(frame)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "validation"]
    if not train_rows:
        raise MinimolError("precision training split is empty")
    proteins = {row["protein_id"] for row in train_rows} & {row["protein_id"] for row in val_rows}
    if proteins:
        raise MinimolError("a protein id appears in both train and validation")
    train_cfg = config.get("train", {})
    seed = int(train_cfg.get("seed", 11))
    set_seed(seed)
    scaler = LabelScaler().fit(np.array([[row["y"]] for row in train_rows]), np.ones((len(train_rows), 1), dtype=bool))
    preset = config.get("model", {}).get("preset", "tiny")
    atom_dim, edge_dim = atom_feature_dim(), bond_feature_dim()
    if preset == "tiny":
        model = TinyPrecision(atom_dim, edge_dim)
    elif preset == "precision_100m":
        model = MinimolPrecision(atom_dim, edge_dim)
    else:
        raise MinimolError(f"unknown precision preset {preset}")
    batch_size = int(train_cfg.get("batch_size_per_gpu", 4))

    def loader(split_rows, shuffle):
        if not split_rows:
            return None
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            PrecisionDataset(split_rows, scaler),
            batch_size=min(batch_size, len(split_rows)),
            shuffle=shuffle,
            generator=generator,
            collate_fn=_collate,
        )

    device = resolve_device(train_cfg.get("device", "auto"))
    lr = float(train_cfg.get("lr", 1e-3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    def loss_fn(module, batch):
        delta, logits = module(
            batch["x"],
            batch["edge_index"],
            batch["edge_attr"],
            batch["batch"],
            batch["n_graphs"],
            batch["protein_wt"],
            batch["protein_mut"],
            batch["mask_wt"],
            batch["mask_mut"],
        )
        regression = masked_huber(delta, batch["y"], batch["y_mask"])
        direction = F.cross_entropy(logits, batch["direction"])
        return regression + 0.5 * direction

    def metric_fn(module, data_loader):
        total = 0.0
        count = 0
        for batch in data_loader:
            batch = move_batch(batch, device)
            delta, _logits = module(
                batch["x"],
                batch["edge_index"],
                batch["edge_attr"],
                batch["batch"],
                batch["n_graphs"],
                batch["protein_wt"],
                batch["protein_mut"],
                batch["mask_wt"],
                batch["mask_mut"],
            )
            pred = scaler.inverse_torch(delta.unsqueeze(-1)).squeeze(-1)
            total += float(torch.mean(torch.abs(pred - batch["y_raw"])))
            count += 1
        return -total / max(count, 1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task": "precision",
        "preset": preset,
        "directions": list(DIRECTIONS),
        "scaler": scaler.state_dict(),
        "seed": seed,
        "param_count": count_parameters(model),
        "git_sha": git_sha(),
        "research_use_only": True,
        "clinical_use": False,
        "label_quality": "synthetic_matched_pair",
        "data_sha256": sha256_file(data_path),
    }
    result = fit(
        model,
        loader(train_rows, True),
        loader(val_rows, False),
        optimizer,
        loss_fn,
        metric_fn,
        epochs=int(train_cfg.get("epochs", 1)),
        patience=int(train_cfg.get("early_stopping_patience", 5)),
        grad_clip=1.0,
        accumulation_steps=1,
        precision="fp32",
        device=device,
        max_steps=train_cfg.get("max_steps", 2),
        checkpoint_dir=out_dir,
        metadata=metadata,
        base_lr=lr,
    )
    load_checkpoint(out_dir / "checkpoint_best.pt", model, map_location=device)
    payload = {
        "task": "precision",
        "preset": preset,
        "param_count": metadata["param_count"],
        "steps": result["steps"],
        "directions": list(DIRECTIONS),
        "research_use_only": True,
        "clinical_use": False,
        "note": "Synthetic matched pairs. Direction is decrease, no_change, or increase on this table's thresholds.",
        "data_sha256": metadata["data_sha256"],
    }
    write_json(out_dir / "metrics.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    from minimol.config import load_yaml

    parser = argparse.ArgumentParser(prog="minimol.training.train_precision")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    if args.max_steps is not None:
        config.setdefault("train", {})["max_steps"] = args.max_steps
    try:
        train_precision(config, Path(args.data), Path(args.out))
    except MinimolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
