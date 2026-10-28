"""Binding trainer.

`preset: tiny` is a smoke network. `preset: bind_60m` is the research fusion
model. Neither output is a measured Ki unless the table's label_quality says so.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from minimol.data.featurize import atom_feature_dim, bond_feature_dim, collate_graphs, mol_to_graph
from minimol.data.normalize import LabelScaler, canonicalize_split
from minimol.data.sequences import AA_ALPHABET, one_hot_protein
from minimol.errors import MinimolError
from minimol.jsonutil import write_json
from minimol.models.blocks import count_parameters
from minimol.models.gnn import GINEEncoder
from minimol.models.protein_fusion import FusionMLP, MinimolBind, ProteinCNN
from minimol.seeding import set_seed
from minimol.tracking import git_sha, sha256_file
from minimol.training.checkpoint import load_checkpoint
from minimol.training.loop import fit, move_batch, resolve_device
from minimol.training.losses import masked_huber


class TinyBind(nn.Module):
    def __init__(self, atom_dim: int, edge_dim: int):
        super().__init__()
        self.mol = GINEEncoder(atom_dim, edge_dim, hidden=32, depth=2, expansion=1, dropout=0.0)
        self.protein = ProteinCNN(channels=32, n_blocks=2, dropout=0.0)
        self.fusion = FusionMLP(self.mol.out_dim, self.protein.out_dim, hidden=32, depth=1, out_dim=1)

    def forward(self, x, edge_index, edge_attr, batch, n_graphs, protein, protein_mask=None):
        return self.fusion(self.mol(x, edge_index, edge_attr, batch, n_graphs), self.protein(protein, protein_mask)).squeeze(-1)


class BindDataset(Dataset):
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
            "protein": one_hot_protein(row["protein"]),
            "y": torch.tensor(scaled, dtype=torch.float32),
            "y_raw": torch.tensor(raw, dtype=torch.float32),
        }


def _rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for record in frame.itertuples(index=False):
        graph = mol_to_graph(record.smiles_canonical, self_loops=True)
        if graph is None:
            continue
        rows.append(
            {
                "graph": graph,
                "protein": record.protein_sequence,
                "y": float(record.measurement),
                "split": canonicalize_split(record.split),
            }
        )
    return rows


def _collate(batch):
    payload = collate_graphs([item["graph"] for item in batch])
    width = max(item["protein"].shape[-1] for item in batch)
    proteins = []
    masks = []
    for item in batch:
        protein = item["protein"]
        length = protein.shape[-1]
        if length < width:
            pad = torch.zeros(len(AA_ALPHABET), width - length)
            pad[-1] = 1.0
            protein = torch.cat([protein, pad], dim=-1)
            mask = torch.cat([torch.ones(length), torch.zeros(width - length)]).bool()
        else:
            mask = torch.ones(width, dtype=torch.bool)
        proteins.append(protein)
        masks.append(mask)
    payload["protein"] = torch.stack(proteins)
    payload["protein_mask"] = torch.stack(masks)
    payload["y"] = torch.stack([item["y"] for item in batch]).view(-1)
    payload["y_raw"] = torch.stack([item["y_raw"] for item in batch]).view(-1)
    payload["y_mask"] = torch.ones(len(batch))
    return payload


def train_bind(config: dict, data_path: Path, out_dir: Path) -> dict:
    frame = pd.read_parquet(data_path)
    if frame["endpoint"].nunique() != 1:
        raise MinimolError("train one binding endpoint at a time")
    rows = _rows(frame)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "validation"]
    if not train_rows:
        raise MinimolError("binding training split is empty")
    train_cfg = config.get("train", {})
    seed = int(train_cfg.get("seed", 11))
    set_seed(seed)
    scaler = LabelScaler().fit(np.array([[row["y"]] for row in train_rows]), np.ones((len(train_rows), 1), dtype=bool))
    preset = config.get("model", {}).get("preset", "tiny")
    atom_dim, edge_dim = atom_feature_dim(), bond_feature_dim()
    if preset == "tiny":
        model = TinyBind(atom_dim, edge_dim)
    elif preset == "bind_60m":
        model = MinimolBind(atom_dim, edge_dim)
    else:
        raise MinimolError(f"unknown bind preset {preset}")
    batch_size = int(train_cfg.get("batch_size_per_gpu", 4))

    def loader(split_rows, shuffle):
        if not split_rows:
            return None
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            BindDataset(split_rows, scaler),
            batch_size=min(batch_size, len(split_rows)),
            shuffle=shuffle,
            generator=generator,
            collate_fn=_collate,
        )

    device = resolve_device(train_cfg.get("device", "auto"))
    lr = float(train_cfg.get("lr", 1e-3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    def loss_fn(module, batch):
        pred = module(
            batch["x"],
            batch["edge_index"],
            batch["edge_attr"],
            batch["batch"],
            batch["n_graphs"],
            batch["protein"],
            batch["protein_mask"],
        )
        return masked_huber(pred, batch["y"], batch["y_mask"])

    def metric_fn(module, data_loader):
        total = 0.0
        count = 0
        for batch in data_loader:
            batch = move_batch(batch, device)
            pred = scaler.inverse_torch(module(
                batch["x"], batch["edge_index"], batch["edge_attr"], batch["batch"], batch["n_graphs"],
                batch["protein"], batch["protein_mask"],
            ).unsqueeze(-1)).squeeze(-1)
            total += float(torch.mean(torch.abs(pred - batch["y_raw"])))
            count += 1
        return -total / max(count, 1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task": "bind",
        "preset": preset,
        "scaler": scaler.state_dict(),
        "seed": seed,
        "param_count": count_parameters(model),
        "git_sha": git_sha(),
        "research_use_only": True,
        "clinical_use": False,
        "label_quality": str(frame["label_quality"].iloc[0]) if "label_quality" in frame.columns else "unspecified",
        "endpoint": str(frame["endpoint"].iloc[0]),
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
        "task": "bind",
        "preset": preset,
        "param_count": metadata["param_count"],
        "steps": result["steps"],
        "endpoint": metadata["endpoint"],
        "label_quality": metadata["label_quality"],
        "research_use_only": True,
        "clinical_use": False,
        "note": "A synthetic interaction is not a Ki, Kd, or IC50.",
        "data_sha256": metadata["data_sha256"],
    }
    write_json(out_dir / "metrics.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    from minimol.config import load_yaml

    parser = argparse.ArgumentParser(prog="minimol.training.train_bind")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    if args.max_steps is not None:
        config.setdefault("train", {})["max_steps"] = args.max_steps
    try:
        train_bind(config, Path(args.data), Path(args.out))
    except MinimolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
