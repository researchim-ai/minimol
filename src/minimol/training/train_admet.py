"""Train a GINE multitask head. Labels stay in their own endpoint columns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from minimol.data.assays import ASSAYS, task_kind
from minimol.data.featurize import atom_feature_dim, bond_feature_dim, collate_graphs, mol_to_graph
from minimol.data.normalize import LabelScaler, canonicalize_split
from minimol.errors import MinimolError
from minimol.evaluation.metrics import auroc, regression_report
from minimol.jsonutil import write_json
from minimol.models.blocks import count_parameters
from minimol.models.gnn import build_gine
from minimol.seeding import set_seed
from minimol.tracking import git_sha, sha256_file
from minimol.training.checkpoint import load_checkpoint
from minimol.training.loop import fit, move_batch, resolve_device
from minimol.training.losses import multitask_supervised_loss

ARCH_KEYS = {"hidden", "depth", "expansion", "dropout"}


def pivot_chem(frame: pd.DataFrame):
    split_counts = frame.assign(split=frame["split"].map(canonicalize_split)).groupby("smiles_canonical")["split"].nunique()
    if (split_counts > 1).any():
        raise MinimolError("a canonical molecule occurs in multiple splits; repair the split before training")
    endpoints = sorted(frame["endpoint"].unique())
    rows = []
    for smiles, group in frame.groupby("smiles_canonical", sort=True):
        graph = mol_to_graph(smiles, self_loops=True)
        if graph is None:
            continue
        record = {
            "smiles": smiles,
            "graph": graph,
            "split": canonicalize_split(group["split"].iloc[0]),
            "y": {},
            "mask": {},
        }
        for endpoint in endpoints:
            subset = group[group["endpoint"] == endpoint]
            if len(subset) and np.isfinite(subset["measurement"].iloc[0]):
                record["y"][endpoint] = float(subset["measurement"].iloc[0])
                record["mask"][endpoint] = True
            else:
                record["y"][endpoint] = 0.0
                record["mask"][endpoint] = False
        rows.append(record)
    return rows, endpoints


class ChemDataset(Dataset):
    def __init__(self, rows, endpoints, kinds, scaler: LabelScaler):
        self.rows = rows
        self.endpoints = endpoints
        self.kinds = kinds
        self.scaler = scaler

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        raw = np.asarray([row["y"][name] for name in self.endpoints], dtype=np.float32)
        scaled = raw.copy()
        for position, name in enumerate(self.endpoints):
            if self.kinds[name] == "regression":
                scaled[position] = (raw[position] - self.scaler.mean[position]) / self.scaler.std[position]
        return {
            "graph": row["graph"],
            "y": torch.tensor(scaled, dtype=torch.float32),
            "y_raw": torch.tensor(raw, dtype=torch.float32),
            "mask": torch.tensor([1.0 if row["mask"][name] else 0.0 for name in self.endpoints]),
        }


def _collate(endpoints):
    def collate(batch):
        payload = collate_graphs([item["graph"] for item in batch])
        stacked_y = torch.stack([item["y"] for item in batch])
        stacked_raw = torch.stack([item["y_raw"] for item in batch])
        stacked_mask = torch.stack([item["mask"] for item in batch])
        payload["y"] = {name: stacked_y[:, index] for index, name in enumerate(endpoints)}
        payload["y_raw"] = {name: stacked_raw[:, index] for index, name in enumerate(endpoints)}
        payload["mask"] = {name: stacked_mask[:, index] for index, name in enumerate(endpoints)}
        return payload

    return collate


def admet_selection_value(scaled_maes: list[float], aurocs: list[float]) -> float | None:
    """Higher is better. Regression and classification both vote when both are trained."""

    parts = [-float(value) for value in scaled_maes] + [float(value) for value in aurocs]
    if not parts:
        return None
    return float(np.mean(parts))


ADMET_CHECKPOINT_RULE = (
    "mean of -validation MAE/train_std for regression endpoints and AUROC for classification endpoints"
)


def _kinds(endpoints, declared: dict | None = None):
    kinds = {}
    for name in endpoints:
        if declared and name in declared:
            kinds[name] = declared[name]
        elif name in ASSAYS:
            kinds[name] = task_kind(name)
        else:
            kinds[name] = "regression"
    return kinds


def train_admet(config: dict, data_path: Path, out_dir: Path) -> dict:
    frame = pd.read_parquet(data_path)
    if "official_split" in frame.columns:
        raise MinimolError("official_split handling belongs to the splitter; do not relabel test here")
    rows, endpoints = pivot_chem(frame)
    declared = None
    if "task_kind" in frame.columns:
        declared = {str(name): str(kind) for name, kind in frame.groupby("endpoint")["task_kind"].first().items()}
    kinds = _kinds(endpoints, declared)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "validation"]
    if not train_rows:
        raise MinimolError("ADMET training split is empty")
    train_cfg = config.get("train", {})
    seed = int(train_cfg.get("seed", 11))
    set_seed(seed)
    y = np.stack([[row["y"][name] for name in endpoints] for row in train_rows])
    mask = np.stack([[row["mask"][name] for name in endpoints] for row in train_rows])
    # Binary columns are left unscaled by replacing their train values with zeros in a copy
    # and restoring std=1 after the fit. The dataset applies the scaler only to regression.
    y_fit = y.copy()
    for index, name in enumerate(endpoints):
        if kinds[name] != "regression":
            y_fit[:, index] = 0.0
    scaler = LabelScaler().fit(y_fit, mask)
    for index, name in enumerate(endpoints):
        if kinds[name] != "regression":
            scaler.mean[index] = 0.0
            scaler.std[index] = 1.0
    model_cfg = config.get("model", {})
    overrides = {key: value for key, value in (model_cfg.get("overrides") or {}).items() if key in ARCH_KEYS}
    preset = model_cfg.get("preset", "5m")
    model = build_gine(
        atom_feature_dim(),
        bond_feature_dim(),
        preset=preset,
        task_names=endpoints,
        n_assays=int(model_cfg.get("n_assays", 0)),
        **overrides,
    )
    batch_size = int(train_cfg.get("batch_size_per_gpu", 16))
    collate = _collate(endpoints)

    def make_loader(split_rows, shuffle):
        if not split_rows:
            return None
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            ChemDataset(split_rows, endpoints, kinds, scaler),
            batch_size=min(batch_size, len(split_rows)),
            shuffle=shuffle,
            generator=generator,
            collate_fn=collate,
        )

    device = resolve_device(train_cfg.get("device", "auto"))
    lr = float(train_cfg.get("lr", 3e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(train_cfg.get("weight_decay", 0.01)))
    delta = float(config.get("loss", {}).get("delta", 1.0))

    def loss_fn(module, batch):
        pred = module(
            batch["x"],
            batch["edge_index"],
            batch["edge_attr"],
            batch["batch"],
            batch["n_graphs"],
            assay_index=None,
            rev_index=batch["rev_index"],
        )
        return multitask_supervised_loss(pred, batch["y"], batch["mask"], kinds, delta=delta)

    def metric_fn(module, loader):
        report = _evaluate(module, loader, device, endpoints, kinds, scaler)
        scaled_maes = []
        raw_maes = []
        for index, name in enumerate(endpoints):
            item = report[name]
            if kinds[name] != "regression" or item.get("mae") is None:
                continue
            scale = max(abs(float(scaler.std[index])), 1e-6)
            raw_maes.append(float(item["mae"]))
            scaled_maes.append(float(item["mae"]) / scale)
        spearmans = [item["spearman"] for item in report.values() if item.get("spearman") is not None]
        aurocs = [item["auroc"] for item in report.values() if item.get("auroc") is not None]
        value = admet_selection_value(scaled_maes, aurocs)
        return {
            "value": value,
            "val_mae": None if not scaled_maes else float(np.mean(scaled_maes)),
            "val_spearman": None if not spearmans else float(np.mean(spearmans)),
            "val_auroc": None if not aurocs else float(np.mean(aurocs)),
        }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task": "admet",
        "preset": preset,
        "overrides": overrides,
        "endpoints": endpoints,
        "kinds": kinds,
        "scaler": scaler.state_dict(),
        "n_assays": int(model_cfg.get("n_assays", 0)),
        "seed": seed,
        "param_count": count_parameters(model),
        "git_sha": git_sha(),
        "research_use_only": True,
        "clinical_use": False,
        "official_tdc": False,
        "data_sha256": sha256_file(data_path),
        "checkpoint_rule": ADMET_CHECKPOINT_RULE,
    }
    result = fit(
        model,
        make_loader(train_rows, True),
        make_loader(val_rows, False),
        optimizer,
        loss_fn,
        metric_fn,
        epochs=int(train_cfg.get("epochs", 2)),
        patience=int(train_cfg.get("early_stopping_patience", 7)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        accumulation_steps=int(train_cfg.get("accumulation_steps", 1)),
        precision=train_cfg.get("precision", "fp32"),
        device=device,
        max_steps=train_cfg.get("max_steps"),
        checkpoint_dir=out_dir,
        metadata=metadata,
        base_lr=lr,
    )
    load_checkpoint(out_dir / "checkpoint_best.pt", model, map_location=device)
    final = _evaluate(model, make_loader(val_rows, False), device, endpoints, kinds, scaler)
    payload = {
        "task": "admet",
        "model": preset,
        "param_count": metadata["param_count"],
        "seed": seed,
        "endpoints": final,
        "history": result["history"],
        "steps": result["steps"],
        "checkpoint_rule": metadata["checkpoint_rule"],
        "research_use_only": True,
        "clinical_use": False,
        "official_tdc": False,
        "data_sha256": metadata["data_sha256"],
        "git_sha": metadata["git_sha"],
        "note": "Metrics describe the training table. Computed proxies are not experimental ADMET.",
    }
    write_json(out_dir / "metrics.json", payload)
    return payload


def _evaluate(model, loader, device, endpoints, kinds, scaler):
    if loader is None:
        return {}
    store = {name: {"y": [], "yhat": [], "mask": []} for name in endpoints}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            pred = model(
                batch["x"],
                batch["edge_index"],
                batch["edge_attr"],
                batch["batch"],
                batch["n_graphs"],
                rev_index=batch["rev_index"],
            )
            for index, name in enumerate(endpoints):
                values = pred[name]
                if kinds[name] == "regression":
                    mean = torch.as_tensor(scaler.mean[index], device=values.device, dtype=values.dtype)
                    std = torch.as_tensor(scaler.std[index], device=values.device, dtype=values.dtype)
                    values = values * std + mean
                else:
                    values = torch.sigmoid(values)
                store[name]["y"].append(batch["y_raw"][name].cpu().numpy())
                store[name]["yhat"].append(values.cpu().numpy())
                store[name]["mask"].append(batch["mask"][name].cpu().numpy())
    report = {}
    for name in endpoints:
        y = np.concatenate(store[name]["y"])
        yhat = np.concatenate(store[name]["yhat"])
        mask = np.concatenate(store[name]["mask"]) > 0.5
        if kinds[name] == "binary":
            report[name] = {"auroc": auroc(y, yhat, mask), "task": "binary"}
        else:
            metrics = regression_report(y, yhat, mask, [name])["per_target"][name]
            metrics["task"] = "regression"
            report[name] = metrics
    return report


def main(argv: list[str] | None = None) -> int:
    from minimol.config import load_yaml

    parser = argparse.ArgumentParser(prog="minimol.training.train_admet")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    config.setdefault("train", {})
    if args.seed is not None:
        config["train"]["seed"] = args.seed
    if args.max_steps is not None:
        config["train"]["max_steps"] = args.max_steps
    data = args.data or config.get("data", {}).get("path")
    if not data:
        print("error: pass --data", file=sys.stderr)
        return 1
    try:
        train_admet(config, Path(data), Path(args.out))
    except MinimolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
