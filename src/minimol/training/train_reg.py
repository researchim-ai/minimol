"""Train Minimol-Reg on a long-format regulatory table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from minimol.data.normalize import LabelScaler, assay_order, audit_reg_frame, pivot_reg
from minimol.data.sequences import maybe_reverse_complement_batch, one_hot_sequence
from minimol.errors import MinimolError
from minimol.evaluation.bootstrap import grouped_macro_mae_ci
from minimol.evaluation.metrics import prediction_scale, regression_report
from minimol.evaluation.reg import reg_metric_value
from minimol.jsonutil import write_json
from minimol.models.blocks import count_parameters
from minimol.models.reg_cnn import MinimolReg
from minimol.seeding import set_seed
from minimol.tracking import git_sha, sha256_file
from minimol.training.checkpoint import load_checkpoint
from minimol.training.loop import fit, move_batch, resolve_device
from minimol.training.losses import inverse_variance_weights, masked_huber

ARCH_KEYS = {
    "head_kind",
    "channels",
    "n_blocks",
    "dropout",
    "kind",
    "transformer_dim",
    "transformer_layers",
    "heads",
    "ffn_mult",
}


class RegDataset(Dataset):
    def __init__(self, rows, scaler: LabelScaler, weights: np.ndarray):
        self.rows = rows
        self.scaler = scaler
        self.weights = weights

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        target = np.asarray(row["y"], dtype=np.float32)
        return {
            "x": one_hot_sequence(row["sequence"]),
            "y": torch.tensor(self.scaler.transform(target[None, :])[0], dtype=torch.float32),
            "y_raw": torch.tensor(target, dtype=torch.float32),
            "y_mask": torch.tensor(row["mask"], dtype=torch.float32),
            "weight": torch.tensor(self.weights[index], dtype=torch.float32),
        }


def collate_reg(batch):
    width = max(item["x"].shape[-1] for item in batch)
    encoded = []
    masks = []
    for item in batch:
        length = item["x"].shape[-1]
        if length < width:
            pad = torch.zeros(5, width - length)
            pad[4] = 1.0
            encoded.append(torch.cat([item["x"], pad], dim=-1))
            masks.append(torch.cat([torch.ones(length), torch.zeros(width - length)]).bool())
        else:
            encoded.append(item["x"])
            masks.append(torch.ones(length, dtype=torch.bool))
    return {
        "x": torch.stack(encoded),
        "pad_mask": torch.stack(masks),
        "y": torch.stack([item["y"] for item in batch]),
        "y_raw": torch.stack([item["y_raw"] for item in batch]),
        "y_mask": torch.stack([item["y_mask"] for item in batch]),
        "weight": torch.stack([item["weight"] for item in batch]),
    }


def _loaders(rows, assays, batch_size, seed, use_weights: bool):
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "validation"]
    if not train:
        raise MinimolError("training split is empty")
    y = np.stack([row["y"] for row in train])
    mask = np.stack([row["mask"] for row in train])
    scaler = LabelScaler().fit(y, mask)
    if use_weights:
        se = np.stack([row["se"] for row in rows])
        weights = inverse_variance_weights(se)
    else:
        weights = np.ones((len(rows), len(assays)), dtype=np.float64)
    index = {id(row): weight for row, weight in zip(rows, weights, strict=True)}

    def make(split_rows, shuffle):
        if not split_rows:
            return None
        subset_w = np.stack([index[id(row)] for row in split_rows])
        dataset = RegDataset(split_rows, scaler, subset_w)
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=shuffle,
            generator=generator,
            collate_fn=collate_reg,
        )

    return make(train, True), make(val, False), scaler, assays


def train_reg(config: dict, data_path: Path, out_dir: Path) -> dict:
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading {data_path}", flush=True)
    frame = pd.read_parquet(data_path)
    print(f"auditing {len(frame)} regulatory rows", flush=True)
    audit = audit_reg_frame(frame)
    if not audit["passed"]:
        raise MinimolError(f"regulatory table failed audit: {audit['issues'][:3]}")
    assays = assay_order(frame["assay"].tolist())
    rows = pivot_reg(frame, assays)
    train_cfg = config.get("train", {})
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    model_cfg = config.get("model", {})
    overrides = {key: value for key, value in (model_cfg.get("overrides") or {}).items() if key in ARCH_KEYS}
    preset = model_cfg.get("preset", "5m")
    padding_policy = str(model_cfg.get("padding_policy", "mask_aware"))
    model = MinimolReg(preset, n_assays=len(assays), padding_policy=padding_policy, **overrides)
    use_weights = bool(config.get("loss", {}).get("inverse_variance_weights", False))
    train_loader, val_loader, scaler, assays = _loaders(
        rows,
        assays,
        int(train_cfg.get("batch_size_per_gpu", 32)),
        seed,
        use_weights,
    )
    device = resolve_device(train_cfg.get("device", "auto"))
    print(
        f"reg rows {len(rows)} train {sum(row['split'] == 'train' for row in rows)} "
        f"validation {sum(row['split'] == 'validation' for row in rows)} device {device} preset {preset}",
        flush=True,
    )
    lr = float(train_cfg.get("lr", 3e-4))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    delta = float(config.get("loss", {}).get("delta", 1.0))
    rc_augment = bool(model_cfg.get("rc_augment", False))

    def loss_fn(module, batch):
        encoded = batch["x"]
        pad_mask = batch["pad_mask"]
        if rc_augment and module.training:
            encoded, pad_mask = maybe_reverse_complement_batch(encoded, True, pad_mask=pad_mask)
        pred = module(encoded, pad_mask)
        return masked_huber(pred, batch["y"], batch["y_mask"], delta=delta, sample_weight=batch["weight"])

    def metric_fn(module, loader):
        module.eval()
        actuals, preds, masks = [], [], []
        for batch in loader:
            batch = move_batch(batch, device)
            pred = scaler.inverse_torch(module(batch["x"], batch["pad_mask"]))
            actuals.append(batch["y_raw"])
            preds.append(pred)
            masks.append(batch["y_mask"])
        report = regression_report(
            torch.cat(actuals).cpu().numpy(),
            torch.cat(preds).cpu().numpy(),
            torch.cat(masks).cpu().numpy(),
            assays,
        )
        return reg_metric_value(report)

    lengths = [len(row["sequence"]) for row in rows]
    metadata = {
        "task": "reg",
        "head_kind": model.head_kind,
        "padding_policy": model.padding_policy,
        "preset": preset,
        "n_assays": len(assays),
        "overrides": overrides,
        "assay_names": assays,
        "scaler": scaler.state_dict(),
        "seed": seed,
        "training_seed": seed,
        "split_seed": (config.get("data") or {}).get("split_seed"),
        "split_source": (config.get("data") or {}).get("split_source", "as_stored_in_table"),
        "param_count": count_parameters(model),
        "git_sha": git_sha(),
        "research_use_only": True,
        "clinical_use": False,
        "rc_augment": rc_augment,
        "length_min": int(min(lengths)),
        "length_max": int(max(lengths)),
        "receptive_field_bp": model.receptive_field_bp,
        "data_sha256": sha256_file(data_path),
    }
    result = fit(
        model,
        train_loader,
        val_loader,
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
        resume_from=train_cfg.get("resume_from"),
    )
    load_checkpoint(out_dir / "checkpoint_best.pt", model, map_location=device)
    final = None
    if val_loader is not None:
        val_groups = [row["group_id"] for row in rows if row["split"] == "validation"]
        final = _report(model, val_loader, scaler, device, assays, groups=val_groups)
        final["per_assay"] = final["per_target"]
    payload = {
        "task": "reg",
        "head_kind": model.head_kind,
        "padding_policy": model.padding_policy,
        "training_seed": seed,
        "split_seed": metadata["split_seed"],
        "split_source": metadata["split_source"],
        "config": config,
        "n_train_sequences": len(train_loader.dataset),
        "n_validation_sequences": 0 if val_loader is None else len(val_loader.dataset),
        "model": preset,
        "param_count": metadata["param_count"],
        "seed": seed,
        "split": "validation",
        "metrics": final,
        "history": result["history"],
        "steps": result["steps"],
        "research_use_only": True,
        "clinical_use": False,
        "data_sha256": metadata["data_sha256"],
        "git_sha": metadata["git_sha"],
        "receptive_field_bp": model.receptive_field_bp,
        "checkpoint_rule": "lowest validation MAE; Spearman is reported and is not used to pick weights",
    }
    write_json(out_dir / "metrics.json", payload)
    write_json(out_dir / "manifest.json", {"data_sha256": metadata["data_sha256"], "assay_names": assays})
    return payload


def _report(model, loader, scaler, device, assays, groups=None):
    actuals, preds, masks = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            pred = scaler.inverse_torch(model(batch["x"], batch["pad_mask"]))
            actuals.append(batch["y_raw"].cpu().numpy())
            preds.append(pred.cpu().numpy())
            masks.append(batch["y_mask"].cpu().numpy())
    actual = np.concatenate(actuals)
    pred = np.concatenate(preds)
    mask = np.concatenate(masks)
    report = regression_report(actual, pred, mask, assays)
    report["prediction_scale"] = prediction_scale(actual, pred, mask, assays)
    if groups is not None and len(groups) == len(actual):
        report["grouped_mae_ci"] = grouped_macro_mae_ci(actual, pred, mask, groups)
    return report


def main(argv: list[str] | None = None) -> int:
    from minimol.config import load_yaml

    parser = argparse.ArgumentParser(prog="minimol.training.train_reg")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    if args.max_steps is not None:
        config.setdefault("train", {})["max_steps"] = args.max_steps
    data = args.data or config.get("data", {}).get("path")
    if not data:
        print("error: pass --data or set data.path", file=sys.stderr)
        return 1
    try:
        train_reg(config, Path(data), Path(args.out))
    except MinimolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
