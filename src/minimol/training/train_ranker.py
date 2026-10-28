"""Train a pairwise ranker on synthetic task labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from minimol.data.featurize import morgan_fingerprint
from minimol.errors import MinimolError
from minimol.jsonutil import write_json
from minimol.models.blocks import count_parameters
from minimol.models.heads import PairwiseRanker
from minimol.models.presets import RANK_PRESETS, resolve_spec
from minimol.seeding import set_seed
from minimol.tracking import git_sha
from minimol.training.checkpoint import load_checkpoint
from minimol.training.loop import fit, resolve_device


def pairs_from_tasks(tasks: list[dict]) -> list[tuple[str, str, float]]:
    pairs = []
    for task in tasks:
        objective = task["objective"]["property"]
        direction = task["objective"]["direction"]
        valid = [candidate for candidate in task["candidates"] if candidate["valid"]]
        for left in range(len(valid)):
            for right in range(left + 1, len(valid)):
                first = valid[left]["oracle"][objective]
                second = valid[right]["oracle"][objective]
                if first == second:
                    continue
                first_wins = first > second if direction == "maximize" else first < second
                pairs.append((valid[left]["smiles"], valid[right]["smiles"], 1.0 if first_wins else 0.0))
    return pairs


def _features(pairs, fp_dim: int):
    left, right, labels = [], [], []
    for smiles_a, smiles_b, label in pairs:
        fp_a = morgan_fingerprint(smiles_a, n_bits=fp_dim)
        fp_b = morgan_fingerprint(smiles_b, n_bits=fp_dim)
        if fp_a is None or fp_b is None:
            continue
        left.append(fp_a)
        right.append(fp_b)
        labels.append(label)
    if not left:
        raise MinimolError("no valid rank pairs")
    return (
        torch.tensor(np.stack(left), dtype=torch.float32),
        torch.tensor(np.stack(right), dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )


def train_ranker(tasks: list[dict], out_dir: Path, config: dict | None = None) -> dict:
    config = config or {}
    train_cfg = config.get("train", {})
    seed = int(train_cfg.get("seed", 0))
    set_seed(seed)
    preset = config.get("model", {}).get("preset", "5m")
    overrides = config.get("model", {}).get("overrides") or {}
    spec = resolve_spec(RANK_PRESETS, preset, overrides)
    pairs = pairs_from_tasks(tasks)
    fp_a, fp_b, labels = _features(pairs, int(spec["fp_dim"]))
    n_val = max(1, int(0.2 * len(labels)))
    train_ds = TensorDataset(fp_a[:-n_val], fp_b[:-n_val], labels[:-n_val])
    val_ds = TensorDataset(fp_a[-n_val:], fp_b[-n_val:], labels[-n_val:])
    if len(train_ds) == 0:
        train_ds = TensorDataset(fp_a, fp_b, labels)
        val_ds = train_ds
    batch = min(int(train_cfg.get("batch_size_per_gpu", 32)), len(train_ds))

    def collate(batch_rows):
        return {
            "fp_a": torch.stack([item[0] for item in batch_rows]),
            "fp_b": torch.stack([item[1] for item in batch_rows]),
            "y": torch.stack([item[2] for item in batch_rows]),
        }

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, generator=generator, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False, collate_fn=collate)
    model = PairwiseRanker(
        fp_dim=int(spec["fp_dim"]),
        hidden=int(spec["hidden"]),
        depth=int(spec["depth"]),
        dropout=float(spec.get("dropout", 0.0)),
    )
    device = resolve_device(train_cfg.get("device", "auto"))
    lr = float(train_cfg.get("lr", 1e-3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    def loss_fn(module, batch_rows):
        logits = module(batch_rows["fp_a"], batch_rows["fp_b"])
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, batch_rows["y"])

    def metric_fn(module, loader):
        correct = 0
        total = 0
        for batch_rows in loader:
            batch_rows = {key: value.to(device) for key, value in batch_rows.items()}
            pred = torch.sigmoid(module(batch_rows["fp_a"], batch_rows["fp_b"])) >= 0.5
            correct += int((pred == (batch_rows["y"] >= 0.5)).sum())
            total += int(batch_rows["y"].numel())
        return correct / max(total, 1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task": "rank",
        "preset": preset,
        "overrides": overrides,
        "fp_dim": int(spec["fp_dim"]),
        "seed": seed,
        "param_count": count_parameters(model),
        "git_sha": git_sha(),
        "research_use_only": True,
        "clinical_use": False,
        "label_source": "rdkit_oracle_pairs",
    }
    result = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        metric_fn,
        epochs=int(train_cfg.get("epochs", 2)),
        patience=int(train_cfg.get("early_stopping_patience", 5)),
        grad_clip=1.0,
        accumulation_steps=1,
        precision="fp32",
        device=device,
        max_steps=train_cfg.get("max_steps"),
        checkpoint_dir=out_dir,
        metadata=metadata,
        base_lr=lr,
    )
    load_checkpoint(out_dir / "checkpoint_best.pt", model, map_location=device)
    payload = {
        "task": "rank",
        "param_count": metadata["param_count"],
        "steps": result["steps"],
        "history": result["history"],
        "pairwise_accuracy": metric_fn(model, val_loader),
        "n_pairs": int(len(labels)),
        "research_use_only": True,
        "clinical_use": False,
        "note": "Pairs are ordered by exact RDKit properties, not by experimental activity.",
    }
    write_json(out_dir / "metrics.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    import json

    from minimol.config import load_yaml
    from minimol.generation.task_factory import make_tasks

    parser = argparse.ArgumentParser(prog="minimol.training.train_ranker")
    parser.add_argument("--config")
    parser.add_argument("--tasks")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)
    config = load_yaml(args.config) if args.config else {}
    if args.max_steps is not None:
        config.setdefault("train", {})["max_steps"] = args.max_steps
    if args.tasks:
        tasks = [json.loads(line) for line in Path(args.tasks).read_text().splitlines() if line.strip()]
    else:
        tasks = make_tasks(limit=args.limit, seed=int(config.get("train", {}).get("seed", 0)))
    try:
        train_ranker(tasks, Path(args.out), config)
    except MinimolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
