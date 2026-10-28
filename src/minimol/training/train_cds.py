"""Train Minimol-CDS. Soft labels are optional and are not assumed to be Carbon-A."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from minimol.data.codecs import unpack_bits, unpack_float16
from minimol.data.sequences import one_hot_sequence, reverse_complement_one_hot
from minimol.errors import MinimolError
from minimol.evaluation.cds import nucleotide_report
from minimol.jsonutil import write_json
from minimol.models.blocks import count_parameters
from minimol.models.cds_cnn import MinimolCDS
from minimol.seeding import set_seed
from minimol.tracking import git_sha, sha256_file
from minimol.training.checkpoint import load_checkpoint
from minimol.training.loop import fit, move_batch, resolve_device
from minimol.training.losses import cds_loss, rc_strand_consistency_loss

ARCH_KEYS = {
    "channels",
    "n_blocks",
    "dropout",
    "kind",
    "widths",
    "transformer_dim",
    "transformer_layers",
    "heads",
    "ffn_mult",
    "window",
}


def _bytes(value) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return bytes(value)


def load_cds_rows(path: Path) -> tuple[list[dict], str]:
    frame = pd.read_parquet(path)
    rows = []
    for record in frame.itertuples(index=False):
        length = int(record.length)
        plus = unpack_bits(_bytes(record.gt_plus), length)
        minus = unpack_bits(_bytes(record.gt_minus), length)
        mask = unpack_bits(_bytes(record.valid_mask), length)
        teacher = np.stack(
            [
                unpack_float16(_bytes(record.teacher_p_plus), length),
                unpack_float16(_bytes(record.teacher_p_minus), length),
            ]
        )
        clipped = np.clip(teacher, 1e-4, 1.0 - 1e-4)
        logits = np.log(clipped / (1.0 - clipped)).astype(np.float32)
        rows.append(
            {
                "sequence": record.sequence,
                "y": np.stack([plus, minus]).astype(np.float32),
                "mask": np.stack([mask, mask]).astype(np.float32),
                "teacher": logits,
                "split": record.split,
            }
        )
    kind = str(frame["teacher_kind"].iloc[0]) if "teacher_kind" in frame.columns else "unknown"
    return rows, kind


class CDSDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            "x": one_hot_sequence(row["sequence"]),
            "y": torch.tensor(row["y"]),
            "mask": torch.tensor(row["mask"]),
            "teacher": torch.tensor(row["teacher"]),
        }


def collate_cds(batch):
    width = max(item["x"].shape[-1] for item in batch)

    def pad_x(tensor):
        if tensor.shape[-1] == width:
            return tensor
        extra = torch.zeros(5, width - tensor.shape[-1])
        extra[4] = 1.0
        return torch.cat([tensor, extra], dim=-1)

    def pad_channels(tensor):
        if tensor.shape[-1] == width:
            return tensor
        return F.pad(tensor, (0, width - tensor.shape[-1]))

    return {
        "x": torch.stack([pad_x(item["x"]) for item in batch]),
        "pad_mask": torch.stack(
            [
                torch.cat(
                    [
                        torch.ones(item["x"].shape[-1], dtype=torch.bool),
                        torch.zeros(width - item["x"].shape[-1], dtype=torch.bool),
                    ]
                )
                if item["x"].shape[-1] < width
                else torch.ones(width, dtype=torch.bool)
                for item in batch
            ]
        ),
        "y": torch.stack([pad_channels(item["y"]) for item in batch]),
        "mask": torch.stack([pad_channels(item["mask"]) for item in batch]),
        "teacher": torch.stack([pad_channels(item["teacher"]) for item in batch]),
    }


def _score(model, loader, device) -> dict:
    truths = []
    probs = []
    masks = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            pred = torch.sigmoid(model(batch["x"], batch["pad_mask"]))
            truths.append(batch["y"].cpu().numpy())
            probs.append(pred.cpu().numpy())
            masks.append(batch["mask"].cpu().numpy())
    truth = np.concatenate(truths, axis=0)
    prob = np.concatenate(probs, axis=0)
    mask = np.concatenate(masks, axis=0)
    # Micro-average by concatenating sequences on the length axis after masking per row is
    # sequence-length dependent. Report the mean of per-sequence macro F1 instead.
    scores = []
    boundaries = []
    for index in range(truth.shape[0]):
        report = nucleotide_report(truth[index], prob[index], mask[index])
        if report["macro_f1"] is not None:
            scores.append(report["macro_f1"])
        boundaries.append(report["strands"]["plus"]["boundary_f1"]["3"])
    return {
        "macro_f1": float(np.mean(scores)) if scores else None,
        "mean_boundary_f1_tol3": float(np.mean(boundaries)) if boundaries else None,
        "n": int(truth.shape[0]),
    }


def train_cds(config: dict, data_path: Path, out_dir: Path) -> dict:
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, teacher_kind = load_cds_rows(data_path)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "validation"]
    if not train_rows:
        raise MinimolError("CDS training split is empty")
    train_cfg = config.get("train", {})
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    model_cfg = config.get("model", {})
    overrides = {key: value for key, value in (model_cfg.get("overrides") or {}).items() if key in ARCH_KEYS}
    preset = model_cfg.get("preset", "5m")
    model = MinimolCDS(preset, **overrides)
    loss_cfg = config.get("loss", {})
    lambda_gt = float(loss_cfg.get("lambda_gt", 1.0))
    lambda_kd = float(loss_cfg.get("lambda_kd", 0.0))
    lambda_boundary = float(loss_cfg.get("lambda_boundary", 0.0))
    lambda_rc = float(loss_cfg.get("lambda_rc", 0.0))
    temperature = float(loss_cfg.get("temperature", 1.0))
    scale_by_t2 = bool(loss_cfg.get("scale_by_t2", False))
    if lambda_kd and teacher_kind == "missing":
        raise MinimolError("KD was requested but the table has no teacher labels")
    batch_size = int(train_cfg.get("batch_size_per_gpu", 4))

    def make_loader(split_rows, shuffle):
        if not split_rows:
            return None
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            CDSDataset(split_rows),
            batch_size=min(batch_size, len(split_rows)),
            shuffle=shuffle,
            generator=generator,
            collate_fn=collate_cds,
        )

    train_loader = make_loader(train_rows, True)
    val_loader = make_loader(val_rows, False)
    device = resolve_device(train_cfg.get("device", "auto"))
    lr = float(train_cfg.get("lr", 3e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(train_cfg.get("weight_decay", 0.01)))

    def loss_fn(module, batch):
        logits = module(batch["x"], batch["pad_mask"])
        total, _parts = cds_loss(
            logits,
            batch["y"],
            batch["mask"],
            teacher_logits=batch["teacher"] if lambda_kd else None,
            lambda_gt=lambda_gt,
            lambda_kd=lambda_kd,
            temperature=temperature,
            scale_by_t2=scale_by_t2,
            lambda_boundary=lambda_boundary,
        )
        if lambda_rc:
            flipped = reverse_complement_one_hot(batch["x"])
            flipped_mask = torch.flip(batch["pad_mask"], dims=(-1,))
            logits_rc = module(flipped, flipped_mask)
            total = total + lambda_rc * rc_strand_consistency_loss(logits, logits_rc)
        return total

    def metric_fn(module, loader):
        report = _score(module, loader, device)
        if report["macro_f1"] is None:
            return None
        return float(report["macro_f1"])

    metadata = {
        "task": "cds",
        "preset": preset,
        "overrides": overrides,
        "seed": seed,
        "param_count": count_parameters(model),
        "git_sha": git_sha(),
        "research_use_only": True,
        "clinical_use": False,
        "teacher_kind": teacher_kind,
        "lambda_kd": lambda_kd,
        "receptive_field_bp": model.receptive_field_bp,
        "attention_window": model.attention_window,
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
    )
    load_checkpoint(out_dir / "checkpoint_best.pt", model, map_location=device)
    final = _score(model, val_loader, device) if val_loader is not None else None
    note = None
    if lambda_kd and teacher_kind != "carbon_a":
        note = (
            "KD used a local teacher_kind="
            + teacher_kind
            + ". Do not report this run as Carbon-A distillation."
        )
    payload = {
        "task": "cds",
        "model": preset,
        "param_count": metadata["param_count"],
        "seed": seed,
        "split": "validation",
        "metrics": final,
        "history": result["history"],
        "steps": result["steps"],
        "teacher_kind": teacher_kind,
        "lambda_kd": lambda_kd,
        "note": note,
        "research_use_only": True,
        "clinical_use": False,
        "receptive_field_bp": model.receptive_field_bp,
        "attention_window": model.attention_window,
        "data_sha256": metadata["data_sha256"],
        "git_sha": metadata["git_sha"],
    }
    write_json(out_dir / "metrics.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    from minimol.config import load_yaml

    parser = argparse.ArgumentParser(prog="minimol.training.train_cds")
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
        train_cds(config, Path(data), Path(args.out))
    except MinimolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
