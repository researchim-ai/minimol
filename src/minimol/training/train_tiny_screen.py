"""Stage-B Tiny screen: one inner split, one step budget, six recipes.

Development chooses the epoch. Calibration only fits the probability map.
Official validation and test are not read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from minimol.data.inner_split import load_inner_split
from minimol.data.normalize import LabelScaler, assay_order, pivot_reg
from minimol.errors import MinimolError
from minimol.evaluation.metrics import mae, pearson, spearman
from minimol.evaluation.reg_decide import REPORT_THRESHOLDS, TRAIN_THRESHOLDS, calibrate_temperature
from minimol.jsonutil import write_json
from minimol.models.blocks import count_parameters
from minimol.models.tiny_screen import BIN_CENTERS, TinyScreen, event_probability
from minimol.seeding import set_seed
from minimol.tracking import sha256_file
from minimol.training.checkpoint import load_checkpoint
from minimol.training.loop import fit, move_batch, resolve_device
from minimol.training.losses import masked_huber
from minimol.training.train_reg import RegDataset, collate_reg

RECIPES = ("huber", "distribution", "staged", "positional", "ordinal", "legnet")
FACTORIAL = ("huber", "huber_matched", "positional_huber", "distribution", "positional")
ALLOWED_RECIPES = set(RECIPES) | set(FACTORIAL)
EVENT_LAMBDA = 1.0
REGRESSION_SELECTOR = {"huber", "huber_matched", "positional_huber"}
PARTITION_TO_SPLIT = {"inner_train": "train", "development": "validation", "calibration": "calibration"}


def _object_brier(probability, target, mask) -> float:
    error = (probability - target).square()
    selected = mask.unsqueeze(-1).expand_as(error)
    per_object = (error * selected).sum(dim=(1, 2)) / selected.sum(dim=(1, 2)).clamp(min=1)
    keep = mask.any(dim=-1)
    if int(keep.sum()) == 0:
        return float("nan")
    return float(per_object[keep].mean())


def _predict(model, rows, scaler, assays, device):
    if not rows:
        raise MinimolError("partition is empty")
    loader = DataLoader(
        RegDataset(rows, scaler, np.ones((len(rows), len(assays)))),
        batch_size=128,
        shuffle=False,
        collate_fn=collate_reg,
    )
    mus, scales, ordinals = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(batch["x"], batch["pad_mask"])
            if torch.is_tensor(output) and output.dim() == 3:
                ordinals.append(output.float().cpu())
                centers = output.new_tensor((-0.25, 0.25, 0.75, 1.5, 3.0))
                mu = (output * centers).sum(-1)
                var = (output * (centers - mu.unsqueeze(-1)).square()).sum(-1).clamp(min=1e-4)
                mus.append(mu.float().cpu())
                scales.append(var.sqrt().float().cpu())
            elif isinstance(output, tuple):
                mu, scale = output
                mus.append(mu.float().cpu())
                scales.append(scale.float().cpu())
            else:
                mus.append(output.float().cpu())
                scales.append(torch.ones_like(output.float().cpu()))
    mu = torch.cat(mus)
    scale = torch.cat(scales)
    y = torch.tensor(np.stack([row["y"] for row in rows]), dtype=torch.float32)
    mask = torch.tensor(np.stack([row["mask"] for row in rows]), dtype=torch.bool)
    ordinal = torch.cat(ordinals) if ordinals else None
    return mu, scale, y, mask, ordinal


def _constant_scale(mu, y, mask) -> torch.Tensor:
    """Per-assay residual scale. Fit this on calibration only, then reuse the vector."""

    values = []
    for index in range(y.shape[1]):
        keep = mask[:, index]
        residual = y[keep, index] - mu[keep, index]
        value = float(residual.std()) if int(keep.sum()) > 1 else 0.5
        values.append(max(0.05, value))
    return torch.tensor(values, dtype=torch.float32)


def _apply_scale(mu: torch.Tensor, per_assay: torch.Tensor) -> torch.Tensor:
    return per_assay.to(dtype=mu.dtype).view(1, -1).expand_as(mu)


def _selection_probability(output, thresholds: torch.Tensor) -> torch.Tensor:
    """Epoch score uses the network's own distribution at temperature 1."""

    if torch.is_tensor(output) and output.dim() == 3:
        return event_probability(output, thresholds)
    if isinstance(output, tuple):
        mu, scale = output
        return event_probability(mu, thresholds, scale)
    return event_probability(output, thresholds, torch.ones_like(output))


def _brier(mu, scale, y, mask, ordinal, temperature: float, thresholds=REPORT_THRESHOLDS) -> float:
    thresholds = mu.new_tensor(thresholds)
    if ordinal is not None:
        probability = event_probability(ordinal, thresholds)
    else:
        probability = event_probability(mu, thresholds, scale, temperature)
    target = (y.unsqueeze(-1) > thresholds).to(probability.dtype)
    return _object_brier(probability, target, mask)


def train_tiny_recipe(
    recipe: str,
    rows_by_split: dict[str, list],
    assays: list[str],
    out_dir: Path,
    *,
    split_version: str,
    seed: int = 11,
    epochs: int = 8,
    patience: int = 3,
    batch_size: int = 128,
) -> dict:
    if recipe not in ALLOWED_RECIPES:
        raise MinimolError(f"unknown recipe {recipe}")
    set_seed(seed)
    device = resolve_device("auto")
    model = TinyScreen(recipe, n_assays=len(assays))
    train_rows = rows_by_split["train"]
    dev_rows = rows_by_split["validation"]
    scaler = LabelScaler().fit(np.stack([row["y"] for row in train_rows]), np.stack([row["mask"] for row in train_rows]))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        RegDataset(train_rows, scaler, np.ones((len(train_rows), len(assays)))),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_reg,
    )
    dev_loader = DataLoader(
        RegDataset(dev_rows, scaler, np.ones((len(dev_rows), len(assays)))),
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed + 1),
        collate_fn=collate_reg,
    )
    lr = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warm_steps = max(1, len(train_loader) * max(epochs // 2, 1)) if recipe == "staged" else 0
    seen = {"steps": 0}

    def _event_loss(probability, target, weight):
        event = (target.unsqueeze(-1) > probability.new_tensor(TRAIN_THRESHOLDS)).to(probability.dtype)
        sample = weight.unsqueeze(-1).expand_as(probability).to(probability.dtype)
        # Mean over valid assay-threshold events. Lambda is 1, so extra thresholds do not multiply the loss.
        return EVENT_LAMBDA * ((probability - event).square() * sample).sum() / sample.sum().clamp(min=1)

    def loss_fn(module, batch):
        seen["steps"] += 1
        output = module(batch["x"], batch["pad_mask"])
        target = batch["y_raw"]
        weight = batch["y_mask"]
        if torch.is_tensor(output) and output.dim() == 3:
            mu = (output * output.new_tensor(BIN_CENTERS)).sum(-1)
            probability = event_probability(output, output.new_tensor(TRAIN_THRESHOLDS))
            return masked_huber(mu, target, weight) + _event_loss(probability, target, weight)
        if isinstance(output, tuple):
            mu, scale = output
            huber = masked_huber(mu, target, weight)
            if recipe == "staged" and seen["steps"] <= warm_steps:
                return huber
            probability = event_probability(mu, mu.new_tensor(TRAIN_THRESHOLDS), scale)
            return huber + _event_loss(probability, target, weight)
        return masked_huber(output, target, weight)

    def metric_fn(module, loader):
        module.eval()
        actuals, preds, masks = [], [], []
        probabilities, targets = [], []
        with torch.no_grad():
            for batch in loader:
                batch = move_batch(batch, device)
                output = module(batch["x"], batch["pad_mask"])
                if recipe in REGRESSION_SELECTOR:
                    if torch.is_tensor(output) and output.dim() == 3:
                        mu = (output * output.new_tensor(BIN_CENTERS)).sum(-1)
                    elif isinstance(output, tuple):
                        mu = output[0]
                    else:
                        mu = output
                    actuals.append(batch["y_raw"])
                    preds.append(mu)
                    masks.append(batch["y_mask"])
                    continue
                thresholds = output[0].new_tensor(REPORT_THRESHOLDS) if isinstance(output, tuple) else output.new_tensor(REPORT_THRESHOLDS)
                probability = _selection_probability(output, thresholds)
                target = (batch["y_raw"].unsqueeze(-1) > thresholds).to(probability.dtype)
                probabilities.append(probability)
                targets.append(target)
                masks.append(batch["y_mask"] > 0.5)
        if recipe not in REGRESSION_SELECTOR:
            return -_object_brier(torch.cat(probabilities), torch.cat(targets), torch.cat(masks))
        y = torch.cat(actuals)
        yhat = torch.cat(preds)
        mask = torch.cat(masks) > 0.5
        y_np = y.cpu().numpy()
        yhat_np = yhat.float().cpu().numpy()
        mask_np = mask.cpu().numpy()
        values = [mae(y_np[:, index], yhat_np[:, index], mask_np[:, index]) for index in range(y_np.shape[1])]
        values = [value for value in values if value is not None]
        return None if not values else -float(np.mean(values))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task": "tiny_screen",
        "recipe": recipe,
        "seed": seed,
        "training_seed": seed,
        "split_source": split_version,
        "padding_policy": "mask_aware",
        "head_kind": recipe,
        "param_count": count_parameters(model),
        "n_assays": len(assays),
        "assay_names": assays,
        "scaler": scaler.state_dict(),
        "epochs_budget": epochs,
        "patience": patience,
    }
    result = fit(
        model,
        train_loader,
        dev_loader,
        optimizer,
        loss_fn,
        metric_fn,
        epochs=epochs,
        patience=patience,
        grad_clip=1.0,
        accumulation_steps=1,
        precision="bf16",
        device=device,
        max_steps=None,
        checkpoint_dir=out_dir,
        metadata=metadata,
        base_lr=lr,
    )
    load_checkpoint(out_dir / "checkpoint_best.pt", model, map_location=device)
    dev_mu, dev_scale, dev_y, dev_mask, dev_ordinal = _predict(model, dev_rows, scaler, assays, device)
    cal_mu, cal_scale, cal_y, cal_mask, cal_ordinal = _predict(model, rows_by_split["calibration"], scaler, assays, device)
    if dev_ordinal is None and recipe in REGRESSION_SELECTOR:
        fitted = _constant_scale(cal_mu, cal_y, cal_mask)
        dev_scale = _apply_scale(dev_mu, fitted)
        cal_scale = _apply_scale(cal_mu, fitted)
    if dev_ordinal is None:
        temperature = calibrate_temperature(cal_mu, cal_scale, cal_y, cal_mask)
    else:
        temperature = 1.0
    brier = _brier(dev_mu, dev_scale, dev_y, dev_mask, dev_ordinal, temperature)
    brier_seen = _brier(dev_mu, dev_scale, dev_y, dev_mask, dev_ordinal, temperature, TRAIN_THRESHOLDS)
    y = dev_y.numpy()
    yhat = dev_mu.numpy()
    mask = dev_mask.numpy()
    per_assay = {}
    for index, assay in enumerate(assays):
        per_assay[assay] = {
            "n": int(mask[:, index].sum()),
            "mae": mae(y[:, index], yhat[:, index], mask[:, index]),
            "pearson": pearson(y[:, index], yhat[:, index], mask[:, index]),
            "spearman": spearman(y[:, index], yhat[:, index], mask[:, index]),
        }
    report = {
        "task": "tiny_screen",
        "recipe": recipe,
        "seed": seed,
        "split_version": split_version,
        "param_count": metadata["param_count"],
        "epochs_ran": len(result["history"]),
        "steps": result["steps"],
        "development_object_macro_brier": brier,
        "development_brier_train_thresholds": brier_seen,
        "held_out_thresholds": list(REPORT_THRESHOLDS),
        "train_thresholds": list(TRAIN_THRESHOLDS),
        "event_loss_lambda": EVENT_LAMBDA,
        "temperature": temperature,
        "probability_fit": "per-assay scale and temperature fit on calibration only",
        "epoch_selection": (
            "development MAE"
            if recipe in REGRESSION_SELECTOR
            else "uncalibrated development object-macro Brier on held-out thresholds, using the model's own scale"
        ),
        "per_assay": per_assay,
        "history": result["history"],
        "official_validation_used": False,
        "official_test_used": False,
        "research_use_only": True,
        "clinical_use": False,
    }
    np.savez_compressed(
        out_dir / "development_predictions.npz",
        mu=dev_mu.numpy(),
        scale=dev_scale.numpy(),
        y=dev_y.numpy(),
        mask=dev_mask.numpy(),
        sequence_hash=np.asarray([row["sequence_hash"] for row in dev_rows]),
        group_id=np.asarray([row["group_id"] for row in dev_rows]),
        locus_group=np.asarray([row.get("locus_group", "") for row in dev_rows]),
        held_out_thresholds=np.asarray(REPORT_THRESHOLDS, dtype=np.float32),
        train_thresholds=np.asarray(TRAIN_THRESHOLDS, dtype=np.float32),
        temperature=np.asarray([temperature], dtype=np.float32),
    )
    write_json(out_dir / "metrics.json", report)
    print(json.dumps({"recipe": recipe, "brier": brier, "mae": {k: v["mae"] for k, v in per_assay.items()}, "params": report["param_count"]}), flush=True)
    return report


def load_screen_rows(data_path: Path, split_dir: Path) -> tuple[dict, list[str], dict[str, list]]:
    _manifest, assignments = load_inner_split(split_dir)
    frame = pd.read_parquet(data_path, filters=[("split", "==", "train")])
    merged = frame.merge(assignments, on="sequence_hash", how="inner")
    if merged["sequence_hash"].nunique() != assignments["sequence_hash"].nunique():
        raise MinimolError("inner split does not cover every official training sequence")
    merged = merged.loc[merged["partition"] != "purged"].copy()
    merged["split"] = merged["partition"].map(PARTITION_TO_SPLIT)
    assays = assay_order(merged["assay"].tolist())
    rows = pivot_reg(merged, assays)
    grouped = {name: [row for row in rows if row["split"] == name] for name in ("train", "validation", "calibration")}
    if not grouped["train"] or not grouped["validation"] or not grouped["calibration"]:
        raise MinimolError("inner split is missing a partition")
    return _manifest, assays, grouped


def run_screen(data_path: Path, split_dir: Path, out_dir: Path, recipes=RECIPES, epochs: int = 8, patience: int = 3, seed: int = 11) -> dict:
    manifest, assays, grouped = load_screen_rows(data_path, split_dir)
    print(
        f"inner train {len(grouped['train'])} development {len(grouped['validation'])} "
        f"calibration {len(grouped['calibration'])}",
        flush=True,
    )
    out_dir = Path(out_dir)
    reports = []
    for recipe in recipes:
        finished = out_dir / recipe / "metrics.json"
        if finished.exists():
            print(f"reuse {recipe}", flush=True)
            reports.append(json.loads(finished.read_text()))
            continue
        reports.append(
            train_tiny_recipe(
                recipe,
                grouped,
                assays,
                out_dir / recipe,
                split_version=manifest["version"],
                seed=seed,
                epochs=epochs,
                patience=patience,
            )
        )
    control = next(item for item in reports if item["recipe"] == "huber")
    for item in reports:
        item["brier_delta_vs_huber"] = item["development_object_macro_brier"] - control["development_object_macro_brier"]
    summary = {
        "split_version": manifest["version"],
        "data_sha256": manifest["data_sha256"],
        "seed": seed,
        "epochs_budget": epochs,
        "patience": patience,
        "primary": "development object-macro Brier on thresholds withheld from the loss; regression scale and temperature fit on calibration after checkpoint selection",
        "code_sha256": {
            "train_tiny_screen.py": sha256_file(Path(__file__).resolve()),
            "tiny_screen.py": sha256_file(Path(event_probability.__code__.co_filename).resolve()),
        },
        "recipes": reports,
        "official_test_used": False,
    }
    write_json(out_dir / "screen_summary.json", summary)
    return summary
