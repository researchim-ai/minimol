"""Rescore the saved v1 Huber checkpoint with a calibration-only scale.

The original screen fit that scale on development labels. This script does not
rewrite artifacts/tiny_screen and does not re-forward the positional recipe:
those bins were taken from the padded length, so a new forward would mix in a
different change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from minimol.data.inner_split import load_inner_split
from minimol.data.normalize import LabelScaler, assay_order, pivot_reg
from minimol.evaluation.metrics import mae
from minimol.evaluation.reg_decide import TRAIN_THRESHOLDS, calibrate_temperature
from minimol.jsonutil import write_json
from minimol.models.tiny_screen import TinyScreen
from minimol.training.train_tiny_screen import (
    PARTITION_TO_SPLIT,
    _apply_scale,
    _brier,
    _constant_scale,
    _predict,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/malinois/data.parquet"))
    parser.add_argument("--split", type=Path, default=Path("data/processed/malinois_inner_v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/tiny_screen/huber/checkpoint_best.pt"))
    parser.add_argument("--published", type=Path, default=Path("artifacts/tiny_screen/huber/metrics.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/tiny_screen_v1_calibration_rescore"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest, assignments = load_inner_split(args.split)
    if manifest.get("version") != "malinois_inner_v1":
        raise SystemExit(f"expected malinois_inner_v1, found {manifest.get('version')}")
    published = json.loads(args.published.read_text())
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = blob["metadata"]
    if metadata.get("recipe") != "huber":
        raise SystemExit(f"expected the huber recipe, found {metadata.get('recipe')}")
    frame = pd.read_parquet(args.data, filters=[("split", "==", "train")])
    merged = frame.merge(assignments, on="sequence_hash", how="inner")
    if merged["sequence_hash"].nunique() != assignments["sequence_hash"].nunique():
        raise SystemExit("inner split v1 does not cover every official training sequence")
    holdout = merged.loc[merged["partition"].isin(["development", "calibration"])].copy()
    holdout["split"] = holdout["partition"].map(PARTITION_TO_SPLIT)
    assays = assay_order(holdout["assay"].tolist())
    grouped = {name: [] for name in ("validation", "calibration")}
    for row in pivot_reg(holdout, assays):
        grouped[row["split"]].append(row)
    model = TinyScreen("huber", n_assays=len(assays))
    model.load_state_dict(blob["model_state"])
    model.eval()
    scaler = LabelScaler().load_state_dict(metadata["scaler"])
    device = torch.device("cpu")
    dev_mu, _, dev_y, dev_mask, dev_ordinal = _predict(model, grouped["validation"], scaler, assays, device)
    cal_mu, _, cal_y, cal_mask, _ = _predict(model, grouped["calibration"], scaler, assays, device)
    leaked_dev = _apply_scale(dev_mu, _constant_scale(dev_mu, dev_y, dev_mask))
    leaked_cal = _apply_scale(cal_mu, _constant_scale(cal_mu, cal_y, cal_mask))
    leaked_temperature = calibrate_temperature(cal_mu, leaked_cal, cal_y, cal_mask)
    leaked_brier = _brier(dev_mu, leaked_dev, dev_y, dev_mask, dev_ordinal, leaked_temperature)
    fitted = _constant_scale(cal_mu, cal_y, cal_mask)
    fixed_dev = _apply_scale(dev_mu, fitted)
    fixed_cal = _apply_scale(cal_mu, fitted)
    temperature = calibrate_temperature(cal_mu, fixed_cal, cal_y, cal_mask)
    brier = _brier(dev_mu, fixed_dev, dev_y, dev_mask, dev_ordinal, temperature)
    brier_train = _brier(dev_mu, fixed_dev, dev_y, dev_mask, dev_ordinal, temperature, TRAIN_THRESHOLDS)
    y = dev_y.numpy()
    yhat = dev_mu.numpy()
    mask = dev_mask.numpy()
    per_assay = {
        assay: {"n": int(mask[:, index].sum()), "mae": mae(y[:, index], yhat[:, index], mask[:, index])}
        for index, assay in enumerate(assays)
    }
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "source_checkpoint": str(args.checkpoint),
        "split_version": manifest["version"],
        "assignments_sha256": manifest["assignments_sha256"],
        "published_development_brier": published.get("development_object_macro_brier"),
        "reproduced_development_label_scale_brier": leaked_brier,
        "calibration_only_scale_brier_report_thresholds": brier,
        "calibration_only_scale_brier_train_thresholds": brier_train,
        "published_temperature_procedure": "scale fit separately on development labels and on calibration",
        "corrected_temperature": temperature,
        "reproduced_temperature": leaked_temperature,
        "per_assay_scale": [float(value) for value in fitted],
        "per_assay_mae": per_assay,
        "thresholds_note": (
            "v1 Huber trained with the four report thresholds inside the loss. "
            "The corrected report-threshold Brier is therefore not an unseen-threshold result. "
            "Train thresholds were not targets of that loss."
        ),
        "positional_not_rescored": (
            "positional v1 bins depended on padded length. A forward with the current encoder "
            "would change the representation, so it is not part of this scale-only check."
        ),
        "other_recipes": (
            "distribution, staged, legnet, and ordinal kept a model scale. "
            "The development-label scale fit applied only to huber."
        ),
        "official_validation_used": False,
        "official_test_used": False,
        "research_use_only": True,
    }
    write_json(args.out / "huber.json", report)
    print(json.dumps({key: report[key] for key in (
        "published_development_brier",
        "reproduced_development_label_scale_brier",
        "calibration_only_scale_brier_report_thresholds",
        "calibration_only_scale_brier_train_thresholds",
    )}))


if __name__ == "__main__":
    main()
