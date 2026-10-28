"""Mean, GC, k-mer ridge, and fingerprint baselines.

Neural baselines are the 1M CNN and the small directed MPNN, trained through
the same trainers with a small preset. XGBoost is used when it imports; otherwise
HistGradientBoosting is named explicitly in the metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from minimol.data.featurize import morgan_fingerprint
from minimol.data.normalize import assay_order, pivot_reg
from minimol.data.sequences import gc_fraction
from minimol.evaluation.metrics import auroc, mae, regression_report
from minimol.jsonutil import write_json
from minimol.oracles.rdkit_properties import MedianImputer, descriptor_vector
from minimol.tracking import sha256_file


def kmer_matrix(sequences, ks=(3, 4, 5, 6)) -> np.ndarray:
    widths = [4**k for k in ks]
    matrix = np.zeros((len(sequences), sum(widths)), dtype=np.float32)
    table = {"A": 0, "C": 1, "G": 2, "T": 3}
    for row, sequence in enumerate(sequences):
        offset = 0
        for k, width in zip(ks, widths, strict=True):
            counts = np.zeros(width, dtype=np.float32)
            windows = max(len(sequence) - k + 1, 0)
            for start in range(windows):
                acc = 0
                good = True
                for shift in range(k):
                    value = table.get(sequence[start + shift])
                    if value is None:
                        good = False
                        break
                    acc = acc * 4 + value
                if good:
                    counts[acc] += 1.0
            if windows:
                counts /= windows
            matrix[row, offset : offset + width] = counts
            offset += width
    return matrix


def mean_predict(y_train, mask_train, n_rows: int) -> np.ndarray:
    means = []
    for index in range(y_train.shape[1]):
        values = y_train[mask_train[:, index], index]
        means.append(float(values.mean()) if values.size else 0.0)
    return np.repeat(np.asarray(means)[None, :], n_rows, axis=0)


def fit_gc_baseline(sequences, y, mask):
    gc = np.asarray([gc_fraction(seq) for seq in sequences], dtype=np.float64)
    models = []
    for index in range(y.shape[1]):
        chosen = mask[:, index]
        if int(chosen.sum()) < 2:
            models.append((0.0, float(np.nanmean(y[:, index])) if y.size else 0.0))
            continue
        slope, intercept = np.polyfit(gc[chosen], y[chosen, index], 1)
        models.append((float(slope), float(intercept)))
    return models


def predict_gc(models, sequences) -> np.ndarray:
    gc = np.asarray([gc_fraction(seq) for seq in sequences], dtype=np.float64)
    pred = np.zeros((len(sequences), len(models)), dtype=np.float64)
    for index, (slope, intercept) in enumerate(models):
        pred[:, index] = slope * gc + intercept
    return pred


def fit_ridge(features, y, mask, alpha: float = 1.0):
    models = []
    for index in range(y.shape[1]):
        chosen = mask[:, index]
        model = Ridge(alpha=alpha)
        if int(chosen.sum()) < 2:
            models.append(None)
            continue
        model.fit(features[chosen], y[chosen, index])
        models.append(model)
    return models


def predict_ridge(models, features, fallback) -> np.ndarray:
    pred = np.zeros((features.shape[0], len(models)), dtype=np.float64)
    for index, model in enumerate(models):
        if model is None:
            pred[:, index] = fallback[index]
        else:
            pred[:, index] = model.predict(features)
    return pred


def _xy(rows, assays):
    sequences = [row["sequence"] for row in rows]
    y = np.stack([row["y"] for row in rows])
    mask = np.stack([row["mask"] for row in rows])
    return sequences, y, mask


def run_reg_baselines(data_path: Path, out_dir: Path, k_max: int = 6, alpha: float = 1.0) -> dict:
    frame = pd.read_parquet(data_path)
    assays = assay_order(frame["assay"].tolist())
    rows = pivot_reg(frame, assays)
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "validation"]
    seq_tr, y_tr, m_tr = _xy(train, assays)
    seq_va, y_va, m_va = _xy(val, assays)
    ks = tuple(range(3, k_max + 1))
    x_tr = kmer_matrix(seq_tr, ks)
    x_va = kmer_matrix(seq_va, ks)
    mean_hat = mean_predict(y_tr, m_tr, len(val))
    gc_models = fit_gc_baseline(seq_tr, y_tr, m_tr)
    gc_hat = predict_gc(gc_models, seq_va)
    ridge_models = fit_ridge(x_tr, y_tr, m_tr, alpha=alpha)
    fallback = mean_hat[0]
    ridge_hat = predict_ridge(ridge_models, x_va, fallback)
    payload = {
        "task": "reg",
        "model": "baselines",
        "split": "validation",
        "k_max": k_max,
        "param_count": 0,
        "research_use_only": True,
        "clinical_use": False,
        "data_sha256": sha256_file(data_path),
        "mean": regression_report(y_va, mean_hat, m_va, assays),
        "gc": regression_report(y_va, gc_hat, m_va, assays),
        "ridge": regression_report(y_va, ridge_hat, m_va, assays),
    }
    for key in ("mean", "gc", "ridge"):
        payload[key]["per_assay"] = payload[key]["per_target"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", payload)
    write_json(out_dir / "manifest.json", {"data_sha256": payload["data_sha256"], "assay_names": assays})
    return payload


def _tree():
    try:
        import xgboost as xgb

        def make(task, seed, rounds):
            if task == "binary":
                return xgb.XGBClassifier(
                    n_estimators=rounds,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=seed,
                    n_jobs=2,
                    eval_metric="logloss",
                )
            return xgb.XGBRegressor(
                n_estimators=rounds,
                max_depth=3,
                learning_rate=0.1,
                random_state=seed,
                n_jobs=2,
            )

        return make, "xgboost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        def make(task, seed, rounds):
            if task == "binary":
                return HistGradientBoostingClassifier(max_iter=rounds, random_state=seed)
            return HistGradientBoostingRegressor(max_iter=rounds, random_state=seed)

        return make, "hist_gradient_boosting"


def tanimoto_knn(train_fp, train_y, test_fp, k: int = 5) -> np.ndarray:
    train = np.asarray(train_fp, dtype=np.float64)
    test = np.asarray(test_fp, dtype=np.float64)
    target = np.asarray(train_y, dtype=np.float64)
    inter = test @ train.T
    union = test.sum(1, keepdims=True) + train.sum(1) - inter
    sims = inter / np.maximum(union, 1e-8)
    pred = np.zeros(test.shape[0], dtype=np.float64)
    for row in range(test.shape[0]):
        order = np.argsort(sims[row])[::-1][:k]
        weights = sims[row, order]
        if float(weights.sum()) <= 0:
            pred[row] = float(target.mean())
        else:
            pred[row] = float(np.average(target[order], weights=weights))
    return pred


def _chem_rows(frame: pd.DataFrame):
    endpoints = sorted(frame["endpoint"].unique())
    rows = []
    for smiles, group in frame.groupby("smiles_canonical", sort=True):
        record = {"smiles": smiles, "split": str(group["split"].iloc[0]), "y": {}, "mask": {}}
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


def run_chem_baselines(data_path: Path, out_dir: Path, seed: int = 0, rounds: int = 40) -> dict:
    frame = pd.read_parquet(data_path)
    rows, endpoints = _chem_rows(frame)
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "validation"]
    make, backend = _tree()
    reports = {}
    desc_train = np.stack([descriptor_vector(row["smiles"]) for row in train])
    desc_val = np.stack([descriptor_vector(row["smiles"]) for row in val])
    imputer = MedianImputer().fit(desc_train)
    desc_train = imputer.transform(desc_train)
    desc_val = imputer.transform(desc_val)
    fp_train = np.stack([morgan_fingerprint(row["smiles"]) for row in train])
    fp_val = np.stack([morgan_fingerprint(row["smiles"]) for row in val])
    for endpoint in endpoints:
        y_tr = np.asarray([row["y"][endpoint] for row in train], dtype=np.float64)
        m_tr = np.asarray([row["mask"][endpoint] for row in train], dtype=bool)
        y_va = np.asarray([row["y"][endpoint] for row in val], dtype=np.float64)
        m_va = np.asarray([row["mask"][endpoint] for row in val], dtype=bool)
        binary = set(np.unique(y_tr[m_tr])).issubset({0.0, 1.0}) and np.unique(y_tr[m_tr]).size <= 2
        task = "binary" if binary else "regression"
        model = make(task, seed, rounds)
        model.fit(desc_train[m_tr], y_tr[m_tr])
        if binary:
            scores = model.predict_proba(desc_val)[:, 1]
            reports[endpoint] = {"auroc": auroc(y_va, scores, m_va), "task": task, "backend": backend}
        else:
            pred = model.predict(desc_val)
            reports[endpoint] = {
                "mae": mae(y_va, pred, m_va),
                "mean_mae": mae(y_va, np.full_like(y_va, y_tr[m_tr].mean()), m_va),
                "task": task,
                "backend": backend,
            }
        if not binary:
            neighbor = tanimoto_knn(fp_train[m_tr], y_tr[m_tr], fp_val)
            reports[endpoint]["knn_mae"] = mae(y_va, neighbor, m_va)
    payload = {
        "task": "admet_baselines",
        "backend": backend,
        "endpoints": reports,
        "research_use_only": True,
        "clinical_use": False,
        "official_tdc": False,
        "data_sha256": sha256_file(data_path),
        "note": "Descriptor baselines on the supplied table. A computed proxy is not an experimental ADMET result.",
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", payload)
    write_json(out_dir / "manifest.json", {"data_sha256": payload["data_sha256"]})
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minimol.training.baselines")
    parser.add_argument("--config")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--task", choices=("reg", "admet"), default="reg")
    parser.add_argument("--k-max", type=int, default=6)
    args = parser.parse_args(argv)
    try:
        if args.task == "reg":
            run_reg_baselines(Path(args.data), Path(args.out), k_max=args.k_max)
        else:
            run_chem_baselines(Path(args.data), Path(args.out))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
