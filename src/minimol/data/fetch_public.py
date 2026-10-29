"""Download public training tables into the gitignored data/raw cache.

LOCAL_USE records a local training copy. It does not approve committing the
tables or redistributing them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from minimol.data.download import assert_download_allowed
from minimol.data.manifest import build_manifest, write_manifest
from minimol.data.registry import load_registry
from minimol.data.standardize import standardize_smiles
from minimol.errors import LicenseError, MinimolError

MALINOIS_REPO = "HuggingFaceBio/malinois-mpra-regression"
MALINOIS_ASSAYS = (
    ("K562", "K562_log2FC", "K562_lfcSE"),
    ("HepG2", "HepG2_log2FC", "HepG2_lfcSE"),
    ("SK-N-SH", "SKNSH_log2FC", "SKNSH_lfcSE"),
)
MALINOIS_COLUMNS = [
    "id",
    "split",
    "chromosome",
    "sequence",
    "K562_log2FC",
    "HepG2_log2FC",
    "SKNSH_log2FC",
    "K562_lfcSE",
    "HepG2_lfcSE",
    "SKNSH_lfcSE",
]


def assert_local_cache(dest: Path) -> Path:
    path = Path(dest)
    parts = path.parts
    if "data" not in parts or "raw" not in parts:
        raise MinimolError("downloads must land under data/raw so they stay out of git")
    if "raw" not in parts[parts.index("data") :]:
        raise MinimolError("downloads must land under data/raw so they stay out of git")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _registry_row(registry: Path, source: str):
    frame = load_registry(registry)
    rows = frame.loc[frame["source"] == source]
    if rows.empty:
        raise LicenseError(f"{source} is not in the registry")
    row = rows.iloc[0]
    decision = str(row["decision"]).strip()
    revision = str(row["version"]).strip()
    license_name = str(row["license"]).strip()
    assert_download_allowed(decision, revision, license_name)
    if decision == "LOCAL_USE" and str(row["derivative_redistribution"]).strip().lower() != "no":
        raise LicenseError(f"{source}: LOCAL_USE forbids redistribution")
    return row


def _write_provenance(dest: Path, payload: dict) -> None:
    path = Path(dest) / "provenance.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def download_malinois(registry: Path, dest: Path) -> dict:
    row = _registry_row(registry, "malinois_mpra")
    dest = assert_local_cache(dest)
    revision = str(row["version"]).strip()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise MinimolError("huggingface_hub is not installed") from exc
    snapshot_download(
        repo_id=MALINOIS_REPO,
        repo_type="dataset",
        revision=revision,
        local_dir=str(dest),
        allow_patterns=["data/*.parquet", "README.md"],
    )
    payload = {
        "source": "malinois_mpra",
        "repo": MALINOIS_REPO,
        "revision": revision,
        "license": str(row["license"]),
        "decision": str(row["decision"]),
        "redistribution": "no",
        "note": "Local training cache. Not committed.",
    }
    _write_provenance(dest, payload)
    return payload


def import_malinois(raw_dir: Path, out_dir: Path, *, se_max: float = 1.0, revision: str, license_id: str = "malinois_mpra") -> dict:
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("data/*.parquet")) or sorted(raw_dir.glob("*.parquet"))
    if not files:
        raise MinimolError(f"no parquet files in {raw_dir}")
    frames = [pd.read_parquet(path, columns=MALINOIS_COLUMNS) for path in files]
    wide = pd.concat(frames, ignore_index=True)
    wide["sequence"] = wide["sequence"].astype(str).str.strip().str.upper().str.replace("U", "T", regex=False)
    keep = wide["sequence"].str.fullmatch(r"[ACGTN]+").fillna(False)
    n_bad = int((~keep).sum())
    wide = wide.loc[keep].copy()
    wide["split"] = wide["split"].astype(str).str.strip().str.lower()
    known = {"train", "validation", "test"}
    wide.loc[~wide["split"].isin(known), "split"] = "quarantine"
    conflict_sequences = wide.groupby("sequence")["split"].nunique()
    conflict_sequences = set(conflict_sequences[conflict_sequences > 1].index)
    wide.loc[wide["sequence"].isin(conflict_sequences), "split"] = "quarantine"
    pieces = []
    for assay, value_column, error_column in MALINOIS_ASSAYS:
        part = pd.DataFrame(
            {
                "sequence": wide["sequence"].to_numpy(),
                "assay": assay,
                "split": wide["split"].to_numpy(),
                "group_id": wide["id"].astype(str).to_numpy(),
                "chromosome": wide["chromosome"].astype(str).to_numpy(),
                "y_log2fc": pd.to_numeric(wide[value_column], errors="coerce").to_numpy(),
                "y_se": pd.to_numeric(wide[error_column], errors="coerce").to_numpy(),
            }
        )
        pieces.append(part)
    long = pd.concat(pieces, ignore_index=True)
    values = long["y_log2fc"].to_numpy(dtype=np.float64)
    errors = long["y_se"].to_numpy(dtype=np.float64)
    long["valid_y"] = np.isfinite(values) & np.isfinite(errors) & (errors <= se_max)
    unique = pd.unique(long["sequence"])
    hashed = {sequence: hashlib.sha256(sequence.encode()).hexdigest() for sequence in unique}
    long["sequence_hash"] = long["sequence"].map(hashed)
    source_id = "gosai2024_nature_moesm4"
    raw_ids = (
        long["sequence"]
        + "\t"
        + long["assay"]
        + "\t"
        + source_id
        + "\t"
        + long["group_id"]
        + "\t"
        + revision
    )
    long["sample_id"] = ["sha256:" + hashlib.sha256(text.encode()).hexdigest() for text in raw_ids]
    long["source_id"] = source_id
    long["data_revision"] = revision
    long["license_id"] = license_id
    long["relation"] = "="
    long["units"] = "log2FC"
    long["label_quality"] = "experimental_mpra"
    long["target_name"] = "mpra_log2fc"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "data.parquet"
    long.to_parquet(data_path, index=False)
    manifest = build_manifest(
        long,
        task="reg",
        data_path=data_path,
        extra={
            "label_quality": "experimental_mpra",
            "note": "Gosai et al. Nature 2024 MPRA. Official chromosome splits. Local cache, not a synthetic table.",
            "se_max": se_max,
            "n_bad_sequences": n_bad,
            "n_sequences_quarantined": int(len(conflict_sequences)),
            "official_split": "chromosome",
            "data_revision": revision,
            "redistribution": "no",
        },
    )
    write_manifest(out_dir / "manifest.json", manifest)
    return manifest


TDC_DATAVERSE_FILE = "4426004"
TDC_DATAVERSE_URL = f"https://dataverse.harvard.edu/api/access/datafile/{TDC_DATAVERSE_FILE}"
TDC_METRICS = {
    "caco2_wang": "mae",
    "hia_hou": "roc-auc",
    "pgp_broccatelli": "roc-auc",
    "bioavailability_ma": "roc-auc",
    "lipophilicity_astrazeneca": "mae",
    "solubility_aqsoldb": "mae",
    "bbb_martins": "roc-auc",
    "ppbr_az": "mae",
    "vdss_lombardo": "spearman",
    "cyp2c9_veith": "pr-auc",
    "cyp2d6_veith": "pr-auc",
    "cyp3a4_veith": "pr-auc",
    "cyp2c9_substrate_carbonmangels": "pr-auc",
    "cyp3a4_substrate_carbonmangels": "roc-auc",
    "cyp2d6_substrate_carbonmangels": "pr-auc",
    "half_life_obach": "spearman",
    "clearance_hepatocyte_az": "spearman",
    "clearance_microsome_az": "spearman",
    "ld50_zhu": "mae",
    "herg": "roc-auc",
    "ames": "roc-auc",
    "dili": "roc-auc",
}


def tdc_style_train_valid(frame: pd.DataFrame, seed: int, entity: str = "Drug") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scaffold split of a TDC train_val table.

    Fractions match TDC ``get_train_valid_split``: 0.875 train and 0.125
    validation, with the published test file left untouched. Scaffold groups
    are assigned the same way as the Chemprop splitter TDC uses.
    """

    from collections import defaultdict
    from random import Random

    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    frac_train = 0.875
    scaffolds: dict[str, set[int]] = defaultdict(set)
    errors = 0
    for index, smiles in enumerate(frame[entity].astype(str).tolist()):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles), includeChirality=False)
            scaffolds[scaffold].add(index)
        except Exception:
            errors += 1
    usable = len(frame) - errors
    train_size = int(usable * frac_train)
    val_size = int(usable * 0.125)
    test_size = usable - train_size - val_size
    big, small = [], []
    for index_set in scaffolds.values():
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big.append(index_set)
        else:
            small.append(index_set)
    random = Random(seed)
    random.shuffle(big)
    random.shuffle(small)
    train_index: list[int] = []
    valid_index: list[int] = []
    for index_set in big + small:
        if len(train_index) + len(index_set) <= train_size:
            train_index.extend(index_set)
        else:
            valid_index.extend(index_set)
    return frame.iloc[train_index].reset_index(drop=True), frame.iloc[valid_index].reset_index(drop=True)


def download_tdc_admet(registry: Path, dest: Path, out_dir: Path, seed: int = 1) -> dict:
    row = _registry_row(registry, "tdc_admet")
    dest = assert_local_cache(dest)
    zip_path = dest / "admet_group.zip"
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        import urllib.request

        request = urllib.request.Request(TDC_DATAVERSE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response, zip_path.open("wb") as handle:
            handle.write(response.read())
    import zipfile

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)
    tables = {}
    for train_val_path in dest.rglob("train_val.csv"):
        test_path = train_val_path.parent / "test.csv"
        if test_path.exists():
            tables[train_val_path.parent.name] = (train_val_path, test_path)
    if len(tables) < len(TDC_METRICS):
        raise MinimolError(f"TDC archive has {len(tables)} tables, expected {len(TDC_METRICS)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (train_val_path, test_path) in sorted(tables.items()):
        train_val = pd.read_csv(train_val_path)
        test = pd.read_csv(test_path)
        train, valid = tdc_style_train_valid(train_val, seed=seed)
        manifest = _write_tdc_table(name, train, valid, test, out_dir / name, seed=seed, revision=str(row["version"]))
        written.append(
            {
                "name": name,
                "n_rows": manifest["n_rows"],
                "task_kind": manifest["task_kind"],
                "metric": manifest.get("tdc_metric"),
            }
        )
    payload = {
        "source": "tdc_admet",
        "revision": str(row["version"]),
        "license": str(row["license"]),
        "decision": str(row["decision"]),
        "seed": seed,
        "datasets": written,
        "redistribution": "no",
        "note": "Local benchmark cache. Scaffold splits are the TDC default for this seed. Not a leaderboard submission.",
    }
    _write_provenance(dest, payload)
    (out_dir / "index.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _write_tdc_table(name: str, train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, out_dir: Path, seed: int, revision: str) -> dict:
    pieces = []
    for split, frame in (("train", train), ("validation", valid), ("test", test)):
        copied = frame.copy()
        copied["split"] = split
        pieces.append(copied)
    combined = pd.concat(pieces, ignore_index=True)
    smiles_column = "Drug" if "Drug" in combined.columns else combined.columns[0]
    label_column = "Y" if "Y" in combined.columns else combined.columns[1]
    metric = TDC_METRICS.get(name, "")
    kind = "binary" if metric in {"roc-auc", "pr-auc"} else "regression"
    rows = []
    dropped = 0
    for record in combined.itertuples(index=False):
        original = getattr(record, smiles_column)
        standardized = standardize_smiles(str(original))
        if not standardized["valid"]:
            dropped += 1
            continue
        rows.append(
            {
                "sample_id": "sha256:"
                + hashlib.sha256(f"{name}\t{standardized['smiles_canonical']}\t{record.split}".encode()).hexdigest(),
                "smiles_original": standardized["smiles_original"],
                "smiles_canonical": standardized["smiles_canonical"],
                "inchikey": standardized["inchikey"],
                "scaffold_id": standardized["scaffold_id"],
                "scaffold": standardized["scaffold"],
                "assay_id": name,
                "endpoint": name,
                "target_name": name,
                "assay_type": "tdc_admet",
                "units": "tdc_native",
                "measurement": float(getattr(record, label_column)),
                "relation": "=",
                "source": "tdc_admet",
                "split": record.split,
                "label_quality": "tdc_benchmark",
                "license_id": "tdc_admet",
                "task_kind": kind,
                "stereochemistry_status": standardized["stereochemistry_status"],
                "organism": "as_published",
                "conditions": f"tdc_scaffold_split_seed_{seed}",
                "data_revision": revision,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise MinimolError(f"{name} produced no valid molecules")
    frame = frame.drop_duplicates(["smiles_canonical", "endpoint", "split"])
    conflict = frame.groupby("smiles_canonical")["split"].nunique()
    conflict_ids = set(conflict[conflict > 1].index)
    frame.loc[frame["smiles_canonical"].isin(conflict_ids), "split"] = "quarantine"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "data.parquet"
    frame.to_parquet(data_path, index=False)
    manifest = build_manifest(
        frame,
        task="admet",
        data_path=data_path,
        extra={
            "label_quality": "tdc_benchmark",
            "benchmark_name": name,
            "task_kind": kind,
            "tdc_metric": metric,
            "split_seed": seed,
            "n_invalid_smiles": dropped,
            "n_smiles_quarantined": int(len(conflict_ids)),
            "official_tdc": True,
            "note": "TDC scaffold split kept. Not an RDKit proxy and not a leaderboard submission by itself.",
            "redistribution": "no",
        },
    )
    write_manifest(out_dir / "manifest.json", manifest)
    return manifest
