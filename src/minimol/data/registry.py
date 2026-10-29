"""Source registry. REVIEW_REQUIRED rows are not training data."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from minimol.errors import LicenseError, MinimolError

REGISTRY_COLUMNS = [
    "source",
    "version",
    "url",
    "citation",
    "license",
    "commercial_use",
    "derivative_redistribution",
    "PII_risk",
    "assay_type",
    "download_date",
    "checksum",
    "reviewer",
    "decision",
    "notes",
]
DECISIONS = {"APPROVED", "LOCAL_USE", "REVIEW_REQUIRED", "REJECTED", "INTERNAL_ONLY"}
UNKNOWN_LICENSES = {"", "unknown", "unverified", "na", "n/a"}


def load_registry(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_registry(path: str | Path) -> list[str]:
    frame = load_registry(path)
    errors = []
    missing = [column for column in REGISTRY_COLUMNS if column not in frame.columns]
    if missing:
        errors.append("missing columns: " + ", ".join(missing))
        return errors
    if frame["source"].duplicated().any():
        errors.append("duplicate source ids")
    for record in frame.itertuples(index=False):
        decision = str(record.decision).strip()
        if decision not in DECISIONS:
            errors.append(f"{record.source}: bad decision {decision}")
            continue
        license_name = str(record.license).strip().lower()
        if decision == "APPROVED" and license_name in UNKNOWN_LICENSES:
            errors.append(f"{record.source}: APPROVED requires a named license")
        if decision == "APPROVED" and str(record.commercial_use).strip().lower() in {"no", "unknown"}:
            errors.append(f"{record.source}: APPROVED commercial_use must be yes")
        if decision == "LOCAL_USE" and license_name in UNKNOWN_LICENSES:
            errors.append(f"{record.source}: LOCAL_USE requires a named license")
        if decision == "LOCAL_USE" and str(record.derivative_redistribution).strip().lower() != "no":
            errors.append(f"{record.source}: LOCAL_USE forbids redistribution of the downloaded tables")
    return errors


def approved_sources(path: str | Path) -> list[str]:
    errors = validate_registry(path)
    if errors:
        raise MinimolError("; ".join(errors))
    frame = load_registry(path)
    return sorted(frame.loc[frame["decision"] == "APPROVED", "source"].astype(str).tolist())


def assert_source_approved(path: str | Path, source: str) -> None:
    frame = load_registry(path)
    rows = frame.loc[frame["source"] == source]
    if rows.empty:
        raise LicenseError(f"{source} is not in the registry")
    decision = str(rows.iloc[0]["decision"])
    if decision != "APPROVED":
        raise LicenseError(f"{source} decision={decision}; not included in training")


def resolve_registry_path(path: str | Path) -> Path:
    file_path = Path(path)
    if file_path.suffix in {".yaml", ".yml"}:
        from minimol.config import load_yaml

        configured = load_yaml(file_path).get("registry")
        if not configured:
            raise MinimolError(f"{file_path} does not set registry")
        return Path(configured)
    return file_path


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="minimol.data.registry")
    sub = parser.add_subparsers(dest="cmd", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    errors = validate_registry(resolve_registry_path(args.config))
    if errors:
        print("error: " + "; ".join(errors), file=sys.stderr)
        return 1
    print("registry ok")
    return 0


def write_registry_template(path: str | Path, rows: list[dict]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in REGISTRY_COLUMNS})


if __name__ == "__main__":
    raise SystemExit(main())
