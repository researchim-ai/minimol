"""Build processed tables. External corpora stay behind the license gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from minimol.data.synthetic import (
    write_admet_sample,
    write_bind_sample,
    write_cds_sample,
    write_precision_sample,
    write_reg_sample,
    write_tox_sample,
)
from minimol.errors import LicenseError, MinimolError


def build_dataset(dataset: str, out: Path, limit: int | None = None, seed: int = 0) -> dict:
    out = Path(out)
    if dataset in {"tdc_admet", "chembl", "bindingdb", "zinc", "plinder"}:
        raise LicenseError(
            f"{dataset} is not APPROVED. Refusing to download it or to emit a leaderboard table."
        )
    if dataset == "synthetic_reg":
        return write_reg_sample(out, n_groups=limit or 40, seed=seed)
    if dataset == "synthetic_cds":
        return write_cds_sample(out, per_species=limit or 8, seed=seed)
    if dataset == "synthetic_admet":
        return write_admet_sample(out, seed=seed)
    if dataset == "synthetic_tox":
        return write_tox_sample(out, seed=seed)
    if dataset == "synthetic_bind":
        return write_bind_sample(out, seed=seed)
    if dataset == "synthetic_precision":
        return write_precision_sample(out, seed=seed)
    raise MinimolError(f"unknown dataset {dataset}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minimol.data.build")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        manifest = build_dataset(args.dataset, Path(args.out), limit=args.limit, seed=args.seed)
    except (MinimolError, LicenseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {manifest['n_rows']} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
