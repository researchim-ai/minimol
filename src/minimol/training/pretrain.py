"""Descriptor pretraining is an ablation, not a default.

The synthetic ADMET table is already a computed-descriptor task. Calling this
module records that fact and, when data is passed, runs the ordinary supervised
trainer under recipe name `synthetic_pretrain`. Compare it with supervised-only
before keeping the extra stage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from minimol.errors import MinimolError
from minimol.training.train_admet import train_admet


def pretrain(config: dict, data_path: Path, out_dir: Path) -> dict:
    payload = train_admet(config, data_path, out_dir)
    payload["recipe"] = "synthetic_pretrain"
    payload["note"] = (
        "Auxiliary descriptor pretraining on computed properties. "
        "Keep it only if the supervised experimental model does not get worse."
    )
    from minimol.jsonutil import write_json

    write_json(Path(out_dir) / "metrics.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    from minimol.config import load_yaml

    parser = argparse.ArgumentParser(prog="minimol.training.pretrain")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    if args.max_steps is not None:
        config.setdefault("train", {})["max_steps"] = args.max_steps
    try:
        pretrain(config, Path(args.data), Path(args.out))
    except MinimolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
