"""Tox training is the ADMET trainer on an endpoint-specific table.

The endpoint name and label definition stay attached to the metrics. A computed
proxy is not a clinical toxicity probability.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from minimol.errors import MinimolError
from minimol.training.train_admet import train_admet


def train_tox(config: dict, data_path: Path, out_dir: Path) -> dict:
    payload = train_admet(config, data_path, out_dir)
    payload["task"] = "tox"
    payload["note"] = (
        "Tox heads are endpoint-specific. Do not pool them into a generic toxicity score."
    )
    from minimol.jsonutil import write_json

    write_json(Path(out_dir) / "metrics.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    from minimol.config import load_yaml

    parser = argparse.ArgumentParser(prog="minimol.training.train_tox")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    if args.max_steps is not None:
        config.setdefault("train", {})["max_steps"] = args.max_steps
    try:
        train_tox(config, Path(args.data), Path(args.out))
    except MinimolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
