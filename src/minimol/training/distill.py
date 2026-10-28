"""Distillation is shipped only when it improves the held-out metric."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from minimol.errors import MinimolError
from minimol.jsonutil import write_json


def choose_recipe(metrics: dict, primary: str = "macro_spearman", min_epochs_met: bool = True) -> dict:
    if not min_epochs_met:
        return {
            "decision": "INCONCLUSIVE",
            "ship": None,
            "reason": "training budget is below the declared minimum",
        }
    if "supervised" not in metrics:
        raise MinimolError("a supervised baseline is required before distillation")

    def value(name: str) -> float:
        return float(metrics[name][primary])

    best = max(metrics, key=value)
    if value(best) <= value("supervised"):
        return {
            "decision": "NO-GO",
            "ship": "supervised",
            "reason": "teacher did not improve the held-out metric",
        }
    return {
        "decision": "GO",
        "ship": best,
        "reason": "teacher improved the held-out metric",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minimol.training.distill")
    parser.add_argument("--config")
    parser.add_argument("--metrics", required=True, help="JSON object of recipe -> metric dict")
    parser.add_argument("--primary", default="macro_spearman")
    parser.add_argument("--min-epochs-met", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        report = choose_recipe(
            json.loads(Path(args.metrics).read_text()),
            primary=args.primary,
            min_epochs_met=args.min_epochs_met,
        )
    except MinimolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    write_json(args.out, report)
    print(report["decision"])
    return 0 if report["decision"] != "INCONCLUSIVE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
