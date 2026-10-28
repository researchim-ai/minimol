"""Model-card rendering and the automated half of a release check.

commercial_release is always false here. A human legal and science review is
not something this script can grant.
"""

from __future__ import annotations

import json
from pathlib import Path

from minimol import RESEARCH_DISCLAIMER, RESEARCH_DISCLAIMER_RU
from minimol.data.registry import validate_registry
from minimol.errors import MinimolError

REQUIRED_PHRASES = (RESEARCH_DISCLAIMER, "не для клинического применения")


def render_model_card(metrics: dict, destination: Path) -> str:
    if metrics.get("clinical_use"):
        raise MinimolError("clinical_use is not allowed in a model card")
    measured = metrics.get("metrics") or metrics.get("endpoints") or metrics.get("ridge")
    body = "\n".join(
        [
            "# Minimol model card",
            "",
            RESEARCH_DISCLAIMER,
            "",
            RESEARCH_DISCLAIMER_RU,
            "",
            "Не для клинического применения.",
            "",
            f"- task: `{metrics.get('task')}`",
            f"- parameters: `{metrics.get('param_count', 'не измерено')}`",
            f"- seed: `{metrics.get('seed', 'не измерено')}`",
            "",
            "## Metrics",
            "",
            "```json",
            json.dumps(measured, indent=2, default=str) if measured is not None else "не измерено",
            "```",
            "",
            "Teacher soft labels are not ground truth. A parameter count is not evidence of biological skill.",
            "",
        ]
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(body)
    return body


def release_check(registry: Path, model_card: Path | None = None) -> dict:
    errors = validate_registry(registry)
    card_ok = False
    if model_card and Path(model_card).exists():
        text = Path(model_card).read_text()
        folded = text.casefold()
        card_ok = all(phrase.casefold() in folded for phrase in REQUIRED_PHRASES)
    else:
        errors.append("model card missing")
    return {
        "registry_errors": errors,
        "model_card_ok": card_ok,
        "commercial_release": False,
        "human_legal_review": "pending",
        "human_science_review": "pending",
        "passed_automated": not errors and card_ok,
    }
