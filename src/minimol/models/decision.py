"""Typed router. Unsupported endpoints abstain instead of inventing a probability."""

from __future__ import annotations

from dataclasses import dataclass, field

from minimol import RESEARCH_DISCLAIMER, RESEARCH_DISCLAIMER_RU

SUPPORTED = {
    ("regulatory", "regression"),
    ("cds", "per_base"),
    ("admet", "regression"),
    ("admet", "noul"),
    ("admet", "choice"),
    ("admet", "score"),
    ("rank", "choice"),
    ("bind", "regression"),
    ("precision", "regression"),
}


@dataclass
class DecisionRequest:
    task: str
    kind: str
    endpoint: str | None = None
    options: list[str] | None = None
    event: dict | None = None
    bins: list[float] | None = None


@dataclass
class DecisionAnswer:
    abstain: bool
    reason: str | None = None
    value: float | None = None
    probability: float | None = None
    distribution: list[float] | None = None
    options: list[str] | None = None
    interval: list[float] | None = None
    warnings: list[str] = field(
        default_factory=lambda: ["research use only", RESEARCH_DISCLAIMER_RU]
    )

    def as_dict(self) -> dict:
        return {
            "abstain": self.abstain,
            "reason": self.reason,
            "value": self.value,
            "probability": self.probability,
            "distribution": self.distribution,
            "options": self.options,
            "interval": self.interval,
            "warnings": list(self.warnings),
            "disclaimer": RESEARCH_DISCLAIMER,
        }


class DecisionRouter:
    def __init__(self, known_endpoints: list[str] | None = None):
        self.known_endpoints = set(known_endpoints or [])

    def validate(self, request: DecisionRequest) -> DecisionAnswer | None:
        if (request.task, request.kind) not in SUPPORTED:
            return DecisionAnswer(True, "unsupported_question")
        if request.endpoint and self.known_endpoints and request.endpoint not in self.known_endpoints:
            return DecisionAnswer(True, "unsupported_endpoint")
        if request.kind == "choice" and not request.options:
            return DecisionAnswer(True, "choice_requires_options")
        if request.kind == "noul" and not request.event:
            return DecisionAnswer(True, "noul_requires_event")
        if request.kind == "score" and not request.bins:
            return DecisionAnswer(True, "score_requires_bins")
        return None

    def answer(self, request: DecisionRequest, prediction: dict | None = None) -> dict:
        refusal = self.validate(request)
        if refusal is not None:
            return refusal.as_dict()
        if prediction is None:
            return DecisionAnswer(True, "weights_not_loaded").as_dict()
        if prediction.get("ood"):
            return DecisionAnswer(True, "out_of_domain").as_dict()
        answer = DecisionAnswer(False)
        if request.kind == "regression":
            answer.value = prediction.get("value")
            answer.interval = prediction.get("interval")
            answer.probability = None
        elif request.kind == "noul":
            answer.probability = prediction.get("probability")
        elif request.kind == "choice":
            dist = prediction.get("distribution")
            answer.distribution = dist
            answer.options = list(request.options or [])
        elif request.kind == "score":
            answer.distribution = prediction.get("distribution")
            answer.options = [str(edge) for edge in request.bins or []]
        elif request.kind == "per_base":
            answer.value = None
            answer.probability = None
        return answer.as_dict()
