"""FastAPI surface. Sequences are not written to the log."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from minimol import RESEARCH_DISCLAIMER, RESEARCH_DISCLAIMER_RU, __version__
from minimol.data.normalize import LabelScaler
from minimol.data.sequences import n_fraction, one_hot_sequence
from minimol.data.standardize import standardize_smiles
from minimol.models.cds_cnn import MinimolCDS
from minimol.models.reg_cnn import MinimolReg
from minimol.oracles.rdkit_properties import DETERMINISTIC_PROPERTIES, properties_from_smiles
from minimol.serving.schemas import CDSRequest, PredictRequest, RankRequest, RegulatoryRequest

LOGGER = logging.getLogger("minimol.serving")
WARNINGS = ["research use only", "не для клинического применения", RESEARCH_DISCLAIMER_RU]
ORACLE_UNITS = {
    "qed": "unitless_0_1",
    "logp": "log10_partition",
    "tpsa": "angstrom2",
    "mw": "dalton",
    "hbd": "count",
    "hba": "count",
    "rotatable": "count",
    "rings": "count",
    "sa_proxy": "proxy_not_ertl_sa",
}
ORACLE_ALIAS = {
    "logp_rdkit": "logp",
    "tpsa_rdkit": "tpsa",
    "qed_rdkit": "qed",
    "high_logp_proxy": "logp",
}


@dataclass
class LoadedModel:
    model: torch.nn.Module
    meta: dict
    scaler: LabelScaler | None = None


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.time()
        window = [stamp for stamp in self.hits.get(key, []) if now - stamp < 60]
        if len(window) >= self.per_minute:
            self.hits[key] = window
            return False
        window.append(now)
        self.hits[key] = window
        return True


def load_reg_checkpoint(path) -> LoadedModel:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    meta = blob["metadata"]
    model = MinimolReg(meta["preset"], n_assays=meta["n_assays"], **(meta.get("overrides") or {}))
    model.load_state_dict(blob["model_state"])
    model.eval()
    scaler = LabelScaler().load_state_dict(meta["scaler"])
    return LoadedModel(model, meta, scaler)


def load_cds_checkpoint(path) -> LoadedModel:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    meta = blob["metadata"]
    model = MinimolCDS(meta["preset"], **(meta.get("overrides") or {}))
    model.load_state_dict(blob["model_state"])
    model.eval()
    return LoadedModel(model, meta, None)


def _base_payload() -> dict:
    return {
        "research_use_only": True,
        "disclaimer": RESEARCH_DISCLAIMER,
        "warnings": list(WARNINGS),
    }


def create_app(
    reg: LoadedModel | None = None,
    cds: LoadedModel | None = None,
    token: str | None = None,
    rate_limit_per_minute: int = 0,
    max_reg_length: int = 2000,
    max_cds_length: int = 8192,
) -> FastAPI:
    app = FastAPI(title="minimol", version=__version__)
    limiter = RateLimiter(rate_limit_per_minute)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        if request.url.path != "/health" and token:
            header = request.headers.get("authorization", "")
            if header != f"Bearer {token}":
                return JSONResponse({"error": "unauthorized", "request_id": request_id}, status_code=401)
        client = request.client.host if request.client else "local"
        if request.url.path != "/health" and not limiter.allow(client):
            return JSONResponse({"error": "rate_limited", "request_id": request_id}, status_code=429)
        LOGGER.info("request id=%s path=%s", request_id, request.url.path)
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "research_use_only": True,
            "auth_enabled": bool(token),
            "weights": {"reg": reg is not None, "cds": cds is not None},
        }

    @app.post("/v1/regulatory/predict")
    def regulatory(body: RegulatoryRequest):
        payload = _base_payload()
        payload.update({"assay": body.assay, "model_version": body.model, "request_warning": None})
        if len(body.sequence) > max_reg_length:
            payload.update({"abstain": True, "reason": "sequence_too_long", "predicted_log2fc": None})
            return payload
        if n_fraction(body.sequence) > 0.5:
            payload.update({"abstain": True, "reason": "too_many_n", "predicted_log2fc": None, "ood_warning": True})
            return payload
        if reg is None:
            payload.update({"abstain": True, "reason": "weights_not_loaded", "predicted_log2fc": None, "ood_warning": None})
            return payload
        encoded = one_hot_sequence(body.sequence).unsqueeze(0)
        with torch.no_grad():
            pred = reg.scaler.inverse_torch(reg.model(encoded)).squeeze(0)
        index = reg.meta["assay_names"].index(body.assay)
        length = len(body.sequence)
        ood = length < reg.meta.get("length_min", 1) or length > reg.meta.get("length_max", 10**9)
        payload.update(
            {
                "abstain": False,
                "predicted_log2fc": float(pred[index]),
                "prediction_interval": None,
                "ood_warning": bool(ood),
                "model_version": reg.meta.get("preset", body.model),
                "units": "log2FC",
                "calibration_version": "uncalibrated",
                "uncertainty": None,
                "is_out_of_distribution": bool(ood),
            }
        )
        return payload

    @app.post("/v1/cds/annotate")
    def cds_annotate(body: CDSRequest):
        payload = _base_payload()
        payload["coordinate_system"] = "0-based-half-open"
        if len(body.sequence) > max_cds_length:
            payload.update({"abstain": True, "reason": "sequence_too_long", "strand_probabilities": None})
            return payload
        if cds is None:
            payload.update(
                {
                    "abstain": True,
                    "reason": "weights_not_loaded",
                    "strand_probabilities": None,
                    "model_version": None,
                }
            )
            return payload
        encoded = one_hot_sequence(body.sequence).unsqueeze(0)
        with torch.no_grad():
            probs = torch.sigmoid(cds.model(encoded)).squeeze(0).cpu().tolist()
        payload.update(
            {
                "abstain": False,
                "model_version": cds.meta.get("preset", body.model),
                "strand_probabilities": {"+": probs[0], "-": probs[1]},
                "example_only": False,
                "note": "Per-base coding probabilities are not a GFF gene annotation.",
            }
        )
        return payload

    @app.post("/v1/predict")
    def predict(body: PredictRequest):
        outputs = []
        seen = {}
        for molecule in body.molecules:
            standardized = standardize_smiles(molecule.smiles)
            canonical = standardized.get("smiles_canonical")
            if not standardized["valid"]:
                outputs.append({"id": molecule.id, "error": "unparseable_smiles", "outputs": [], "ood_flag": True})
                continue
            if standardized["unsupported_element"] or standardized["metallocomplex"]:
                outputs.append({"id": molecule.id, "error": "unsupported_element", "outputs": [], "ood_flag": True})
                continue
            if canonical not in seen:
                seen[canonical] = properties_from_smiles(canonical)
            props = seen[canonical]
            answers = []
            for question in body.questions:
                key = ORACLE_ALIAS.get(question.endpoint, question.endpoint)
                if key not in DETERMINISTIC_PROPERTIES or props is None:
                    answers.append(
                        {
                            "question_id": question.id,
                            "abstain": True,
                            "reason": "unsupported_endpoint",
                            "value": None,
                            "probability": None,
                        }
                    )
                    continue
                value = float(props[key])
                if question.type == "regression":
                    answers.append(
                        {
                            "question_id": question.id,
                            "abstain": False,
                            "value": value,
                            "units": ORACLE_UNITS[key],
                            "probability": None,
                            "interval": None,
                            "uncertainty": None,
                            "calibration_version": "deterministic_oracle",
                            "endpoint_id": question.endpoint,
                            "is_out_of_distribution": False,
                            "provenance": "rdkit",
                        }
                    )
                elif question.type == "noul" and question.event and "gt" in question.event:
                    answers.append(
                        {
                            "question_id": question.id,
                            "abstain": False,
                            "probability": 1.0 if value > float(question.event["gt"]) else 0.0,
                            "value": None,
                            "calibration_version": "deterministic_oracle",
                            "endpoint_id": question.endpoint,
                            "note": "Deterministic comparison against an RDKit property, not a learned event probability.",
                        }
                    )
                else:
                    answers.append(
                        {
                            "question_id": question.id,
                            "abstain": True,
                            "reason": "weights_not_loaded",
                            "value": None,
                            "probability": None,
                        }
                    )
            outputs.append({"id": molecule.id, "outputs": answers, "ood_flag": False, "smiles_canonical": canonical})
        return {**_base_payload(), "model_version": body.model, "predictions": outputs}

    @app.post("/v1/rank")
    def rank(body: RankRequest):
        if body.objective not in DETERMINISTIC_PROPERTIES:
            return {**_base_payload(), "abstain": True, "reason": "unsupported_objective", "ranked": []}
        scored = []
        rejected = []
        for smiles in body.candidates:
            props = properties_from_smiles(smiles)
            if props is None:
                rejected.append({"smiles": smiles, "reason": "invalid"})
            else:
                scored.append({"smiles": smiles, "score": float(props[body.objective])})
        reverse = body.direction == "maximize"
        scored.sort(key=lambda item: item["score"], reverse=reverse)
        return {
            **_base_payload(),
            "abstain": False,
            "method": "rdkit_oracle",
            "learned_model": False,
            "objective": body.objective,
            "ranked": scored,
            "rejected": rejected,
        }

    return app
