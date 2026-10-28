"""Request and response schemas. Regression values are not probabilities."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from minimol.data.sequences import normalize_sequence

ASSAYS = ("K562", "HepG2", "SK-N-SH")


class RegulatoryRequest(BaseModel):
    sequence: str = Field(min_length=1, max_length=8192)
    assay: Literal["K562", "HepG2", "SK-N-SH"]
    model: str = "minimol-reg-10m-v1"

    @field_validator("sequence")
    @classmethod
    def check_sequence(cls, value: str) -> str:
        return normalize_sequence(value)


class CDSRequest(BaseModel):
    sequence: str = Field(min_length=1, max_length=8192)
    model: str = "minimol-cds-10m-v1"

    @field_validator("sequence")
    @classmethod
    def check_sequence(cls, value: str) -> str:
        return normalize_sequence(value)


class MoleculeIn(BaseModel):
    id: str
    smiles: str


class QuestionIn(BaseModel):
    id: str
    type: Literal["regression", "noul", "choice", "score"]
    endpoint: str
    event: dict | None = None
    options: list[str] | None = None
    bins: list[float] | None = None


class PredictRequest(BaseModel):
    model: str = "minimol-admet-15m@0.1.0"
    molecules: list[MoleculeIn]
    questions: list[QuestionIn]


class RankRequest(BaseModel):
    candidates: list[str]
    objective: str
    direction: Literal["maximize", "minimize"] = "maximize"
    model: str = "rdkit-oracle"
