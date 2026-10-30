"""CPU SDK for typed regulatory questions and self-contained decision bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from minimol.data.normalize import LabelScaler
from minimol.data.sequences import n_fraction, normalize_sequence, one_hot_sequence
from minimol.models.reg_cnn import MinimolReg
from minimol.models.reg_decide import RegDecisionHead
from minimol.tracking import sha256_file


class DnaCandidate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(min_length=1, max_length=128)
    sequence: str = Field(min_length=1, max_length=8192)


class RegQuery(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(min_length=1, max_length=128)
    kind: Literal['noul', 'score', 'candidate_choice', 'assay_choice']
    candidate_id: str | None = None
    candidate_ids: list[str] | None = Field(default=None, min_length=2, max_length=16)
    assay: str | None = None
    assays: list[str] | None = Field(default=None, min_length=2, max_length=3)
    threshold: float | None = Field(default=None, ge=-.5, le=2, allow_inf_nan=False)
    bins: list[float] | None = Field(default=None, min_length=1, max_length=9)
    direction: Literal['maximize', 'minimize'] = 'maximize'

    @model_validator(mode='after')
    def validate_question(self):
        if self.kind in {'noul', 'score'} and (not self.candidate_id or not self.assay):
            raise ValueError('noul/score require candidate_id and assay')
        if self.kind == 'noul' and self.threshold is None:
            raise ValueError('noul requires a threshold in log2FC')
        if self.kind == 'score':
            if not self.bins or any(not np.isfinite(v) or v < -.5 or v > 2 for v in self.bins):
                raise ValueError('score bins must be finite and within [-0.5, 2] log2FC')
            if self.bins != sorted(set(self.bins)):
                raise ValueError('score bins must be strictly increasing')
        if self.kind == 'candidate_choice' and (not self.assay or not self.candidate_ids):
            raise ValueError('candidate_choice requires candidate_ids and assay')
        if self.kind == 'assay_choice' and (not self.candidate_id or not self.assays):
            raise ValueError('assay_choice requires candidate_id and assays')
        options = self.assays if self.kind == 'assay_choice' else self.candidate_ids
        if options is not None and len(set(options)) != len(options):
            raise ValueError('options must be unique')
        return self


class RegDecideRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    candidates: list[DnaCandidate] = Field(min_length=1, max_length=16)
    questions: list[RegQuery] = Field(min_length=1, max_length=64)
    minimum_selection_probability: float = Field(default=.6, ge=.5, le=1)

    @model_validator(mode='after')
    def unique_ids(self):
        if len({x.id for x in self.candidates}) != len(self.candidates):
            raise ValueError('candidate IDs must be unique')
        if len({x.id for x in self.questions}) != len(self.questions):
            raise ValueError('question IDs must be unique')
        return self


class RegDecider:
    def __init__(self, bundle: dict, *, model_version='local', backend='calibrated_regression'):
        if bundle.get('format_version') != 1:
            raise ValueError('unsupported decision bundle version')
        if backend not in {'trained', 'calibrated_regression'}:
            raise ValueError('unknown decision backend')
        self.bundle, self.backend, self.model_version = bundle, backend, model_version
        self.encoder = MinimolReg.from_metadata(bundle['encoder_metadata']).eval()
        self.encoder.load_state_dict(bundle['encoder_state'])
        spec = bundle['head_metadata']
        self.head = RegDecisionHead(spec['feature_dim'], spec['n_assays'], spec['hidden']).eval()
        self.head.load_state_dict(bundle['head_state'])
        self.scaler = LabelScaler().load_state_dict(bundle['encoder_metadata']['scaler'])
        self.assays = list(bundle['assays'])
        self.temperature = bundle['temperature'] if backend == 'trained' else bundle['baseline_temperature']

    @classmethod
    def load(cls, path, *, backend='calibrated_regression'):
        path = Path(path)
        if path.is_dir():
            path = path / 'decision_bundle.pt'
        return cls(torch.load(path, map_location='cpu', weights_only=False),
                   model_version='reg-decide-' + sha256_file(path)[:12], backend=backend)

    def predict(self, request: RegDecideRequest | dict) -> dict:
        request = request if isinstance(request, RegDecideRequest) else RegDecideRequest.model_validate(request)
        valid, rejected = [], {}
        meta = self.bundle['encoder_metadata']
        for candidate in request.candidates:
            try:
                seq = normalize_sequence(candidate.sequence)
            except ValueError:
                rejected[candidate.id] = 'invalid_sequence'
                continue
            if not meta['length_min'] <= len(seq) <= meta['length_max']:
                rejected[candidate.id] = 'sequence_length_out_of_domain'
            elif n_fraction(seq) > .2:
                rejected[candidate.id] = 'too_many_unknown_bases'
            else:
                valid.append((candidate.id, seq))
        distributions = {}
        if valid:
            width = max(len(seq) for _, seq in valid)
            x, mask = torch.zeros(len(valid), 5, width), torch.zeros(len(valid), width, dtype=torch.bool)
            for index, (_, seq) in enumerate(valid):
                x[index, :, :len(seq)] = one_hot_sequence(seq)
                mask[index, :len(seq)] = True
            with torch.inference_mode():
                hidden = self.encoder.encode(x, mask)
                raw = self.encoder.out(self.encoder.head_activation(hidden[:, None, :] + self.encoder.assay_emb.weight[None, :, :])).squeeze(-1)
                base = self.scaler.inverse_torch(raw)
                if self.backend == 'calibrated_regression':
                    mu = base
                    scale = torch.tensor(self.bundle['baseline_scale'])[None, :].expand_as(base)
                else:
                    mu, scale = self.head(hidden, base)
                distance = torch.sqrt(((hidden - self.head.feature_mean) / self.head.feature_std).square().mean(-1))
                for index, (identifier, _) in enumerate(valid):
                    if float(distance[index]) > self.bundle['ood_rms_z_threshold']:
                        rejected[identifier] = 'embedding_out_of_domain'
                    distributions[identifier] = (mu[index], scale[index])
        outputs = []
        for question in request.questions:
            required = question.candidate_ids if question.kind == 'candidate_choice' else [question.candidate_id]
            assay_options = question.assays if question.kind == 'assay_choice' else [question.assay]
            reason = next((rejected.get(identifier, 'unknown_candidate') for identifier in required
                           if identifier in rejected or identifier not in distributions), None)
            if any(name not in self.assays for name in assay_options):
                reason = 'unsupported_assay'
            output = {'question_id': question.id, 'kind': question.kind, 'abstain': True, 'reason': reason}
            if reason:
                outputs.append(output)
                continue
            direction = 1 if question.direction == 'maximize' else -1
            if question.kind in {'noul', 'score'}:
                mu, scale = distributions[question.candidate_id]
                index = self.assays.index(question.assay)
                if question.kind == 'noul':
                    probability = float(torch.sigmoid((mu[index] - question.threshold) / (scale[index] * self.temperature)))
                    output.update(probability=probability, answer=probability >= .5)
                    acceptance = max(probability, 1-probability)
                else:
                    p = self.head.bin_probabilities(mu[index], scale[index], torch.tensor(question.bins), self.temperature)
                    labels = [f'(-inf,{question.bins[0]}]'] + [f'({a},{b}]' for a, b in zip(question.bins[:-1], question.bins[1:], strict=True)] + [f'({question.bins[-1]},inf)']
                    output.update(distribution=p.tolist(), options=labels, answer=labels[int(p.argmax())])
                    acceptance = float(p.max())
            elif question.kind == 'assay_choice':
                mu, scale = distributions[question.candidate_id]
                indices = [self.assays.index(name) for name in question.assays]
                p = torch.softmax(self.head.assay_logits(mu[indices], scale[indices], direction, self.temperature), -1)
                output.update(distribution=p.tolist(), options=question.assays, answer=question.assays[int(p.argmax())])
                acceptance = float(p.max())
            else:
                index = self.assays.index(question.assay)
                mu = torch.stack([distributions[name][0][index] for name in question.candidate_ids])
                scale = torch.stack([distributions[name][1][index] for name in question.candidate_ids])
                if len(mu) == 2:
                    logit = direction * self.head.pair_logits(mu[0], scale[0], mu[1], scale[1], self.temperature)
                    first = torch.sigmoid(logit)
                    p = torch.stack([first, 1-first])
                else:
                    p = torch.softmax(direction * mu / (scale.mean() * self.temperature), -1)
                output.update(distribution=p.tolist(), options=question.candidate_ids, answer=question.candidate_ids[int(p.argmax())])
                acceptance = float(p.max())
            output.update(selection_probability=acceptance,
                          abstain=acceptance < request.minimum_selection_probability,
                          reason='below_selection_probability_policy' if acceptance < request.minimum_selection_probability else None)
            if question.kind == 'candidate_choice' and len(question.candidate_ids) > 2:
                output.update(abstain=True, reason='candidate_sets_larger_than_two_not_validated')
            outputs.append(output)
        return {'model_version': self.model_version, 'backend': self.backend, 'research_use_only': True,
                'learned_decision_head_active': self.backend == 'trained',
                'calibration_version': self.bundle['calibration_version'],
                'calibration_scope': self.bundle['calibration_scope'], 'head_seed_gate': self.bundle['gate'],
                'answers': outputs, 'encoded_objects': len(valid)}
