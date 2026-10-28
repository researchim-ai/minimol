"""Losses for regression, per-base CDS, and Bernoulli distillation.

T^2 scaling of Bernoulli KL is optional. It is not applied unless scale_by_t2 is set.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _weight(mask: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
    weight = mask.to(dtype=torch.float32)
    if sample_weight is not None:
        weight = weight * sample_weight
    return weight


def masked_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.huber_loss(pred, target, reduction="none", delta=delta)
    weight = _weight(mask, sample_weight)
    return (loss * weight).sum() / weight.sum().clamp(min=1.0)


def masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight
    )
    weight = mask.to(dtype=loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp(min=1.0)


def bernoulli_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    return p * (p.log() - q.log()) + (1.0 - p) * ((1.0 - p).log() - (1.0 - q).log())


def soft_bernoulli_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    scale_by_t2: bool = False,
) -> torch.Tensor:
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher = torch.sigmoid(teacher_logits / temperature)
    student = torch.sigmoid(student_logits / temperature)
    kl = bernoulli_kl(teacher, student)
    scale = temperature * temperature if scale_by_t2 else 1.0
    weight = mask.to(dtype=kl.dtype)
    return scale * (kl * weight).sum() / weight.sum().clamp(min=1.0)


def cosine_feature_loss(student_h: torch.Tensor, teacher_h: torch.Tensor) -> torch.Tensor:
    student = F.normalize(student_h, dim=-1)
    teacher = F.normalize(teacher_h.detach(), dim=-1)
    return (1.0 - (student * teacher).sum(dim=-1)).mean()


def gaussian_nll(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    log_var = log_var.clamp(-8.0, 8.0)
    inv = torch.exp(-log_var)
    nll = 0.5 * (inv * (target - mean) ** 2 + log_var)
    weight = mask.to(dtype=nll.dtype)
    return (nll * weight).sum() / weight.sum().clamp(min=1.0)


def inverse_variance_weights(se, cap_quantile: float = 0.9, eps: float = 1e-3) -> np.ndarray:
    """Cap measurement weights so a few tiny standard errors cannot dominate."""

    uncertainty = np.asarray(se, dtype=np.float64)
    raw = 1.0 / np.maximum(uncertainty, eps) ** 2
    finite = np.isfinite(raw)
    weights = np.zeros_like(raw)
    if finite.any():
        cap = float(np.quantile(raw[finite], cap_quantile))
        weights[finite] = np.minimum(raw[finite], cap)
    return weights


def boundary_focus_mask(target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """1 on bases next to a 0/1 transition, else 0. Invalid bases stay 0."""

    hard = (target > 0.5).to(dtype=torch.float32)
    change = torch.zeros_like(hard)
    change[..., 1:] = (hard[..., 1:] != hard[..., :-1]).to(dtype=torch.float32)
    change[..., :-1] = torch.maximum(
        change[..., :-1],
        (hard[..., 1:] != hard[..., :-1]).to(dtype=torch.float32),
    )
    return change * valid.to(dtype=torch.float32)


def rc_strand_consistency_loss(logits: torch.Tensor, logits_rc: torch.Tensor) -> torch.Tensor:
    """p_plus(RC(x)) should match reverse(p_minus(x)), and the converse.

    The original strand is detached so the penalty pulls the reverse-complement
    view toward the swapped coordinates instead of collapsing both to a constant.
    """

    probs = torch.sigmoid(logits)
    probs_rc = torch.sigmoid(logits_rc)
    target_plus = torch.flip(probs[:, 1], dims=(-1,)).detach()
    target_minus = torch.flip(probs[:, 0], dims=(-1,)).detach()
    return F.mse_loss(probs_rc[:, 0], target_plus) + F.mse_loss(probs_rc[:, 1], target_minus)


def cds_loss(
    student_logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    teacher_logits: torch.Tensor | None = None,
    lambda_gt: float = 1.0,
    lambda_kd: float = 0.0,
    temperature: float = 1.0,
    scale_by_t2: bool = False,
    lambda_boundary: float = 0.0,
    pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    bce = masked_bce_with_logits(student_logits, target, mask, pos_weight=pos_weight)
    total = lambda_gt * bce
    parts = {"bce": float(bce.detach())}
    if teacher_logits is not None and lambda_kd:
        kd = soft_bernoulli_kd(
            student_logits,
            teacher_logits,
            mask,
            temperature=temperature,
            scale_by_t2=scale_by_t2,
        )
        total = total + lambda_kd * kd
        parts["kd"] = float(kd.detach())
    if lambda_boundary:
        focus = boundary_focus_mask(target, mask)
        extra = masked_bce_with_logits(student_logits, target, focus, pos_weight=pos_weight)
        total = total + lambda_boundary * extra
        parts["boundary"] = float(extra.detach())
    return total, parts


def multitask_supervised_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    kinds: dict[str, str],
    delta: float = 1.0,
) -> torch.Tensor:
    total = predictions[next(iter(predictions))].new_zeros(())
    denom = total.new_zeros(())
    for name, kind in kinds.items():
        if name not in predictions:
            continue
        pred = predictions[name]
        target = targets[name]
        mask = masks[name].to(dtype=pred.dtype)
        if kind == "binary":
            loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        elif kind == "regression":
            loss = F.huber_loss(pred, target, reduction="none", delta=delta)
        else:
            raise ValueError(f"unsupported task kind: {kind}")
        total = total + (loss * mask).sum()
        denom = denom + mask.sum()
    return total / denom.clamp(min=1.0)
