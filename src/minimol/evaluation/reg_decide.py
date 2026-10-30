"""Proper-score evaluation, held-out temperature calibration, and selective risk."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from minimol.evaluation.metrics import auprc, auroc, expected_calibration_error
from minimol.models.reg_decide import RegDecisionHead as Head

TRAIN_THRESHOLDS = (-.5, 0., .5, 1., 1.5, 2.)
REPORT_THRESHOLDS = (.25, .75, 1.25, 1.75)
BINS = (0., .5, 1., 2.)


def _masked(values, mask):
    weight = mask.to(values.dtype).expand_as(values)
    return (values * weight).sum() / weight.sum().clamp(min=1)


def answer_losses(mu, scale, y, mask, *, temperature=1.0, thresholds=TRAIN_THRESHOLDS):
    thresholds = mu.new_tensor(thresholds)
    logits = Head.event_logits(mu, scale, thresholds, temperature)
    target = (y[..., None] > thresholds).to(mu.dtype)
    event = _masked(F.binary_cross_entropy_with_logits(logits, target, reduction='none'), mask[..., None])
    bins = mu.new_tensor(BINS)
    probabilities = Head.bin_probabilities(mu, scale, bins, temperature)
    labels = (y[..., None] > bins).sum(-1)
    ordinal = _masked(-probabilities.gather(-1, labels[..., None]).squeeze(-1).log(), mask)
    complete = mask.all(-1)
    choices = []
    for direction in (1, -1):
        logits = Head.assay_logits(mu, scale, direction, temperature)
        labels = (direction * y).argmax(-1)
        choices.append(_masked(F.cross_entropy(logits, labels, reduction='none'), complete))
    other = torch.arange(len(y), device=y.device).roll(1)
    pair_logits = Head.pair_logits(mu, scale, mu[other], scale[other], temperature)
    pair_mask = mask & mask[other] & (y != y[other])
    pair = _masked(F.binary_cross_entropy_with_logits(pair_logits, (y > y[other]).float(), reduction='none'), pair_mask)
    return {'noul': event, 'score': ordinal, 'assay_choice': sum(choices) / 2, 'pair_choice': pair}


def combined_loss(parts):
    return parts['noul'] + .3 * parts['score'] + .5 * parts['assay_choice'] + .5 * parts['pair_choice']


def calibrate_temperature(mu, scale, y, mask, steps=120):
    """Single shared scale multiplier; monotonicity and argmax choices are retained."""
    mu, scale, y, mask = [value.detach().cpu() for value in (mu, scale, y, mask)]
    log_t = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([log_t], lr=.04)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = combined_loss(answer_losses(mu, scale, y, mask, temperature=log_t.clamp(-3, 3).exp()))
        loss.backward()
        optimizer.step()
    return float(log_t.detach().clamp(-3, 3).exp())


def risk_coverage(correct, confidence):
    order = np.argsort(-np.asarray(confidence), kind='stable')
    truth = np.asarray(correct)[order]
    return {str(fraction): {'n': max(1, int(len(order) * fraction)),
                           'risk': float(1 - truth[:max(1, int(len(order) * fraction))].mean())}
            for fraction in (1., .9, .75, .5)} if len(order) else {}


def decision_report(mu, scale, y, mask, *, temperature=1.0):
    thresholds = mu.new_tensor(REPORT_THRESHOLDS)
    with torch.no_grad():
        probabilities = torch.sigmoid(Head.event_logits(mu, scale, thresholds, temperature))
        labels = (y[..., None] > thresholds).float()
        selected = mask[..., None].expand_as(probabilities)
        p = probabilities[selected].cpu().numpy()
        target = labels[selected].cpu().numpy()
        error = (probabilities - labels).square()
        object_count = selected.sum(dim=(1, 2)).clamp(min=1)
        per_object_brier = ((error * selected).sum(dim=(1, 2)) / object_count).cpu().numpy()
        below = Head.bin_probabilities(mu, scale, mu.new_tensor(BINS), temperature)
        ordinal_target = (y[..., None] > y.new_tensor(BINS)).sum(-1)
        chosen = mask
        score_p = below[chosen]
        score_y = ordinal_target[chosen]
        score = {'accuracy': float((score_p.argmax(-1) == score_y).float().mean()),
                 'nll': float(-score_p.gather(-1, score_y[:, None]).log().mean())}
        complete = mask.all(-1)
        cp = torch.softmax(Head.assay_logits(mu, scale, temperature=temperature), -1)[complete]
        cy = y[complete].argmax(-1)
        cc = (cp.argmax(-1) == cy).cpu().numpy()
        choice = {'n': int(complete.sum()), 'accuracy': float(np.mean(cc)) if len(cc) else None,
                  'nll': float(-cp.gather(-1, cy[:, None]).log().mean()) if len(cc) else None,
                  'risk_coverage': risk_coverage(cc, cp.max(-1).values.cpu().numpy())}
        pairwise = []
        for column in range(y.shape[1]):
            ids = torch.where(mask[:, column])[0]
            ids = ids[torch.randperm(len(ids), generator=torch.Generator().manual_seed(20261030))]
            ids = ids[:len(ids) // 2 * 2].reshape(-1, 2)
            a, b = ids[:, 0], ids[:, 1]
            differences = y[a, column] - y[b, column]
            lp = Head.pair_logits(mu[a, column], scale[a, column], mu[b, column], scale[b, column], temperature)
            keep = differences != 0
            correct = ((lp > 0) == (differences > 0))[keep]
            pairwise.append({'n': int(keep.sum()), 'accuracy': float(correct.float().mean()) if len(correct) else None})
        noul = {'n': int(len(p)), 'brier': float(np.mean((p-target)**2)),
                'object_macro_brier': float(per_object_brier[mask.any(-1).cpu().numpy()].mean()),
                'nll': float(np.mean(-(target * np.log(np.clip(p, 1e-7, 1)) + (1-target)*np.log(np.clip(1-p, 1e-7, 1))))),
                'ece': expected_calibration_error(p, target), 'auroc': auroc(target, p),
                'average_precision': auprc(target, p),
                'accuracy': float(np.mean((p >= .5) == target)),
                'risk_coverage': risk_coverage((p >= .5) == target, np.maximum(p, 1-p))}
    return {'noul': noul, 'score': score, 'assay_choice': choice, 'pair_choice_per_assay': pairwise,
            'report_thresholds': list(REPORT_THRESHOLDS)}, per_object_brier


def paired_group_ci(differences, groups, n_boot=300, seed=20261030):
    values = np.asarray(differences, dtype=float)
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        ids = rng.integers(0, len(unique), size=len(unique))
        draws.append(float(sums[ids].sum() / counts[ids].sum()))
    return {'mean_difference': float(values.mean()), 'low': float(np.quantile(draws, .025)),
            'high': float(np.quantile(draws, .975)), 'n_groups': len(unique),
            'n_boot': n_boot, 'direction': 'negative favours trained head',
            'scope': 'grouped exploratory development interval; parent encoder used original validation for selection'}
