"""Aggregate all declared seeds; never silently publish only the winning seed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from minimol.jsonutil import write_json


def summarize_decide_runs(runs: list[Path], out: Path) -> dict:
    reports = [json.loads((Path(run) / 'metrics.json').read_text()) for run in runs]
    if len({report['cache_sha256'] for report in reports}) != 1:
        raise ValueError('seed runs must share the same feature cache')
    seeds = [report['seed'] for report in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError('seed runs must be distinct')
    values = [report['trained']['noul']['object_macro_brier'] for report in reports]
    baseline = [report['calibrated_regression_baseline']['noul']['object_macro_brier'] for report in reports]
    report = {
        'task': 'reg_decide_seed_summary', 'seeds': seeds, 'n_seeds': len(seeds),
        'primary': reports[0]['primary'], 'trained_brier_mean': float(np.mean(values)),
        'trained_brier_std': float(np.std(values, ddof=1)) if len(values) > 1 else None,
        'baseline_brier_mean': float(np.mean(baseline)),
        'mean_brier_delta': float(np.mean(np.array(values)-baseline)),
        'seed_decisions': {str(item['seed']): item['decision'] for item in reports},
        'decision': 'GO' if len(seeds) >= 3 and all(item['decision'] == 'GO' for item in reports) else 'INCONCLUSIVE',
        'recommended_backend': 'trained' if len(seeds) >= 3 and all(item['decision'] == 'GO' for item in reports) else 'calibrated_regression',
        'scope': 'three head seeds over one frozen parent encoder, exploratory development report',
        'test_used': False,
        'runs': [str(run) for run in runs],
        'note': 'No best-seed selection. Repeated questions and correlated seed results are not independent biological samples.',
    }
    # A scalar mean over seed scores is not an ensemble prediction.
    write_json(out, report)
    return report


def question_controls(cache: Path, run: Path) -> dict:
    """Question-only frequencies fit on train, and a deliberately query-blind baseline."""
    import torch

    from minimol.data.reg_decisions import load_cache
    from minimol.evaluation.reg_decide import REPORT_THRESHOLDS

    _, arrays = load_cache(cache)
    train, report = arrays['train'], arrays['report']
    bundle = torch.load(Path(run) / 'decision_bundle.pt', map_location='cpu', weights_only=False)
    thresholds = np.asarray(REPORT_THRESHOLDS)
    labels = report['y'][..., None] > thresholds
    mask = np.broadcast_to(report['mask'][..., None], labels.shape)
    priors = np.stack([(train['y'][train['mask'][:, col], col, None] > thresholds).mean(0)
                       for col in range(train['y'].shape[1])])
    blind_p = 1 / (1 + np.exp(-(report['base_mu'] - .5) /
                             (np.asarray(bundle['baseline_scale'])[None, :] * bundle['baseline_temperature'])))
    return {
        'question_only_train_frequency_brier': float(((priors - labels)**2)[mask].mean()),
        'ignore_numeric_threshold_brier': float(((blind_p[..., None] - labels)**2)[mask].mean()),
        'scope': 'same report labels and thresholds; priors use train labels only',
    }
