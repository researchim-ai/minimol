"""Frozen-encoder decision pilot: train/development/calibration/report are separate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from minimol.data.reg_decisions import load_cache
from minimol.errors import MinimolError
from minimol.evaluation.reg_decide import (
    answer_losses,
    calibrate_temperature,
    combined_loss,
    decision_report,
    paired_group_ci,
)
from minimol.jsonutil import write_json
from minimol.models.blocks import count_parameters
from minimol.models.reg_decide import RegDecisionHead
from minimol.seeding import set_seed
from minimol.tracking import sha256_file
from minimol.training.checkpoint import load_checkpoint
from minimol.training.loop import fit, move_batch, resolve_device


def _tensors(values):
    return [torch.from_numpy(values[key]) for key in ('features', 'base_mu', 'y', 'mask')]


def train_reg_decide(config: dict, cache: Path, out: Path) -> dict:
    torch.set_num_threads(2)
    manifest, arrays = load_cache(cache)
    cfg = config.get('train', {})
    seed = int(cfg.get('seed', 11))
    set_seed(seed)
    device = resolve_device(cfg.get('device', 'auto'))
    hidden = int(config.get('model', {}).get('hidden', 128))
    head = RegDecisionHead(manifest['feature_dim'], len(manifest['assays']), hidden)
    head.feature_mean.copy_(torch.tensor(manifest['feature_mean']))
    head.feature_std.copy_(torch.tensor(manifest['feature_std']))
    train_data = TensorDataset(*_tensors(arrays['train']))
    dev_data = TensorDataset(*_tensors(arrays['development']))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_data, batch_size=int(cfg.get('batch_size', 256)), shuffle=True, generator=generator)
    dev_loader = DataLoader(dev_data, batch_size=4096, shuffle=False,
                            generator=torch.Generator().manual_seed(seed + 1))
    lr = float(cfg.get('lr', .001))
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=.001)

    def loss_fn(module, batch):
        mu, scale = module(batch[0], batch[1])
        loss = combined_loss(answer_losses(mu, scale, batch[2], batch[3]))
        mask = batch[3].float()
        auxiliary = (F.huber_loss(mu, batch[2], reduction='none') * mask).sum() / mask.sum().clamp(min=1)
        return loss + .05 * auxiliary

    def metric_fn(module, batches):
        total, count = 0., 0
        for batch in batches:
            batch = move_batch(batch, device)
            mu, scale = module(batch[0], batch[1])
            value = combined_loss(answer_losses(mu, scale, batch[2], batch[3]))
            total += float(value) * len(batch[0])
            count += len(batch[0])
        return -total / count

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {'task': 'reg_decide_head', 'data_sha256': sha256_file(Path(cache) / 'manifest.json'),
                'encoder_checkpoint_sha256': manifest['checkpoint_sha256'], 'seed': seed,
                'feature_dim': manifest['feature_dim'], 'hidden': hidden,
                'n_assays': len(manifest['assays']), 'config': config}
    write_json(out / 'config.json', config)
    result = fit(head, loader, dev_loader, optimizer, loss_fn, metric_fn,
                 epochs=int(cfg.get('epochs', 20)), patience=int(cfg.get('patience', 5)),
                 grad_clip=1., accumulation_steps=1, precision='fp32', device=device,
                 max_steps=cfg.get('max_steps'), checkpoint_dir=out, metadata=metadata,
                 base_lr=lr, resume_from=cfg.get('resume_from'))
    load_checkpoint(out / 'checkpoint_best.pt', head, map_location=device)
    head.eval()

    def predict(name):
        h, base, y, mask = _tensors(arrays[name])
        with torch.no_grad():
            mu, scale = head(h.to(device), base.to(device))
        return mu.cpu(), scale.cpu(), y, mask

    train = arrays['train']
    baseline_scale = []
    for index in range(len(manifest['assays'])):
        keep = train['mask'][:, index]
        residual = train['y'][keep, index] - train['base_mu'][keep, index]
        baseline_scale.append(max(.05, float(residual.std()) * np.sqrt(3) / np.pi))
    cmu, cscale, cy, cmask = predict('calibration')
    temperature = calibrate_temperature(cmu, cscale, cy, cmask)
    base_cmu = torch.from_numpy(arrays['calibration']['base_mu'])
    base_cscale = torch.tensor(baseline_scale)[None, :].expand_as(base_cmu)
    baseline_temperature = calibrate_temperature(base_cmu, base_cscale, cy, cmask)
    mu, scale, y, mask = predict('report')
    trained, per_object = decision_report(mu, scale, y, mask, temperature=temperature)
    baseline_mu = torch.from_numpy(arrays['report']['base_mu'])
    baseline_sigmas = torch.tensor(baseline_scale)[None, :].expand_as(baseline_mu)
    baseline, base_object = decision_report(baseline_mu, baseline_sigmas, y, mask, temperature=baseline_temperature)
    keep = mask.any(-1).numpy()
    interval = paired_group_ci((per_object - base_object)[keep], arrays['report']['group_id'][keep])
    choice_delta = trained['assay_choice']['accuracy'] - baseline['assay_choice']['accuracy']
    passed = interval['high'] < 0 and choice_delta >= -.01
    distance = np.sqrt(np.mean(((arrays['report']['features'] - np.array(manifest['feature_mean'])) /
                               np.array(manifest['feature_std']))**2, axis=1))
    report = {'task': 'reg_decide_pilot', 'seed': seed, 'head_parameters': count_parameters(head),
              'runtime_parameters': count_parameters(head) + manifest['encoder_metadata']['param_count'],
              'trained': trained, 'calibrated_regression_baseline': baseline,
              'temperature': temperature, 'baseline_temperature': baseline_temperature,
              'primary': 'object_macro_noul_brier_at_unseen_thresholds', 'paired_group_ci': interval,
              'assay_choice_accuracy_delta': choice_delta, 'decision': 'GO' if passed else 'NO-GO',
              'gate': 'CI high < 0 for Brier delta and assay-choice accuracy drop <= 1 percentage point',
              'n_report_objects': len(y), 'ood_fraction': float(np.mean(distance > manifest['ood_rms_z_threshold'])),
              'steps': result['steps'], 'history': result['history'], 'cache_sha256': metadata['data_sha256'],
              'encoder_checkpoint_sha256': manifest['checkpoint_sha256'], 'test_used': False,
              'limitation': manifest['encoder_selection_exposure']}
    parent = Path(manifest['parent_checkpoint'])
    if sha256_file(parent) != manifest['checkpoint_sha256']:
        raise MinimolError('parent checkpoint changed after feature extraction')
    parent_blob = torch.load(parent, map_location='cpu', weights_only=False)
    bundle = {'format_version': 1, 'encoder_metadata': manifest['encoder_metadata'],
              'encoder_state': parent_blob['model_state'], 'head_metadata': metadata,
              'head_state': {key: val.cpu() for key, val in head.state_dict().items()},
              'assays': manifest['assays'], 'temperature': temperature,
              'calibration_version': f'group-heldout-temperature-seed{seed}-v1',
              'calibration_scope': manifest['encoder_selection_exposure'],
              'baseline_temperature': baseline_temperature, 'baseline_scale': baseline_scale,
              'ood_rms_z_threshold': manifest['ood_rms_z_threshold'], 'gate': report['decision'],
              'encoder_checkpoint_sha256': manifest['checkpoint_sha256']}
    torch.save(bundle, out / 'decision_bundle.pt')
    np.savez_compressed(out / 'report_predictions.npz', mu=mu.numpy(), scale=scale.numpy(),
                        y=y.numpy(), mask=mask.numpy(), group_id=arrays['report']['group_id'],
                        sample_id=arrays['report']['sample_id'])
    write_json(out / 'metrics.json', report)
    write_json(out / 'calibration.json', {'temperature': temperature, 'baseline_temperature': baseline_temperature,
                                         'fit_partition': 'calibration', 'report_partition': 'report'})
    print(json.dumps({'seed': seed, 'decision': report['decision'], 'brier_delta_ci': interval,
                      'accuracy_delta': choice_delta}), flush=True)
    return report
