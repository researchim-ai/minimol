"""Recover validation metrics and a decision baseline from a saved Reg checkpoint."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from minimol.data.normalize import LabelScaler, pivot_reg
from minimol.evaluation.decision_baseline import regression_decisions
from minimol.evaluation.metrics import prediction_scale, regression_report
from minimol.evaluation.reg_slices import slice_metrics
from minimol.jsonutil import write_json
from minimol.models.reg_cnn import MinimolReg
from minimol.tracking import sha256_file
from minimol.training.loop import move_batch, resolve_device
from minimol.training.train_reg import RegDataset, collate_reg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    torch.set_num_threads(2)
    start = time.perf_counter()
    blob = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    meta = blob['metadata']
    if meta.get('task') != 'reg':
        raise ValueError('expected a regulatory checkpoint')
    data_hash = sha256_file(args.data)
    if data_hash != meta['data_sha256']:
        raise ValueError('data hash does not match the checkpoint')
    print('Data hash matches. Reading validation only.', flush=True)
    frame = pd.read_parquet(args.data, filters=[('split', '==', 'validation')])
    assays = meta['assay_names']
    rows = pivot_reg(frame, assays)
    if not rows:
        raise ValueError('empty validation split')
    print(f'Validation sequences: {len(rows)}', flush=True)
    scaler = LabelScaler().load_state_dict(meta['scaler'])
    loader = DataLoader(RegDataset(rows, scaler, np.ones((len(rows), len(assays)))),
                        batch_size=128, shuffle=False, collate_fn=collate_reg)
    device = resolve_device(args.device)
    model = MinimolReg.from_metadata(meta).to(device).eval()
    model.load_state_dict(blob['model_state'])
    pred = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            batch = move_batch(batch, device)
            pred.append(scaler.inverse_torch(model(batch['x'], batch['pad_mask'])).float().cpu().numpy())
            if index % 100 == 0:
                print(f'Validation batch {index}/{len(loader)}', flush=True)
    prediction = np.concatenate(pred)
    y = np.stack([row['y'] for row in rows])
    mask = np.stack([row['mask'] for row in rows])
    args.out.mkdir(parents=True, exist_ok=True)
    values = {'sequence_hash': [row['sequence_hash'] for row in rows],
              'group_id': [row['group_id'] for row in rows]}
    for index, assay in enumerate(assays):
        values[f'{assay}_observed'] = y[:, index]
        values[f'{assay}_prediction'] = prediction[:, index]
        values[f'{assay}_valid'] = mask[:, index]
    pd.DataFrame(values).to_parquet(args.out / 'validation_predictions.parquet', index=False)
    metrics = regression_report(y, prediction, mask, assays)
    metrics['prediction_scale'] = prediction_scale(y, prediction, mask, assays)
    report = {
        'task': 'reg_recovered_evaluation', 'split': 'validation',
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha256_file(args.checkpoint),
        'data_sha256': data_hash, 'seed': meta['seed'],
        'head_kind': model.head_kind, 'padding_policy': model.padding_policy,
        'epoch_one_based': meta['epoch'] + 1, 'steps': meta['step'],
        'param_count': meta['param_count'], 'n_sequences': len(rows),
        'metrics': metrics,
        'decision_baselines': regression_decisions(y, prediction, mask, assays, train_means=scaler.mean),
        'slices': slice_metrics(rows, prediction, assays),
        'elapsed_s': time.perf_counter() - start, 'device': str(device),
        'training_completed': False,
        'note': 'Re-evaluation of a saved checkpoint from an incomplete run. Test was not scored.',
    }
    write_json(args.out / 'metrics.json', report)
    print(f"Recovered MAE={metrics['macro_mae']:.6f}, Spearman={metrics['macro_spearman']:.6f}", flush=True)
    print(args.out / 'metrics.json', flush=True)


if __name__ == '__main__':
    main()
