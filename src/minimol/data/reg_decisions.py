"""Frozen Reg feature cache with object-group partitions before question creation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from minimol.data.normalize import LabelScaler, pivot_reg
from minimol.errors import MinimolError
from minimol.jsonutil import write_json
from minimol.models.reg_cnn import MinimolReg
from minimol.tracking import sha256_file
from minimol.training.loop import move_batch, resolve_device
from minimol.training.train_reg import RegDataset, collate_reg


def group_key(raw: str) -> str:
    """Exact chr:position locus where present, otherwise the supplied group ID.

    This does not claim interval-overlap or homology clustering.
    """
    match = re.match(r'^(?:chr)?([0-9XY]+):(\d+)(?::|$)', str(raw))
    value = ':'.join(match.groups()) if match else str(raw)
    return hashlib.sha256(value.encode()).hexdigest()


def partition_group(key: str) -> str:
    bucket = int(key[:8], 16) % 10
    return 'development' if bucket < 2 else 'calibration' if bucket < 4 else 'report'


def _select(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    objects = frame.drop_duplicates('sequence_hash').sort_values(['decision_group', 'sequence_hash'])
    # Group-level cap never splits a group between partitions.
    keep = []
    count = 0
    for key, group in objects.groupby('decision_group', sort=False):
        if count >= limit:
            break
        keep.append(key)
        count += len(group)
    return frame.loc[frame.decision_group.isin(keep)].copy()


def prepare_cache(checkpoint: Path, data: Path, out: Path, *, train_limit=40000,
                  validation_limit=20000, device='auto') -> dict:
    out = Path(out)
    if (out / 'manifest.json').exists():
        raise MinimolError('feature cache already exists; use a new path or the existing cache')
    data_hash, checkpoint_hash = sha256_file(data), sha256_file(checkpoint)
    blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
    meta = blob['metadata']
    if meta.get('task') != 'reg' or meta['data_sha256'] != data_hash:
        raise MinimolError('Reg checkpoint and data hashes must match')
    torch.set_num_threads(2)
    device = resolve_device(device)
    model = MinimolReg.from_metadata(meta).to(device).eval()
    model.load_state_dict(blob['model_state'])
    print('Reading train/validation only; test is excluded.', flush=True)
    frame = pd.read_parquet(data, filters=[('split', 'in', ['train', 'validation'])])
    mapping = {str(group): group_key(str(group)) for group in frame.group_id.unique()}
    frame['decision_group'] = frame.group_id.astype(str).map(mapping)
    cross = frame.groupby('decision_group').split.nunique()
    if (cross > 1).any():
        raise MinimolError('source object groups overlap train and validation')
    train = _select(frame.loc[frame.split == 'train'], train_limit)
    validation = frame.loc[frame.split == 'validation'].copy()
    validation['decision_partition'] = validation.decision_group.map(partition_group)
    parts = {'train': train}
    for name, fraction in [('development', .2), ('calibration', .2), ('report', .6)]:
        parts[name] = _select(validation.loc[validation.decision_partition == name],
                              max(1, int(validation_limit * fraction)))
    out.mkdir(parents=True, exist_ok=True)
    partitions = {}
    all_groups: dict[str, set] = {}
    scaler = LabelScaler().load_state_dict(meta['scaler'])
    for name, part in parts.items():
        rows = pivot_reg(part, meta['assay_names'])
        if len(rows) < 4:
            raise MinimolError(f'{name} has fewer than 4 objects; increase data limits')
        lookup = dict(zip(part.sequence_hash, part.decision_group, strict=False))
        groups = np.asarray([lookup[row['sequence_hash']] for row in rows])
        all_groups[name] = set(groups.tolist())
        loader = DataLoader(RegDataset(rows, scaler, np.ones((len(rows), meta['n_assays']))),
                            batch_size=128, shuffle=False, collate_fn=collate_reg)
        features, means = [], []
        with torch.inference_mode():
            for batch in loader:
                batch = move_batch(batch, device)
                hidden = model.encode(batch['x'], batch['pad_mask'])
                pred = model.out(model.head_activation(hidden[:, None, :] + model.assay_emb.weight[None, :, :])).squeeze(-1)
                features.append(hidden.float().cpu().numpy())
                means.append(scaler.inverse_torch(pred).float().cpu().numpy())
        path = out / f'{name}.npz'
        np.savez_compressed(path, features=np.concatenate(features), base_mu=np.concatenate(means),
                            y=np.stack([row['y'] for row in rows]).astype('float32'),
                            mask=np.stack([row['mask'] for row in rows]),
                            se=np.stack([row['se'] for row in rows]).astype('float32'),
                            group_id=groups,
                            sample_id=np.asarray([row['sequence_hash'] for row in rows]))
        partitions[name] = {'n_objects': len(rows), 'n_groups': len(all_groups[name]),
                            'file': path.name, 'sha256': sha256_file(path)}
        print(f'{name}: {len(rows)} objects / {len(all_groups[name])} groups', flush=True)
    for name, groups in all_groups.items():
        for other, other_groups in all_groups.items():
            if name != other and groups & other_groups:
                raise MinimolError(f'groups cross partitions: {name}, {other}')
    h = np.load(out / 'train.npz')['features']
    mu, std = h.mean(0), h.std(0).clip(1e-3)
    distance = np.sqrt(np.mean(((h - mu) / std) ** 2, axis=1))
    manifest = {
        'task': 'reg_decide_features', 'partitions': partitions,
        'feature_dim': int(h.shape[1]), 'encoder_metadata': meta,
        'parent_checkpoint': str(Path(checkpoint).resolve()), 'checkpoint_sha256': checkpoint_hash,
        'data_sha256': data_hash, 'assays': meta['assay_names'],
        'feature_mean': mu.tolist(), 'feature_std': std.tolist(),
        'ood_rms_z_threshold': float(np.quantile(distance, .995)),
        'group_rule': 'exact chr:position where parseable, otherwise supplied group_id; SHA256',
        'encoder_selection_exposure': 'parent encoder epoch was selected on original validation; report is exploratory',
        'test_used': False,
    }
    write_json(out / 'manifest.json', manifest)
    sources = out / 'source_snapshot'
    sources.mkdir(exist_ok=True)
    for module in ('models/reg_cnn.py', 'models/blocks.py', 'data/reg_decisions.py'):
        source = Path(__file__).parents[1] / module
        (sources / module.replace('/', '_')).write_bytes(source.read_bytes())
    return manifest


def load_cache(path: Path) -> tuple[dict, dict]:
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    arrays = {}
    for name, part in manifest['partitions'].items():
        file = path / part['file']
        if sha256_file(file) != part['sha256']:
            raise MinimolError(f'feature cache hash mismatch: {name}')
        with np.load(file, allow_pickle=False) as values:
            arrays[name] = {key: values[key] for key in values.files}
    return manifest, arrays
