import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from minimol.errors import MinimolError
from minimol.seeding import set_seed
from minimol.training.loop import fit


def execute(path, *, stop=None, resume=None):
    set_seed(37)
    model = nn.Sequential(nn.Linear(2, 8), nn.Dropout(.3), nn.Linear(8, 1))
    data = TensorDataset(torch.arange(24).reshape(12, 2).float() / 24, torch.ones(12, 1))
    loader = DataLoader(data, batch_size=3, shuffle=True, generator=torch.Generator().manual_seed(19))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)

    def loss(module, batch):
        return (module(batch[0]) - batch[1]).square().mean() * (.9 + .1 * random.random() + .1 * np.random.rand())

    result = fit(model, loader, None, optimizer, loss, None, epochs=3, patience=0,
                 grad_clip=1., accumulation_steps=1, precision='fp32', device=torch.device('cpu'),
                 max_steps=None, checkpoint_dir=path, metadata={'task': 'test'}, base_lr=.003,
                 stop_after_epochs=stop, resume_from=resume)
    return model, result


def test_epoch_boundary_resume_matches_continuous_dropout_and_shuffle(tmp_path):
    full, result = execute(tmp_path / 'full')
    execute(tmp_path / 'split', stop=1)
    resumed, restored = execute(tmp_path / 'split', resume=tmp_path / 'split/checkpoint_last.pt')
    for a, b in zip(full.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert restored['history'] == result['history']
    assert (tmp_path / 'split/history.jsonl').read_text().count('\n') == 3
    assert 'completed' in (tmp_path / 'split/status.json').read_text()


def test_fresh_run_never_silently_overwrites_old_weights(tmp_path):
    execute(tmp_path / 'run', stop=1)
    with pytest.raises(MinimolError, match='already contains'):
        execute(tmp_path / 'run')
