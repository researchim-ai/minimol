import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from minimol.models.reg_cnn import MinimolReg
from minimol.training.loop import fit


def test_assay_interaction_can_reverse_sequence_order():
    model = MinimolReg('1m', n_assays=2, channels=8, n_blocks=1, dropout=0)
    # Isolate the head with controlled encoder outputs.
    model.encode = lambda x, pad_mask=None: x
    with torch.no_grad():
        model.assay_emb.weight.zero_()
        model.assay_emb.weight[0, 0] = -1.0
        model.assay_emb.weight[1, 0] = 2.0
        model.out.weight.zero_()
        model.out.weight[0, 0] = 1.0
        model.out.bias.zero_()
    features = torch.zeros(2, 8)
    features[:, 0] = torch.tensor([-1.0, 0.0])
    predictions = model(features)
    assert predictions[0, 0] > predictions[1, 0]
    assert predictions[0, 1] < predictions[1, 1]
    for assay in range(2):
        assert torch.allclose(model.forward_assay(features, torch.full((2,), assay)), predictions[:, assay])


def test_old_metadata_preserves_linear_head():
    model = MinimolReg.from_metadata({'preset': '1m', 'n_assays': 2, 'overrides': {'channels': 8, 'n_blocks': 1}})
    assert model.head_kind == 'legacy_linear'
    assert isinstance(model.head_activation, nn.Identity)


def test_accumulation_flushes_and_normalizes_partial_group(tmp_path):
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    loader = DataLoader(TensorDataset(torch.ones(3, 1), torch.ones(3, 1)), batch_size=1)
    result = fit(
        model, loader, None, torch.optim.SGD(model.parameters(), lr=0.1),
        lambda module, batch: (module(batch[0]) - batch[1]).square().mean(),
        None, epochs=1, patience=0, grad_clip=0, accumulation_steps=2,
        precision='fp32', device=torch.device('cpu'), max_steps=None,
        checkpoint_dir=tmp_path, metadata={}, base_lr=0.1,
    )
    # First update w=0.2; cosine sets lr=0.05; final single batch gives w=0.28.
    assert result['steps'] == 2
    assert abs(model.weight.item() - 0.28) < 1e-6


def test_tdc_pr_auc_uses_average_precision_direction():
    from minimol.training.train_admet import tdc_selection_value

    assert tdc_selection_value({"auprc": 0.4}, "pr-auc") == 0.4
    assert tdc_selection_value({"mae": 0.2}, "mae") == -0.2
    assert tdc_selection_value({"spearman": 0.3}, "spearman") == 0.3


def test_shared_scaffold_is_not_called_a_duplicate_label():
    import pandas as pd

    from minimol.data.split_audit import cross_split_identity

    frame = pd.DataFrame(
        {
            "inchikey": ["AAA", "AAA", "BBB"],
            "scaffold_id": ["s", "s", "s"],
            "split": ["train", "test", "validation"],
        }
    )
    report = cross_split_identity(frame)
    assert report["inchikey_cross_split"][0]["inchikey"] == "AAA"
    assert report["scaffold_cross_split"]
    assert "not a duplicated measurement" in report["note"]


def test_selection_keeps_classification_when_regression_is_present():
    from minimol.training.train_admet import admet_selection_value

    both = admet_selection_value([1.0], [0.9])
    regression_only = admet_selection_value([1.0], [])
    assert both == (-1.0 + 0.9) / 2
    assert regression_only == -1.0
    assert both > regression_only


def test_prediction_scale_reports_shift():
    import numpy as np

    from minimol.evaluation.metrics import prediction_scale

    labels = np.array([[0.0, 2.0], [2.0, 4.0]])
    guesses = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.ones_like(labels, dtype=bool)
    report = prediction_scale(labels, guesses, mask, ["a", "b"])
    assert report["a"]["mean_shift"] == 1.0
    assert report["b"]["mean_shift"] == 0.0


def test_chem_training_rejects_molecule_crossing_splits():
    import pandas as pd
    import pytest

    from minimol.errors import MinimolError
    from minimol.training.train_admet import pivot_chem

    frame = pd.DataFrame({
        'smiles_canonical': ['CCO', 'CCO'],
        'split': ['train', 'validation'],
        'endpoint': ['example', 'example'],
        'measurement': [1.0, 1.0],
    })
    with pytest.raises(MinimolError, match='multiple splits'):
        pivot_chem(frame)
