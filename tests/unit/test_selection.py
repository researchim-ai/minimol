import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from minimol.evaluation.reg import reg_metric_value
from minimol.training.loop import fit


def test_regression_selection_uses_negative_mae():
    score = reg_metric_value({"macro_mae": 1.25, "macro_spearman": 0.9})
    assert score["value"] == -1.25
    assert score["val_spearman"] == 0.9


def test_fit_keeps_the_lower_error_epoch(tmp_path):
    model = nn.Linear(1, 1)
    data = TensorDataset(torch.zeros(4, 1), torch.zeros(4, 1))
    loader = DataLoader(data, batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    seen = {"n": 0}

    def loss_fn(module, batch):
        return (module(batch[0]) - batch[1]).pow(2).mean()

    def metric_fn(_model, _loader):
        seen["n"] += 1
        if seen["n"] == 1:
            return {"value": -7.0, "val_mae": 7.0, "val_spearman": 0.9}
        return {"value": -0.4, "val_mae": 0.4, "val_spearman": 0.5}

    result = fit(
        model,
        loader,
        loader,
        optimizer,
        loss_fn,
        metric_fn,
        epochs=2,
        patience=5,
        grad_clip=0.0,
        accumulation_steps=1,
        precision="fp32",
        device=torch.device("cpu"),
        max_steps=None,
        checkpoint_dir=tmp_path,
        metadata={"task": "toy"},
        base_lr=0.0,
    )
    assert result["history"][0]["val_mae"] == 7.0
    assert result["history"][1]["val_mae"] == 0.4
    blob = torch.load(tmp_path / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    assert blob["metadata"]["epoch"] == 1
    assert blob["metadata"]["best_metric"] == -0.4
