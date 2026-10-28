import torch

from minimol.models.reg_cnn import MinimolReg
from minimol.seeding import set_seed
from minimol.serving.export import export_sequence_model
from minimol.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_reload_matches_predictions(tmp_path):
    set_seed(0)
    model = MinimolReg("5m", channels=16, n_blocks=2, dropout=0.0)
    batch = torch.zeros(2, 5, 24)
    expected = model(batch).detach()
    save_checkpoint(tmp_path / "model.pt", model, None, {"task": "reg"})
    other = MinimolReg("5m", channels=16, n_blocks=2, dropout=0.0)
    load_checkpoint(tmp_path / "model.pt", other)
    assert torch.allclose(other(batch), expected)


def test_onnx_matches_pytorch(tmp_path):
    model = MinimolReg("5m", channels=16, n_blocks=2, dropout=0.0)
    report = export_sequence_model(model, tmp_path / "model.onnx", length=24)
    assert report["parity"] is True
    assert report["max_abs_diff"] < 1e-4
