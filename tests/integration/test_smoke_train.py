import torch

from minimol.models.cds_cnn import MinimolCDS
from minimol.models.quantization import quantize_linear_dynamic
from minimol.seeding import set_seed
from minimol.training.train_cds import train_cds
from minimol.training.train_reg import train_reg


def test_tiny_reg_and_cds_smoke(tmp_path):
    from minimol.data.build import build_dataset

    build_dataset("synthetic_reg", tmp_path / "reg", limit=12, seed=3)
    build_dataset("synthetic_cds", tmp_path / "cds", limit=4, seed=3)
    reg = train_reg(
        {
            "model": {"preset": "5m", "rc_augment": False, "overrides": {"channels": 32, "n_blocks": 2, "dropout": 0.0}},
            "train": {
                "seed": 3,
                "epochs": 1,
                "max_steps": 2,
                "batch_size_per_gpu": 8,
                "lr": 3e-4,
                "weight_decay": 0.0,
                "early_stopping_patience": 3,
                "grad_clip": 1.0,
                "accumulation_steps": 1,
                "precision": "fp32",
            },
            "loss": {"delta": 1.0},
        },
        tmp_path / "reg" / "data.parquet",
        tmp_path / "reg_out",
    )
    assert reg["clinical_use"] is False
    assert reg["param_count"] < 200_000
    assert (tmp_path / "reg_out" / "checkpoint_best.pt").exists()
    cds = train_cds(
        {
            "model": {"preset": "5m", "overrides": {"channels": 32, "n_blocks": 2, "dropout": 0.0}},
            "train": {
                "seed": 3,
                "epochs": 1,
                "max_steps": 2,
                "batch_size_per_gpu": 4,
                "lr": 3e-4,
                "weight_decay": 0.0,
                "early_stopping_patience": 3,
                "grad_clip": 1.0,
                "accumulation_steps": 1,
                "precision": "fp32",
            },
            "loss": {"lambda_gt": 1.0, "lambda_kd": 0.25, "temperature": 1.0, "scale_by_t2": False},
        },
        tmp_path / "cds" / "data.parquet",
        tmp_path / "cds_out",
    )
    assert "Carbon-A distillation" in cds["note"]
    assert cds["teacher_kind"] == "synthetic_noisy_label"


def test_eval_reloads_reg_checkpoint(tmp_path):
    from minimol.data.build import build_dataset
    from minimol.evaluation.reload import evaluate_reg_checkpoint
    from minimol.training.train_reg import train_reg

    build_dataset("synthetic_reg", tmp_path / "reg", limit=20, seed=5)
    train_reg(
        {
            "model": {"preset": "5m", "overrides": {"channels": 16, "n_blocks": 2, "dropout": 0.0}},
            "train": {
                "seed": 5,
                "epochs": 1,
                "max_steps": 1,
                "batch_size_per_gpu": 8,
                "lr": 3e-4,
                "weight_decay": 0.0,
                "early_stopping_patience": 2,
                "grad_clip": 1.0,
                "accumulation_steps": 1,
                "precision": "fp32",
                "device": "cpu",
            },
            "loss": {"delta": 1.0},
        },
        tmp_path / "reg" / "data.parquet",
        tmp_path / "reg_out",
    )
    report = evaluate_reg_checkpoint(
        tmp_path / "reg_out" / "checkpoint_best.pt",
        tmp_path / "reg" / "data.parquet",
        tmp_path / "eval_reg",
        repeats=2,
    )
    assert report["split"] == "validation"
    assert report["clinical_use"] is False
    assert report["latency"]["p50_s"] >= 0
    assert (tmp_path / "eval_reg" / "metrics.json").exists()


def test_tiny_precision_smoke(tmp_path):
    from minimol.data.build import build_dataset
    from minimol.training.train_precision import train_precision

    build_dataset("synthetic_precision", tmp_path / "precision", seed=2)
    payload = train_precision(
        {
            "model": {"preset": "tiny"},
            "train": {"seed": 1, "epochs": 1, "max_steps": 1, "batch_size_per_gpu": 4, "lr": 1e-3, "device": "cpu"},
        },
        tmp_path / "precision" / "data.parquet",
        tmp_path / "precision_out",
    )
    assert payload["clinical_use"] is False
    assert payload["directions"] == ["decrease", "no_change", "increase"]
    assert "Synthetic matched pairs" in payload["note"]


def test_tiny_bind_smoke(tmp_path):
    from minimol.data.build import build_dataset
    from minimol.training.train_bind import train_bind

    build_dataset("synthetic_bind", tmp_path / "bind", seed=2)
    payload = train_bind(
        {
            "model": {"preset": "tiny"},
            "train": {"seed": 1, "epochs": 1, "max_steps": 1, "batch_size_per_gpu": 4, "lr": 1e-3},
        },
        tmp_path / "bind" / "data.parquet",
        tmp_path / "bind_out",
    )
    assert payload["clinical_use"] is False
    assert payload["label_quality"] == "synthetic_interaction"
    assert "not a Ki" in payload["note"]


def test_seed_repeats_initialization():
    set_seed(5)
    first = MinimolCDS("5m", channels=16, n_blocks=2, dropout=0.0)
    set_seed(5)
    second = MinimolCDS("5m", channels=16, n_blocks=2, dropout=0.0)
    batch = torch.zeros(1, 5, 32)
    assert torch.allclose(first(batch), second(batch))


def test_dynamic_quantization_runs():
    model = MinimolCDS("5m", channels=16, n_blocks=1, dropout=0.0)
    quantized = quantize_linear_dynamic(model)
    out = quantized(torch.zeros(1, 5, 16))
    assert torch.isfinite(out).all()
