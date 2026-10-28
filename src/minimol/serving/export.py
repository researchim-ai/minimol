"""ONNX export for the sequence models. Variable length stays a PyTorch path unless noted."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from minimol.errors import MinimolError
from minimol.jsonutil import write_json


class _OnnxWrapper(nn.Module):
    """Single-input wrapper. Optional pad masks stay on the PyTorch path."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, sequence_one_hot: torch.Tensor) -> torch.Tensor:
        return self.model(sequence_one_hot)


def export_sequence_model(model: torch.nn.Module, path: Path, length: int, in_channels: int = 5) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    wrapper = _OnnxWrapper(model).eval()
    # A real one-hot. The all-zero tensor is outside the input domain and is a
    # poor parity probe for GroupNorm.
    dummy = torch.zeros(1, in_channels, length)
    positions = torch.arange(length)
    dummy[0, positions % min(4, in_channels), positions] = 1.0
    try:
        torch.onnx.export(
            wrapper,
            dummy,
            str(path),
            input_names=["sequence_one_hot"],
            output_names=["prediction"],
            opset_version=17,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            wrapper,
            dummy,
            str(path),
            input_names=["sequence_one_hot"],
            output_names=["prediction"],
            opset_version=17,
        )
    report = {"path": str(path), "length": length, "parity": None}
    try:
        import onnxruntime as ort

        with torch.no_grad():
            reference = wrapper(dummy).detach().cpu().numpy()
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        exported = session.run(None, {"sequence_one_hot": dummy.numpy()})[0]
        report["parity"] = bool(np.allclose(reference, exported, atol=1e-4, rtol=1e-4))
        report["max_abs_diff"] = float(np.max(np.abs(reference - exported)))
    except Exception as exc:
        raise MinimolError(f"ONNX parity check failed: {exc}") from exc
    write_json(path.with_suffix(".parity.json"), report)
    if not report["parity"]:
        raise MinimolError(
            f"ONNX output differs from PyTorch beyond 1e-4 (max abs {report['max_abs_diff']})"
        )
    return report


def export_checkpoint(run_dir: Path, fmt: str = "onnx", length: int | None = None) -> dict:
    if fmt != "onnx":
        raise MinimolError("only onnx export is implemented")
    checkpoint = Path(run_dir) / "checkpoint_best.pt"
    if not checkpoint.exists():
        raise MinimolError("checkpoint_best.pt is missing")
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = blob["metadata"]
    task = meta.get("task")
    if task == "reg":
        from minimol.models.reg_cnn import MinimolReg

        model = MinimolReg(meta["preset"], n_assays=meta["n_assays"], **(meta.get("overrides") or {}))
    elif task == "cds":
        from minimol.models.cds_cnn import MinimolCDS

        model = MinimolCDS(meta["preset"], **(meta.get("overrides") or {}))
    else:
        raise MinimolError(f"ONNX export is implemented for reg and cds, not {task}")
    model.load_state_dict(blob["model_state"])
    width = int(length or min(int(meta.get("length_max", 64)), 128))
    return export_sequence_model(model, Path(run_dir) / "model.onnx", length=width)


def main(argv=None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="minimol.serving.export")
    parser.add_argument("--run", required=True)
    parser.add_argument("--format", default="onnx")
    parser.add_argument("--length", type=int)
    args = parser.parse_args(argv)
    try:
        report = export_checkpoint(Path(args.run), args.format, args.length)
    except MinimolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report["path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
