"""Dynamic int8 quantization for Linear layers.

INT4 is not exposed: this environment has no checked INT4 kernel. A byte-count
estimate of the weights is not a deployable file size.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from minimol.models.blocks import count_parameters


def quantize_linear_dynamic(model: nn.Module) -> nn.Module:
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def weight_bytes(model: nn.Module, dtype_bytes: int = 4) -> int:
    return count_parameters(model) * int(dtype_bytes)
