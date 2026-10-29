"""Seed Python, NumPy, and PyTorch.

Deterministic cuDNN is off unless requested. Dilated convolutions get much
slower with it, and those flags still do not make GPU kernels bit-identical.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
