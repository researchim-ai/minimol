"""Machine snapshot for artifacts/reports/system.json and artifacts/env.json."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _pkg(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _meminfo() -> dict:
    info = {"ram_total_bytes": None, "ram_available_bytes": None}
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return info
    values = {}
    for line in meminfo.read_text().splitlines():
        key, _, rest = line.partition(":")
        parts = rest.strip().split()
        if parts and parts[0].isdigit():
            values[key] = int(parts[0]) * 1024
    info["ram_total_bytes"] = values.get("MemTotal")
    info["ram_available_bytes"] = values.get("MemAvailable")
    return info


def _nvidia_smi() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def collect_system_info() -> dict:
    disk = shutil.disk_usage(Path.cwd())
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "disk_free_bytes": disk.free,
        "disk_total_bytes": disk.total,
        "packages": {
            name: _pkg(name)
            for name in (
                "torch",
                "numpy",
                "pandas",
                "scikit-learn",
                "rdkit",
                "xgboost",
                "onnx",
                "onnxruntime",
                "fastapi",
            )
        },
        "nvidia_smi": _nvidia_smi(),
        **_meminfo(),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        gpus = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                gpus.append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "total_memory_bytes": int(props.total_memory),
                    }
                )
        info["gpus"] = gpus
        info["bf16_supported"] = bool(
            torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
    except Exception as exc:  # pragma: no cover - torch is a hard dependency
        info["torch_error"] = str(exc)
    info["note"] = (
        "GPU memory is not pooled across devices. A 2x3090 machine has two separate 24 GiB budgets."
    )
    return info


def write_system_info(directory: str | Path) -> dict:
    import json

    info = collect_system_info()
    root = Path(directory)
    for relative in ("artifacts/reports/system.json", "artifacts/env.json"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info, indent=2) + "\n")
    return info
