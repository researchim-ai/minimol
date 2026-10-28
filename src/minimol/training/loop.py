"""Small training loop shared by the tiny models."""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch

from minimol.errors import MinimolError
from minimol.training.checkpoint import save_checkpoint


def move_batch(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_batch(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch(value, device) for value in batch)
    return batch


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@contextmanager
def amp_context(device: torch.device, precision: str):
    use_bf16 = (
        precision == "bf16"
        and device.type == "cuda"
        and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    )
    if use_bf16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        with nullcontext():
            yield


def set_cosine_lr(optimizer, step: int, total: int, base_lr: float) -> None:
    if total <= 1:
        scale = 1.0
    else:
        scale = 0.5 * (1.0 + math.cos(math.pi * min(step, total) / total))
    for group in optimizer.param_groups:
        group["lr"] = base_lr * scale


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    metric_fn,
    *,
    epochs: int,
    patience: int,
    grad_clip: float,
    accumulation_steps: int,
    precision: str,
    device: torch.device,
    max_steps: int | None,
    checkpoint_dir,
    metadata: dict,
    base_lr: float,
) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(accumulation_steps, 1)))
    total_steps = epochs * steps_per_epoch
    if max_steps is not None:
        total_steps = min(total_steps, int(max_steps))
    best = None
    bad_epochs = 0
    history = []
    global_step = 0
    stop = False
    best_path = checkpoint_dir / "checkpoint_best.pt"
    last_path = checkpoint_dir / "checkpoint_last.pt"
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        seen = 0
        for batch_index, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            with amp_context(device, precision):
                loss = loss_fn(model, batch)
                if accumulation_steps > 1:
                    loss = loss / accumulation_steps
            if not torch.isfinite(loss):
                raise MinimolError("non-finite loss; refusing to continue")
            loss.backward()
            if (batch_index + 1) % accumulation_steps == 0:
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                set_cosine_lr(optimizer, global_step, total_steps, base_lr)
            running += float(loss.detach()) * max(accumulation_steps, 1)
            seen += 1
            if max_steps is not None and global_step >= max_steps:
                stop = True
                break
        val_metric = None
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_metric = metric_fn(model, val_loader)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / max(seen, 1),
                "val_metric": None if val_metric is None or not np.isfinite(val_metric) else float(val_metric),
            }
        )
        save_checkpoint(last_path, model, optimizer, {**metadata, "epoch": epoch, "step": global_step})
        improved = (
            val_metric is not None
            and np.isfinite(val_metric)
            and (best is None or float(val_metric) > best)
        )
        if improved:
            best = float(val_metric)
            bad_epochs = 0
            save_checkpoint(best_path, model, optimizer, {**metadata, "epoch": epoch, "step": global_step, "best_metric": best})
        elif val_metric is not None and np.isfinite(val_metric):
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                break
        if stop:
            break
    if not best_path.exists():
        save_checkpoint(best_path, model, optimizer, {**metadata, "step": global_step, "best_metric": best})
    return {"best_metric": best, "history": history, "steps": global_step, "best_path": str(best_path)}
