"""Small training loop shared by the tiny models."""

from __future__ import annotations

import json
import math
import os
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch

from minimol.errors import MinimolError
from minimol.jsonutil import jsonable
from minimol.training.checkpoint import load_checkpoint, restore_rng_state, save_checkpoint


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


def metric_parts(metric) -> tuple[float | None, dict]:
    """Selection value plus extra fields for the history row.

    A dict must contain `value`. Higher `value` is better. Regression trainers
    pass negative MAE so a lucky rank correlation cannot keep a shifted model.
    """

    if isinstance(metric, dict):
        extra = {key: value for key, value in metric.items() if key != "value"}
        return metric.get("value"), extra
    return metric, {}


def set_cosine_lr(optimizer, step: int, total: int, base_lr: float) -> None:
    if total <= 1:
        scale = 1.0
    else:
        scale = 0.5 * (1.0 + math.cos(math.pi * min(step, total) / total))
    for group in optimizer.param_groups:
        group["lr"] = base_lr * scale


def _atomic_text(path: Path, text: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fit(
    model, train_loader, val_loader, optimizer, loss_fn, metric_fn, *,
    epochs: int, patience: int, grad_clip: float, accumulation_steps: int,
    precision: str, device: torch.device, max_steps: int | None,
    checkpoint_dir, metadata: dict, base_lr: float,
    resume_from: str | Path | None = None, stop_after_epochs: int | None = None,
) -> dict:
    """Train with atomic epoch snapshots. Resume retains the original total budget.

    stop_after_epochs provides a deliberate epoch-boundary interruption. Old
    checkpoints without full training/RNG state must be treated as warm-starts.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if accumulation_steps < 1 or epochs < 1 or (max_steps is not None and max_steps < 1):
        raise MinimolError("epochs, accumulation_steps, and max_steps must be positive")
    if stop_after_epochs is not None and stop_after_epochs < 1:
        raise MinimolError("stop_after_epochs must be positive")
    if len(train_loader) == 0:
        raise MinimolError("training loader is empty")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "checkpoint_best.pt"
    last_path = checkpoint_dir / "checkpoint_last.pt"
    if resume_from is None and (best_path.exists() or last_path.exists()):
        raise MinimolError("run directory already contains checkpoints; use resume or a new directory")
    model.to(device)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accumulation_steps))
    total_steps = epochs * steps_per_epoch
    if max_steps is not None:
        total_steps = min(total_steps, int(max_steps))
    plan = {"epochs": epochs, "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
            "accumulation_steps": accumulation_steps, "base_lr": base_lr,
            "precision": precision, "device_type": device.type, "patience": patience,
            "grad_clip": grad_clip, "optimizer_class": type(optimizer).__qualname__,
            "optimizer_options": [{key: value for key, value in group.items()
                                  if key not in {"params", "lr", "initial_lr"}} for group in optimizer.param_groups],
            "batch_size": train_loader.batch_size, "drop_last": train_loader.drop_last,
            "num_workers": train_loader.num_workers}
    best, bad_epochs, history, global_step, start_epoch = None, 0, [], 0, 0
    if resume_from is not None:
        resume_from = Path(resume_from)
        if resume_from.parent.resolve() != checkpoint_dir.resolve():
            raise MinimolError("resume must use the original run directory")
        blob = load_checkpoint(resume_from, model, optimizer, map_location=device)
        state = blob.get("training_state")
        if not state or not blob.get("rng_state") or not state.get("epoch_complete"):
            raise MinimolError("checkpoint has no complete resume state; use an explicit warm-start experiment")
        if state["plan"] != plan:
            raise MinimolError("resume budget, loader, precision, or optimizer schedule changed")
        for key in ("data_sha256", "task", "head_kind", "padding_policy", "seed", "source_hashes"):
            if metadata.get(key) != blob["metadata"].get(key):
                raise MinimolError(f"resume metadata changed: {key}")
        best, bad_epochs = state["best"], state["bad_epochs"]
        history, global_step, start_epoch = state["history"], state["step"], state["next_epoch"]
        for name, loader in (("train", train_loader), ("validation", val_loader)):
            generator = getattr(loader, "generator", None)
            saved = state.get("loader_generators", {}).get(name)
            if saved is not None:
                if generator is None:
                    raise MinimolError("resume loader generator is missing")
                generator.set_state(saved.cpu())
        restore_rng_state(blob["rng_state"])
    reason = "epochs"

    def persist_status(status, error=None):
        payload = {"status": status, "epoch_completed": start_epoch, "step": global_step,
                   "best_metric": best, "stop_reason": reason, "error": error}
        _atomic_text(checkpoint_dir / "status.json", json.dumps(jsonable(payload), indent=2) + "\n")
        _atomic_text(checkpoint_dir / "history.jsonl", "".join(json.dumps(jsonable(row)) + "\n" for row in history))

    persist_status("running")
    try:
        for epoch in range(start_epoch, epochs):
            if global_step >= total_steps or (patience and bad_epochs >= patience):
                reason = "budget_or_patience_already_reached"
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running, seen, completed = 0.0, 0, True
            for batch_index, batch in enumerate(train_loader):
                batch = move_batch(batch, device)
                with amp_context(device, precision):
                    raw_loss = loss_fn(model, batch)
                    group_start = (batch_index // accumulation_steps) * accumulation_steps
                    group_size = min(accumulation_steps, len(train_loader) - group_start)
                    loss = raw_loss / group_size
                if not torch.isfinite(loss):
                    raise MinimolError("non-finite loss; refusing to continue")
                loss.backward()
                if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(train_loader):
                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    set_cosine_lr(optimizer, global_step, total_steps, base_lr)
                running += float(raw_loss.detach())
                seen += 1
                if global_step >= total_steps:
                    completed = batch_index + 1 == len(train_loader)
                    break
            val_metric, extra = None, {}
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    val_metric, extra = metric_parts(metric_fn(model, val_loader))
            row = {"epoch": epoch, "step": global_step, "lr": optimizer.param_groups[0]["lr"],
                   "train_loss": running / max(seen, 1), "val_metric": val_metric, **extra}
            history.append(jsonable(row))
            improved = val_metric is not None and np.isfinite(val_metric) and (best is None or float(val_metric) > best)
            if improved:
                best, bad_epochs = float(val_metric), 0
            elif val_metric is not None and np.isfinite(val_metric):
                bad_epochs += 1
            start_epoch = epoch + 1
            generators = {}
            for name, loader in (("train", train_loader), ("validation", val_loader)):
                generator = getattr(loader, "generator", None)
                if generator is not None:
                    generators[name] = generator.get_state()
            state = {"plan": plan, "epoch_complete": completed, "next_epoch": start_epoch,
                     "best": best, "bad_epochs": bad_epochs, "step": global_step,
                     "history": history, "loader_generators": generators}
            meta = {**metadata, "epoch": epoch, "step": global_step, "best_metric": best}
            save_checkpoint(last_path, model, optimizer, meta, state)
            if improved or val_loader is None or not best_path.exists():
                save_checkpoint(best_path, model, optimizer, meta, state)
            persist_status("running")
            print(f"epoch {epoch + 1}/{epochs} step {global_step} loss {row['train_loss']:.5f} validation {val_metric}", flush=True)
            if patience and bad_epochs >= patience:
                reason = "early_stopping"
                break
            if global_step >= total_steps:
                reason = "max_steps" if max_steps is not None else "epochs"
                break
            if stop_after_epochs is not None and start_epoch >= stop_after_epochs:
                reason = "epoch_boundary_interruption"
                break
    except BaseException as exc:
        persist_status("interrupted", type(exc).__name__ + ": " + str(exc))
        raise
    persist_status("interrupted" if reason == "epoch_boundary_interruption" else "completed")
    return {"best_metric": best, "history": history, "steps": global_step,
            "best_path": str(best_path), "stop_reason": reason}
