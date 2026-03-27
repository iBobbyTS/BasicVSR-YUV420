from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .utils import atomic_write_bytes


def _torch_load(path: Union[str, Path], map_location: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_checkpoint(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    best_psnr: Optional[float] = None,
    metrics: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None and scaler.is_enabled():
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    if best_psnr is not None:
        checkpoint["best_psnr"] = best_psnr
    if metrics is not None:
        checkpoint["metrics"] = metrics
    if model_config is not None:
        checkpoint["model_config"] = model_config

    return checkpoint


def save_checkpoint(path: Union[str, Path], checkpoint: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    atomic_write_bytes(buffer.getvalue(), output_path)


def load_model_weights(
    path: Union[str, Path],
    model: nn.Module,
    *,
    strict: bool = True,
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    state = _torch_load(path, map_location=map_location)
    model_state = state.get("model_state_dict", state)
    model.load_state_dict(model_state, strict=strict)
    return state


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    *,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    strict: bool = True,
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    state = load_model_weights(path, model, strict=strict, map_location=map_location)

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None and scaler.is_enabled() and "scaler_state_dict" in state:
        scaler.load_state_dict(state["scaler_state_dict"])

    return state
