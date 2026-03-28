from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import charbonnier_loss
from .metrics import sequence_psnr, sequence_ssim


@dataclass
class EpochResult:
    loss: float
    psnr: float
    ssim: Optional[float]
    lr: float


def _prepare_batch(
    lr: torch.Tensor,
    hr: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lr = lr.float().permute(0, 1, 4, 2, 3).contiguous().to(device, non_blocking=True)
    hr = hr.float().permute(0, 1, 4, 2, 3).contiguous().to(device, non_blocking=True)
    return lr, hr


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    *,
    scheduler: Optional[_LRScheduler] = None,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = charbonnier_loss,
    scaler: Optional[GradScaler] = None,
    amp_enabled: bool = False,
    compute_ssim: bool = True,
    grad_accum_steps: int = 1,
    clip_grad_norm: Optional[float] = None,
    progress_desc: str = "Train",
) -> EpochResult:
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0 if compute_ssim else None
    current_lr = optimizer.param_groups[0]["lr"]

    progress_bar = tqdm(dataloader, desc=progress_desc, leave=False)
    optimizer.zero_grad(set_to_none=True)

    for batch_index, (lr, hr) in enumerate(progress_bar, start=1):
        lr, hr = _prepare_batch(lr, hr, device)

        with autocast(enabled=amp_enabled):
            prediction = model(lr)
            loss = loss_fn(prediction, hr)

        backward_loss = loss / grad_accum_steps
        should_step = batch_index % grad_accum_steps == 0 or batch_index == len(dataloader)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(backward_loss).backward()
            if should_step:
                if clip_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            backward_loss.backward()
            if should_step:
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if scheduler is not None and should_step:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        with torch.no_grad():
            detached_prediction = prediction.detach()
            batch_psnr = sequence_psnr(detached_prediction, hr).item()
            batch_ssim = None
            if compute_ssim:
                batch_ssim = sequence_ssim(detached_prediction.float(), hr).item()

        total_loss += loss.item()
        total_psnr += batch_psnr
        if batch_ssim is not None and total_ssim is not None:
            total_ssim += batch_ssim

        postfix = {
            "loss": f"{loss.item():.4f}",
            "psnr": f"{batch_psnr:.4f}",
            "lr": f"{current_lr:.2e}",
        }
        if batch_ssim is not None:
            postfix["ssim"] = f"{batch_ssim:.4f}"
        progress_bar.set_postfix(
            **postfix,
        )

    num_batches = max(len(dataloader), 1)
    return EpochResult(
        loss=total_loss / num_batches,
        psnr=total_psnr / num_batches,
        ssim=None if total_ssim is None else total_ssim / num_batches,
        lr=current_lr,
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = charbonnier_loss,
    amp_enabled: bool = False,
    compute_ssim: bool = True,
    progress_desc: str = "Eval",
) -> EpochResult:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0 if compute_ssim else None

    progress_bar = tqdm(dataloader, desc=progress_desc, leave=False)

    for lr, hr in progress_bar:
        lr, hr = _prepare_batch(lr, hr, device)
        with autocast(enabled=amp_enabled):
            prediction = model(lr)
            loss = loss_fn(prediction, hr)

        batch_psnr = sequence_psnr(prediction, hr).item()
        batch_ssim = None
        if compute_ssim:
            batch_ssim = sequence_ssim(prediction.float(), hr).item()

        total_loss += loss.item()
        total_psnr += batch_psnr
        if batch_ssim is not None and total_ssim is not None:
            total_ssim += batch_ssim

        postfix = {
            "loss": f"{loss.item():.4f}",
            "psnr": f"{batch_psnr:.4f}",
        }
        if batch_ssim is not None:
            postfix["ssim"] = f"{batch_ssim:.4f}"
        progress_bar.set_postfix(**postfix)

    num_batches = max(len(dataloader), 1)
    return EpochResult(
        loss=total_loss / num_batches,
        psnr=total_psnr / num_batches,
        ssim=None if total_ssim is None else total_ssim / num_batches,
        lr=0.0,
    )
