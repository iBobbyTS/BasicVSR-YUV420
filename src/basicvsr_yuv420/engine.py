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
    ssim: float
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
    progress_desc: str = "Train",
) -> EpochResult:
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    current_lr = optimizer.param_groups[0]["lr"]

    progress_bar = tqdm(dataloader, desc=progress_desc, leave=False)

    for lr, hr in progress_bar:
        lr, hr = _prepare_batch(lr, hr, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            prediction = model(lr)
            loss = loss_fn(prediction, hr)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        with torch.no_grad():
            batch_psnr = sequence_psnr(prediction.detach(), hr).item()
            batch_ssim = sequence_ssim(prediction.detach(), hr).item()

        total_loss += loss.item()
        total_psnr += batch_psnr
        total_ssim += batch_ssim

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            psnr=f"{batch_psnr:.4f}",
            ssim=f"{batch_ssim:.4f}",
            lr=f"{current_lr:.2e}",
        )

    num_batches = max(len(dataloader), 1)
    return EpochResult(
        loss=total_loss / num_batches,
        psnr=total_psnr / num_batches,
        ssim=total_ssim / num_batches,
        lr=current_lr,
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = charbonnier_loss,
    progress_desc: str = "Eval",
) -> EpochResult:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    progress_bar = tqdm(dataloader, desc=progress_desc, leave=False)

    for lr, hr in progress_bar:
        lr, hr = _prepare_batch(lr, hr, device)
        prediction = model(lr)
        loss = loss_fn(prediction, hr)
        batch_psnr = sequence_psnr(prediction, hr).item()
        batch_ssim = sequence_ssim(prediction, hr).item()

        total_loss += loss.item()
        total_psnr += batch_psnr
        total_ssim += batch_ssim

        progress_bar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{batch_psnr:.4f}", ssim=f"{batch_ssim:.4f}")

    num_batches = max(len(dataloader), 1)
    return EpochResult(
        loss=total_loss / num_batches,
        psnr=total_psnr / num_batches,
        ssim=total_ssim / num_batches,
        lr=0.0,
    )
