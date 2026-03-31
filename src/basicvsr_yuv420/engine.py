from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import charbonnier_loss
from .metrics import (
    rgb_yuv420_secondary_psnr,
    sequence_psnr,
    sequence_ssim,
    weighted_yuv420_psnr,
    weighted_yuv420_ssim,
    yuv420_reconstructed_rgb,
    yuv420_rgb_recon_psnr,
    yuv420_rgb_recon_ssim,
)

BatchTensors = Union[torch.Tensor, Dict[str, torch.Tensor]]


@dataclass
class EpochResult:
    loss: float
    psnr: float
    ssim: Optional[float]
    lr: float
    y_psnr: Optional[float] = None
    uv420_psnr: Optional[float] = None
    loss_y: Optional[float] = None
    loss_uv: Optional[float] = None
    psnr_y: Optional[float] = None
    psnr_uv: Optional[float] = None
    ssim_y: Optional[float] = None
    ssim_uv: Optional[float] = None
    loss_rgb: Optional[float] = None
    psnr_rgb: Optional[float] = None
    ssim_rgb: Optional[float] = None


def _prepare_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[BatchTensors, BatchTensors, str]:
    prepared = {
        key: value.float().contiguous().to(device, non_blocking=True)
        for key, value in batch.items()
    }
    if "lr_rgb" in prepared:
        return prepared["lr_rgb"], prepared["hr_rgb"], "rgb"
    if "lr_y" in prepared:
        target = {"y": prepared["hr_y"], "uv": prepared["hr_uv"]}
        if "hr_rgb" in prepared:
            target["rgb"] = prepared["hr_rgb"]
        return {"y": prepared["lr_y"], "uv": prepared["lr_uv"]}, target, "yuv420"
    raise KeyError(f"Unsupported batch keys: {sorted(prepared)}")


def _detach_prediction(prediction: BatchTensors) -> BatchTensors:
    if isinstance(prediction, dict):
        return {key: value.detach() for key, value in prediction.items()}
    return prediction.detach()


def _compute_batch_statistics(
    prediction: BatchTensors,
    target: BatchTensors,
    *,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    compute_ssim: bool,
    metric_domain: str,
    data_range: float = 1.0,
) -> Dict[str, Optional[torch.Tensor]]:
    if isinstance(prediction, dict):
        if not isinstance(target, dict):
            raise TypeError("YUV420 predictions require dict targets.")
        loss_y = loss_fn(prediction["y"], target["y"])
        loss_uv = loss_fn(prediction["uv"], target["uv"])
        weighted_loss = 0.5 * loss_y + 0.5 * loss_uv
        weighted_psnr = weighted_yuv420_psnr(prediction, target, data_range=data_range)
        result: Dict[str, Optional[torch.Tensor]] = {
            "y_psnr": weighted_psnr["psnr_y"],
            "uv420_psnr": weighted_psnr["psnr_uv"],
            "loss_y": loss_y,
            "loss_uv": loss_uv,
            "psnr_y": weighted_psnr["psnr_y"],
            "psnr_uv": weighted_psnr["psnr_uv"],
            "ssim": None,
            "ssim_y": None,
            "ssim_uv": None,
            "loss_rgb": None,
            "psnr_rgb": None,
            "ssim_rgb": None,
        }
        if compute_ssim:
            result.update(weighted_yuv420_ssim(prediction, target, data_range=data_range))
        if metric_domain == "yuv420":
            result["loss"] = weighted_loss
            result["psnr"] = weighted_psnr["psnr"]
        elif metric_domain == "rgb":
            target_rgb = target.get("rgb")
            if target_rgb is None:
                raise KeyError("RGB-domain YUV420 evaluation requires target['rgb'].")
            prediction_rgb = yuv420_reconstructed_rgb(prediction)
            result["loss_rgb"] = loss_fn(prediction_rgb, target_rgb)
            result["psnr_rgb"] = yuv420_rgb_recon_psnr(prediction, target_rgb, data_range=data_range)
            if compute_ssim:
                result["ssim_rgb"] = yuv420_rgb_recon_ssim(prediction, target_rgb, data_range=data_range)
            result["loss"] = result["loss_rgb"]
            result["psnr"] = result["psnr_rgb"]
            result["ssim"] = result["ssim_rgb"]
        else:
            raise ValueError(f"Unsupported metric domain: {metric_domain}")
        return result

    if isinstance(target, dict):
        raise TypeError("RGB predictions require tensor targets.")
    if metric_domain != "rgb":
        raise ValueError(f"RGB predictions only support rgb metric domain, but got {metric_domain}.")
    result = {
        "loss": loss_fn(prediction, target),
        "psnr": sequence_psnr(prediction, target, data_range=data_range),
        "ssim": None,
        "y_psnr": None,
        "uv420_psnr": None,
        "loss_y": None,
        "loss_uv": None,
        "psnr_y": None,
        "psnr_uv": None,
        "ssim_y": None,
        "ssim_uv": None,
        "loss_rgb": None,
        "psnr_rgb": None,
        "ssim_rgb": None,
    }
    result.update(rgb_yuv420_secondary_psnr(prediction, target, data_range=data_range))
    if compute_ssim:
        result["ssim"] = sequence_ssim(prediction.float(), target, data_range=data_range)
    return result


def _summarize_epoch(totals: Dict[str, float], num_batches: int, lr: float) -> EpochResult:
    divisor = max(num_batches, 1)
    return EpochResult(
        loss=totals["loss"] / divisor,
        psnr=totals["psnr"] / divisor,
        ssim=None if "ssim" not in totals else totals["ssim"] / divisor,
        lr=lr,
        y_psnr=None if "y_psnr" not in totals else totals["y_psnr"] / divisor,
        uv420_psnr=None if "uv420_psnr" not in totals else totals["uv420_psnr"] / divisor,
        loss_y=None if "loss_y" not in totals else totals["loss_y"] / divisor,
        loss_uv=None if "loss_uv" not in totals else totals["loss_uv"] / divisor,
        psnr_y=None if "psnr_y" not in totals else totals["psnr_y"] / divisor,
        psnr_uv=None if "psnr_uv" not in totals else totals["psnr_uv"] / divisor,
        ssim_y=None if "ssim_y" not in totals else totals["ssim_y"] / divisor,
        ssim_uv=None if "ssim_uv" not in totals else totals["ssim_uv"] / divisor,
        loss_rgb=None if "loss_rgb" not in totals else totals["loss_rgb"] / divisor,
        psnr_rgb=None if "psnr_rgb" not in totals else totals["psnr_rgb"] / divisor,
        ssim_rgb=None if "ssim_rgb" not in totals else totals["ssim_rgb"] / divisor,
    )


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
    metric_domain: str = "rgb",
    grad_accum_steps: int = 1,
    clip_grad_norm: Optional[float] = None,
    progress_desc: str = "Train",
) -> EpochResult:
    model.train()
    totals: Dict[str, float] = {}
    current_lr = optimizer.param_groups[0]["lr"]

    progress_bar = tqdm(dataloader, desc=progress_desc, leave=False)
    optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(progress_bar, start=1):
        lr, hr, _ = _prepare_batch(batch, device)

        with autocast(enabled=amp_enabled):
            prediction = model(lr)
            batch_metrics = _compute_batch_statistics(
                prediction,
                hr,
                loss_fn=loss_fn,
                compute_ssim=compute_ssim,
                metric_domain=metric_domain,
            )
            loss = batch_metrics["loss"]
            if loss is None:
                raise RuntimeError("Loss computation returned None.")

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
            detached_prediction = _detach_prediction(prediction)
            detached_metrics = _compute_batch_statistics(
                detached_prediction,
                hr,
                loss_fn=loss_fn,
                compute_ssim=compute_ssim,
                metric_domain=metric_domain,
            )

        for key, value in detached_metrics.items():
            if value is not None:
                totals[key] = totals.get(key, 0.0) + float(value.item())

        postfix = {
            "loss": f"{detached_metrics['loss'].item():.4f}",
            "psnr": f"{detached_metrics['psnr'].item():.4f}",
            "lr": f"{current_lr:.2e}",
        }
        if detached_metrics["ssim"] is not None:
            postfix["ssim"] = f"{detached_metrics['ssim'].item():.4f}"
        progress_bar.set_postfix(
            **postfix,
        )

    return _summarize_epoch(totals, len(dataloader), current_lr)


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
    forward_fn: Optional[Callable[[BatchTensors], BatchTensors]] = None,
    metric_domain: str = "rgb",
) -> EpochResult:
    model.eval()
    totals: Dict[str, float] = {}
    predict = forward_fn if forward_fn is not None else model

    progress_bar = tqdm(dataloader, desc=progress_desc, leave=False)

    for batch in progress_bar:
        lr, hr, _ = _prepare_batch(batch, device)
        with autocast(enabled=amp_enabled):
            prediction = predict(lr)
            batch_metrics = _compute_batch_statistics(
                prediction,
                hr,
                loss_fn=loss_fn,
                compute_ssim=compute_ssim,
                metric_domain=metric_domain,
            )

        for key, value in batch_metrics.items():
            if value is not None:
                totals[key] = totals.get(key, 0.0) + float(value.item())

        postfix = {
            "loss": f"{batch_metrics['loss'].item():.4f}",
            "psnr": f"{batch_metrics['psnr'].item():.4f}",
        }
        if batch_metrics["ssim"] is not None:
            postfix["ssim"] = f"{batch_metrics['ssim'].item():.4f}"
        progress_bar.set_postfix(**postfix)

    return _summarize_epoch(totals, len(dataloader), 0.0)
