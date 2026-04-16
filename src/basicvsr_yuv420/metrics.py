from __future__ import annotations

import math
from typing import Dict

import torch
from torchmetrics.functional.image import structural_similarity_index_measure

from .data.colorspace import rgb_to_yuv420_bt709_full_range, yuv420_to_rgb_bt709_full_range


def sequence_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"Mismatched tensor shapes: {prediction.shape} vs {target.shape}")

    if prediction.ndim == 5:
        reduction_dims = (2, 3, 4)
    elif prediction.ndim == 4:
        reduction_dims = (1, 2, 3)
    else:
        raise ValueError(f"Unsupported tensor rank for PSNR: {prediction.ndim}")

    mse = torch.mean((prediction - target) ** 2, dim=reduction_dims)
    psnr = 20.0 * math.log10(data_range) - 10.0 * torch.log10(mse.clamp_min(eps))
    return psnr.mean()


def sequence_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"Mismatched tensor shapes: {prediction.shape} vs {target.shape}")

    if prediction.ndim == 5:
        batch, steps, channels, height, width = prediction.shape
        prediction = prediction.reshape(batch * steps, channels, height, width)
        target = target.reshape(batch * steps, channels, height, width)
    elif prediction.ndim != 4:
        raise ValueError(f"Unsupported tensor rank for SSIM: {prediction.ndim}")

    return structural_similarity_index_measure(prediction, target, data_range=data_range)


def weighted_yuv420_psnr(
    prediction: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    data_range: float = 1.0,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    psnr_y = sequence_psnr(prediction["y"], target["y"], data_range=data_range, eps=eps)
    psnr_uv = sequence_psnr(prediction["uv"], target["uv"], data_range=data_range, eps=eps)
    return {
        "psnr": 0.5 * psnr_y + 0.5 * psnr_uv,
        "psnr_y": psnr_y,
        "psnr_uv": psnr_uv,
    }


def weighted_yuv420_ssim(
    prediction: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    data_range: float = 1.0,
) -> Dict[str, torch.Tensor]:
    ssim_y = sequence_ssim(prediction["y"], target["y"], data_range=data_range)
    ssim_uv = sequence_ssim(prediction["uv"], target["uv"], data_range=data_range)
    return {
        "ssim": 0.5 * ssim_y + 0.5 * ssim_uv,
        "ssim_y": ssim_y,
        "ssim_uv": ssim_uv,
    }


def yuv420_reconstructed_rgb(prediction: Dict[str, torch.Tensor]) -> torch.Tensor:
    return yuv420_to_rgb_bt709_full_range(prediction["y"], prediction["uv"])


def yuv420_rgb_recon_psnr(
    prediction: Dict[str, torch.Tensor],
    target_rgb: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    return sequence_psnr(yuv420_reconstructed_rgb(prediction), target_rgb, data_range=data_range, eps=eps)


def yuv420_rgb_recon_ssim(
    prediction: Dict[str, torch.Tensor],
    target_rgb: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    return sequence_ssim(yuv420_reconstructed_rgb(prediction), target_rgb, data_range=data_range)


def rgb_yuv420_secondary_psnr(
    prediction_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    prediction_y, prediction_uv = rgb_to_yuv420_bt709_full_range(prediction_rgb)
    target_y, target_uv = rgb_to_yuv420_bt709_full_range(target_rgb)
    return {
        "y_psnr": sequence_psnr(prediction_y, target_y, data_range=data_range, eps=eps),
        "uv420_psnr": sequence_psnr(prediction_uv, target_uv, data_range=data_range, eps=eps),
    }
