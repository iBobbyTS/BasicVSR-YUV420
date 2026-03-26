from __future__ import annotations

import math

import torch
from torchmetrics.functional.image import structural_similarity_index_measure


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
