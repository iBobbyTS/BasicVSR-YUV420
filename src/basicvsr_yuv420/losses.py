from __future__ import annotations

import torch


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.mean(torch.sqrt((prediction - target) ** 2 + eps**2))
