from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple


def _meshgrid(height: int, width: int, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ys = torch.arange(height, device=x.device, dtype=x.dtype)
    xs = torch.arange(width, device=x.device, dtype=x.dtype)
    try:
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    except TypeError:
        grid_y, grid_x = torch.meshgrid(ys, xs)
    return grid_y, grid_x


def flow_warp(
    x: torch.Tensor,
    flow: torch.Tensor,
    interp_mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    if x.ndim != 4 or flow.ndim != 4:
        raise ValueError("flow_warp expects 4D tensors.")
    if x.shape[-2:] != flow.shape[1:3]:
        raise ValueError(f"Mismatched spatial shapes: {x.shape[-2:]} vs {flow.shape[1:3]}")

    _, _, height, width = x.shape
    grid_y, grid_x = _meshgrid(height, width, x)
    grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0)

    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[..., 0] / max(width - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(height - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)

    return F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
