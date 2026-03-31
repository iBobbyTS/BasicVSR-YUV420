from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

BT709_KR = 0.2126
BT709_KB = 0.0722
BT709_KG = 1.0 - BT709_KR - BT709_KB


def _flatten_spatial(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    if tensor.ndim < 3:
        raise ValueError(f"Expected at least 3 dimensions, but found {tensor.ndim}.")
    leading_shape = tensor.shape[:-3]
    channels, height, width = tensor.shape[-3:]
    flattened = tensor.reshape(-1, channels, height, width)
    return flattened, leading_shape


def rgb_to_yuv444_bt709_full_range(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.shape[-3] != 3:
        raise ValueError(f"Expected RGB input with 3 channels, but found {rgb.shape}.")

    r = rgb.select(-3, 0)
    g = rgb.select(-3, 1)
    b = rgb.select(-3, 2)

    y = BT709_KR * r + BT709_KG * g + BT709_KB * b
    u = (b - y) / (2.0 * (1.0 - BT709_KB)) + 0.5
    v = (r - y) / (2.0 * (1.0 - BT709_KR)) + 0.5
    return torch.stack((y, u, v), dim=-3)


def yuv444_to_rgb_bt709_full_range(yuv: torch.Tensor) -> torch.Tensor:
    if yuv.shape[-3] != 3:
        raise ValueError(f"Expected YUV input with 3 channels, but found {yuv.shape}.")

    y = yuv.select(-3, 0)
    u = yuv.select(-3, 1) - 0.5
    v = yuv.select(-3, 2) - 0.5

    r = y + 2.0 * (1.0 - BT709_KR) * v
    b = y + 2.0 * (1.0 - BT709_KB) * u
    g = (y - BT709_KR * r - BT709_KB * b) / BT709_KG
    return torch.stack((r, g, b), dim=-3).clamp(0.0, 1.0)


def downsample_uv_to_420(uv444: torch.Tensor) -> torch.Tensor:
    if uv444.shape[-3] != 2:
        raise ValueError(f"Expected UV input with 2 channels, but found {uv444.shape}.")
    if uv444.shape[-2] % 2 != 0 or uv444.shape[-1] % 2 != 0:
        raise ValueError("UV420 downsampling requires even spatial dimensions.")

    flattened, leading_shape = _flatten_spatial(uv444)
    downsampled = F.avg_pool2d(flattened, kernel_size=2, stride=2)
    return downsampled.reshape(*leading_shape, 2, downsampled.size(-2), downsampled.size(-1))


def upsample_uv420_for_preview(
    uv420: torch.Tensor,
    *,
    size: Tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    if uv420.shape[-3] != 2:
        raise ValueError(f"Expected UV input with 2 channels, but found {uv420.shape}.")

    flattened, leading_shape = _flatten_spatial(uv420)
    upsampled = F.interpolate(flattened, size=size, mode=mode, align_corners=False)
    return upsampled.reshape(*leading_shape, 2, size[0], size[1])


def rgb_to_yuv420_bt709_full_range(rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    yuv444 = rgb_to_yuv444_bt709_full_range(rgb)
    y = yuv444.narrow(-3, 0, 1)
    uv444 = yuv444.narrow(-3, 1, 2)
    uv420 = downsample_uv_to_420(uv444)
    return y, uv420


def yuv420_to_rgb_bt709_full_range(y: torch.Tensor, uv420: torch.Tensor) -> torch.Tensor:
    if y.shape[-3] != 1:
        raise ValueError(f"Expected Y input with 1 channel, but found {y.shape}.")
    if uv420.shape[-3] != 2:
        raise ValueError(f"Expected UV input with 2 channels, but found {uv420.shape}.")

    uv444 = upsample_uv420_for_preview(uv420, size=(y.shape[-2], y.shape[-1]))
    yuv444 = torch.cat((y, uv444), dim=-3)
    return yuv444_to_rgb_bt709_full_range(yuv444)


def rgb_through_yuv420_bt709_full_range(rgb: torch.Tensor) -> torch.Tensor:
    y, uv420 = rgb_to_yuv420_bt709_full_range(rgb)
    return yuv420_to_rgb_bt709_full_range(y, uv420)
