from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import flow_warp
from .spynet import SpyNet


def resize_flow(flow: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if flow.shape[-2:] == size:
        return flow

    scale_x = size[1] / flow.shape[-1]
    scale_y = size[0] / flow.shape[-2]
    resized = F.interpolate(flow, size=size, mode="bilinear", align_corners=False)
    resized[:, 0, :, :] *= scale_x
    resized[:, 1, :, :] *= scale_y
    return resized


def compute_bidirectional_flows_from_luma(spynet: SpyNet, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, steps, _, height, width = y.shape
    start = y[:, :-1, :, :, :].reshape(-1, 1, height, width).repeat(1, 3, 1, 1)
    end = y[:, 1:, :, :, :].reshape(-1, 1, height, width).repeat(1, 3, 1, 1)
    forward = spynet(start, end).view(batch, steps - 1, 2, height, width)
    backward = spynet(end, start).view(batch, steps - 1, 2, height, width)
    return forward, backward


def warp_feature(feature: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    return flow_warp(feature, flow.permute(0, 2, 3, 1))


def downsample_luma_to_uv(y: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(y, kernel_size=2, stride=2)


def split_luma_low_high(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    low = F.avg_pool2d(y, kernel_size=2, stride=2)
    reconstructed_low = F.interpolate(low, size=y.shape[-2:], mode="bilinear", align_corners=False)
    high = y - reconstructed_low
    return low, high


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.conv1(x))
        out = self.conv2(out)
        return self.activation(out + residual)


class ResidualStack(nn.Module):
    def __init__(self, channels: int, blocks: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.projection = nn.Conv2d(channels, channels, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.projection(self.blocks(x))


class RecurrentPropagation(nn.Module):
    def __init__(self, input_channels: int, feature_channels: int, residual_blocks: int) -> None:
        super().__init__()
        self.input_proj = nn.Conv2d(input_channels + feature_channels, feature_channels, 3, stride=1, padding=1)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.body = ResidualStack(feature_channels, residual_blocks)

    def forward(self, current_input: torch.Tensor, propagated: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((propagated, current_input), dim=1)
        return self.body(self.activation(self.input_proj(combined)))


class PixelShuffleUpsampleHead(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int) -> None:
        super().__init__()
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.conv1 = nn.Conv2d(input_channels, hidden_channels * 4, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 4, 3, stride=1, padding=1)
        self.output = nn.Conv2d(hidden_channels, output_channels, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.pixel_shuffle(self.activation(self.conv1(x)), 2)
        out = F.pixel_shuffle(self.activation(self.conv2(out)), 2)
        return self.output(out)


class FiLMConditioner(nn.Module):
    def __init__(self, source_channels: int, target_channels: int) -> None:
        super().__init__()
        self.to_gamma_beta = nn.Conv2d(source_channels, target_channels * 2, 3, stride=1, padding=1)

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.to_gamma_beta(source)
        if gamma_beta.shape[-2:] != target.shape[-2:]:
            gamma_beta = F.interpolate(gamma_beta, size=target.shape[-2:], mode="bilinear", align_corners=False)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return target * (1.0 + torch.tanh(gamma)) + beta


class CrossScaleGate(nn.Module):
    def __init__(self, source_channels: int, target_channels: int) -> None:
        super().__init__()
        self.feature = nn.Conv2d(source_channels, target_channels, 3, stride=1, padding=1)
        self.gate = nn.Sequential(
            nn.Conv2d(source_channels, target_channels, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, source: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        feature = self.feature(source)
        gate = self.gate(source)
        if feature.shape[-2:] != target_size:
            feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            gate = F.interpolate(gate, size=target_size, mode="bilinear", align_corners=False)
        return feature * gate


def build_shared_spynet(load_path: Optional[Union[str, Path]]) -> SpyNet:
    return SpyNet(load_path=load_path)
