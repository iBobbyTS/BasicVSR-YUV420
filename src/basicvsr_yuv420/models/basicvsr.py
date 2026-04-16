from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicvsr_yuv420.data.colorspace import rgb_through_yuv420_bt709_full_range

from .common import flow_warp
from .spynet import SpyNet


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.size()
        pooled = self.avg_pool(x).view(batch, channels)
        weights = self.fc(pooled).view(batch, channels, 1, 1)
        return x * weights.expand_as(x)


class ImprovedResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.channel_attention = ChannelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.channel_attention(out)
        out = out + residual
        return self.activation(out)


class ResidualInResidual(nn.Module):
    def __init__(self, channels: int, residual_blocks: int = 7) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[ImprovedResBlock(channels) for _ in range(residual_blocks)])
        self.conv = nn.Conv2d(channels, channels, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.blocks(x)
        out = self.conv(out)
        return out + residual


class AdaptiveFlowRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels + 2, channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, 2, 3, stride=1, padding=1)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, features: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((features, flow), dim=1)
        flow_residual = self.conv2(self.activation(self.conv1(combined)))
        return flow + flow_residual


class MultiScaleFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(channels, channels, 3, stride=1, padding=3, dilation=3)
        self.fusion_conv = nn.Conv2d(3 * channels, channels, 1, stride=1, padding=0)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale1 = self.activation(self.conv1(x))
        scale2 = self.activation(self.conv2(x))
        scale3 = self.activation(self.conv3(x))
        return self.fusion_conv(torch.cat((scale1, scale2, scale3), dim=1))


class AttentionFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2 * channels, channels, 3, stride=1, padding=1)
        self.attention = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, forward_features: torch.Tensor, backward_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((forward_features, backward_features), dim=1)
        weights = self.attention(combined)
        forward_attended = forward_features * weights
        backward_attended = backward_features * (1.0 - weights)
        return self.conv(torch.cat((forward_attended, backward_attended), dim=1))


class Generator(nn.Module):
    def __init__(
        self,
        spynet: SpyNet,
        *,
        num_channels: int = 64,
        residual_blocks: int = 7,
    ) -> None:
        super().__init__()
        self.spynet = spynet
        self.num_channels = num_channels
        self.activation = nn.LeakyReLU(0.1, inplace=True)

        self.forward_res = nn.Sequential(
            nn.Conv2d(num_channels + 3, num_channels, 3, stride=1, padding=1),
            self.activation,
            ResidualInResidual(num_channels, residual_blocks=residual_blocks),
        )
        self.backward_res = nn.Sequential(
            nn.Conv2d(num_channels + 3, num_channels, 3, stride=1, padding=1),
            self.activation,
            ResidualInResidual(num_channels, residual_blocks=residual_blocks),
        )

        self.multi_scale_fusion = MultiScaleFusion(num_channels)
        self.adaptive_flow_refinement = AdaptiveFlowRefinement(num_channels)
        self.fusion = AttentionFusion(num_channels)

        self.up_conv1 = nn.Conv2d(num_channels, 4 * 64, 3, stride=1, padding=1)
        self.up_conv2 = nn.Conv2d(64, 4 * 64, 3, stride=1, padding=1)
        self.hr_conv = nn.Conv2d(64, 3, 3, stride=1, padding=1)

    def compute_flows(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, steps, channels, height, width = x.shape
        x_start = x[:, :-1, :, :, :].reshape(-1, channels, height, width)
        x_end = x[:, 1:, :, :, :].reshape(-1, channels, height, width)
        forward_flows = self.spynet(x_start, x_end).view(batch, steps - 1, 2, height, width)
        backward_flows = self.spynet(x_end, x_start).view(batch, steps - 1, 2, height, width)
        return forward_flows, backward_flows

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, _, height, width = x.shape
        forward_flows, backward_flows = self.compute_flows(x)

        features = x.new_zeros((batch, self.num_channels, height, width))
        backward_outputs = []  # type: List[torch.Tensor]

        for t in range(steps - 1, -1, -1):
            if t != steps - 1:
                flow = self.adaptive_flow_refinement(features, backward_flows[:, t, :, :, :])
                features = flow_warp(features, flow.permute(0, 2, 3, 1))
            features = self.multi_scale_fusion(features)
            features = torch.cat((features, x[:, t, :, :, :]), dim=1)
            features = self.backward_res(features)
            backward_outputs.append(features)

        backward_outputs = backward_outputs[::-1]

        features = x.new_zeros((batch, self.num_channels, height, width))
        for t in range(steps):
            lr_current = x[:, t, :, :, :]
            if t != 0:
                flow = self.adaptive_flow_refinement(features, forward_flows[:, t - 1, :, :, :])
                features = flow_warp(features, flow.permute(0, 2, 3, 1))
            features = self.multi_scale_fusion(features)
            features = torch.cat((features, lr_current), dim=1)
            features = self.forward_res(features)

            fused = self.fusion(features, backward_outputs[t])
            out = F.pixel_shuffle(self.activation(self.up_conv1(fused)), 2)
            out = F.pixel_shuffle(self.activation(self.up_conv2(out)), 2)
            out = self.hr_conv(out)
            out = out + F.interpolate(lr_current, scale_factor=4.0, mode="bilinear", align_corners=False)
            backward_outputs[t] = out

        return torch.stack(backward_outputs, dim=1)

    def eval_yuv420(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(rgb_through_yuv420_bt709_full_range(x))


def build_generator(
    *,
    spynet_weights: Optional[Union[str, Path]] = None,
    num_channels: int = 64,
    residual_blocks: int = 7,
) -> Generator:
    spynet = SpyNet(load_path=spynet_weights)
    return Generator(spynet, num_channels=num_channels, residual_blocks=residual_blocks)
