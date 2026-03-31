from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..yuv_common import (
    FiLMConditioner,
    PixelShuffleUpsampleHead,
    RecurrentPropagation,
    ResidualStack,
    build_shared_spynet,
    compute_bidirectional_flows_from_luma,
    resize_flow,
    warp_feature,
)


class UVConditionedFiLMModel(nn.Module):
    def __init__(
        self,
        *,
        spynet,
        num_channels: int = 64,
        residual_blocks: int = 7,
    ) -> None:
        super().__init__()
        self.spynet = spynet
        self.num_channels = num_channels
        self.uv_channels = max(num_channels // 2, 16)
        self.y_backward = RecurrentPropagation(1, num_channels, residual_blocks)
        self.y_forward = RecurrentPropagation(1, num_channels, residual_blocks)
        self.uv_backward = RecurrentPropagation(2, self.uv_channels, max(1, residual_blocks // 2))
        self.uv_forward = RecurrentPropagation(2, self.uv_channels, max(1, residual_blocks // 2))
        self.backward_film = FiLMConditioner(self.uv_channels, num_channels)
        self.forward_film = FiLMConditioner(self.uv_channels, num_channels)
        self.y_fusion = nn.Sequential(
            nn.Conv2d(num_channels * 2, num_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualStack(num_channels, max(1, residual_blocks // 2)),
        )
        self.uv_fusion = nn.Sequential(
            nn.Conv2d(self.uv_channels * 2, self.uv_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualStack(self.uv_channels, max(1, residual_blocks // 2)),
        )
        self.y_head = PixelShuffleUpsampleHead(num_channels, num_channels, 1)
        self.uv_head = PixelShuffleUpsampleHead(self.uv_channels, self.uv_channels, 2)

    def _compute_flows(self, y: torch.Tensor, uv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        forward_y, backward_y = compute_bidirectional_flows_from_luma(self.spynet, y)
        uv_size = uv.shape[-2:]
        forward_uv = torch.stack([resize_flow(flow, uv_size) for flow in forward_y.unbind(dim=1)], dim=1)
        backward_uv = torch.stack([resize_flow(flow, uv_size) for flow in backward_y.unbind(dim=1)], dim=1)
        return forward_y, backward_y, forward_uv, backward_uv

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        y = inputs["y"]
        uv = inputs["uv"]
        batch, steps, _, height, width = y.shape
        uv_height, uv_width = uv.shape[-2:]
        forward_y, backward_y, forward_uv, backward_uv = self._compute_flows(y, uv)

        y_state = y.new_zeros((batch, self.num_channels, height, width))
        uv_state = uv.new_zeros((batch, self.uv_channels, uv_height, uv_width))
        backward_y_features: List[torch.Tensor] = []
        backward_uv_features: List[torch.Tensor] = []

        for step in range(steps - 1, -1, -1):
            if step != steps - 1:
                y_state = warp_feature(y_state, backward_y[:, step, :, :, :])
                uv_state = warp_feature(uv_state, backward_uv[:, step, :, :, :])
            y_state = self.y_backward(y[:, step, :, :, :], y_state)
            uv_state = self.uv_backward(uv[:, step, :, :, :], uv_state)
            y_state = self.backward_film(y_state, uv_state)
            backward_y_features.append(y_state)
            backward_uv_features.append(uv_state)

        backward_y_features.reverse()
        backward_uv_features.reverse()

        y_state = y.new_zeros((batch, self.num_channels, height, width))
        uv_state = uv.new_zeros((batch, self.uv_channels, uv_height, uv_width))
        outputs_y: List[torch.Tensor] = []
        outputs_uv: List[torch.Tensor] = []

        for step in range(steps):
            current_y = y[:, step, :, :, :]
            current_uv = uv[:, step, :, :, :]
            if step != 0:
                y_state = warp_feature(y_state, forward_y[:, step - 1, :, :, :])
                uv_state = warp_feature(uv_state, forward_uv[:, step - 1, :, :, :])
            y_state = self.y_forward(current_y, y_state)
            uv_state = self.uv_forward(current_uv, uv_state)
            y_state = self.forward_film(y_state, uv_state)

            fused_y = self.y_fusion(torch.cat((y_state, backward_y_features[step]), dim=1))
            fused_uv = self.uv_fusion(torch.cat((uv_state, backward_uv_features[step]), dim=1))
            outputs_y.append(
                self.y_head(fused_y) + F.interpolate(current_y, scale_factor=4.0, mode="bilinear", align_corners=False)
            )
            outputs_uv.append(
                self.uv_head(fused_uv)
                + F.interpolate(current_uv, scale_factor=4.0, mode="bilinear", align_corners=False)
            )

        return {
            "y": torch.stack(outputs_y, dim=1),
            "uv": torch.stack(outputs_uv, dim=1),
        }


def build_uv_conditioned_film(
    *,
    spynet_weights: Optional[Union[str, Path]] = None,
    num_channels: int = 64,
    residual_blocks: int = 7,
) -> UVConditionedFiLMModel:
    return UVConditionedFiLMModel(
        spynet=build_shared_spynet(spynet_weights),
        num_channels=num_channels,
        residual_blocks=residual_blocks,
    )
