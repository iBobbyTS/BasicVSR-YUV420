from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..yuv_common import (
    PixelShuffleUpsampleHead,
    RecurrentPropagation,
    ResidualStack,
    build_shared_spynet,
    compute_bidirectional_flows_from_luma,
    resize_flow,
    split_luma_low_high,
    warp_feature,
)


class FrequencyDomainLowFrequencyFusionV2Model(nn.Module):
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
        self.high_channels = max(num_channels // 4, 16)
        self.high_blocks = max(1, residual_blocks // 2)

        self.low_backward = RecurrentPropagation(3, num_channels, residual_blocks)
        self.low_forward = RecurrentPropagation(3, num_channels, residual_blocks)
        self.low_fusion = nn.Sequential(
            nn.Conv2d(num_channels * 2, num_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualStack(num_channels, max(1, residual_blocks // 2)),
        )

        # Keep the high-frequency branch narrow so it adds temporal detail without
        # collapsing the inference/memory advantage of the low-resolution backbone.
        self.high_backward = RecurrentPropagation(1, self.high_channels, self.high_blocks)
        self.high_forward = RecurrentPropagation(1, self.high_channels, self.high_blocks)
        self.high_fusion = nn.Sequential(
            nn.Conv2d(self.high_channels * 2, self.high_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualStack(self.high_channels, self.high_blocks),
        )
        self.high_to_y = nn.Sequential(
            nn.Conv2d(self.high_channels, num_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.low_to_y = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.y_refine = nn.Sequential(
            nn.Conv2d(num_channels * 2 + 1, num_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualStack(num_channels, max(1, residual_blocks // 2)),
        )
        self.y_head = PixelShuffleUpsampleHead(num_channels, num_channels, 1)
        self.uv_head = PixelShuffleUpsampleHead(num_channels, max(num_channels // 2, 16), 2)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        y = inputs["y"]
        uv = inputs["uv"]
        batch, steps, _, height, width = y.shape
        uv_size = uv.shape[-2:]

        forward_y, backward_y = compute_bidirectional_flows_from_luma(self.spynet, y)
        forward_uv = torch.stack([resize_flow(flow, uv_size) for flow in forward_y.unbind(dim=1)], dim=1)
        backward_uv = torch.stack([resize_flow(flow, uv_size) for flow in backward_y.unbind(dim=1)], dim=1)

        low_sequence: List[torch.Tensor] = []
        high_sequence: List[torch.Tensor] = []
        for step in range(steps):
            low, high = split_luma_low_high(y[:, step, :, :, :])
            low_sequence.append(low)
            high_sequence.append(high)

        low_state = uv.new_zeros((batch, self.num_channels, uv_size[0], uv_size[1]))
        high_state = y.new_zeros((batch, self.high_channels, height, width))
        low_backward_features: List[torch.Tensor] = []
        high_backward_features: List[torch.Tensor] = []
        for step in range(steps - 1, -1, -1):
            if step != steps - 1:
                low_state = warp_feature(low_state, backward_uv[:, step, :, :, :])
                high_state = warp_feature(high_state, backward_y[:, step, :, :, :])
            low_joint_input = torch.cat((low_sequence[step], uv[:, step, :, :, :]), dim=1)
            low_state = self.low_backward(low_joint_input, low_state)
            high_state = self.high_backward(high_sequence[step], high_state)
            low_backward_features.append(low_state)
            high_backward_features.append(high_state)
        low_backward_features.reverse()
        high_backward_features.reverse()

        low_state = uv.new_zeros((batch, self.num_channels, uv_size[0], uv_size[1]))
        high_state = y.new_zeros((batch, self.high_channels, height, width))
        outputs_y: List[torch.Tensor] = []
        outputs_uv: List[torch.Tensor] = []
        for step in range(steps):
            current_y = y[:, step, :, :, :]
            current_uv = uv[:, step, :, :, :]
            if step != 0:
                low_state = warp_feature(low_state, forward_uv[:, step - 1, :, :, :])
                high_state = warp_feature(high_state, forward_y[:, step - 1, :, :, :])

            low_joint_input = torch.cat((low_sequence[step], current_uv), dim=1)
            low_state = self.low_forward(low_joint_input, low_state)
            low_feature = self.low_fusion(torch.cat((low_state, low_backward_features[step]), dim=1))

            high_state = self.high_forward(high_sequence[step], high_state)
            high_feature = self.high_fusion(torch.cat((high_state, high_backward_features[step]), dim=1))

            low_context = F.interpolate(self.low_to_y(low_feature), scale_factor=2.0, mode="bilinear", align_corners=False)
            high_context = self.high_to_y(high_feature)
            y_feature = self.y_refine(torch.cat((low_context, high_context, current_y), dim=1))

            outputs_y.append(
                self.y_head(y_feature) + F.interpolate(current_y, scale_factor=4.0, mode="bilinear", align_corners=False)
            )
            outputs_uv.append(
                self.uv_head(low_feature)
                + F.interpolate(current_uv, scale_factor=4.0, mode="bilinear", align_corners=False)
            )

        return {
            "y": torch.stack(outputs_y, dim=1),
            "uv": torch.stack(outputs_uv, dim=1),
        }


def build_frequency_domain_low_frequency_fusion_v2(
    *,
    spynet_weights: Optional[Union[str, Path]] = None,
    num_channels: int = 64,
    residual_blocks: int = 7,
) -> FrequencyDomainLowFrequencyFusionV2Model:
    return FrequencyDomainLowFrequencyFusionV2Model(
        spynet=build_shared_spynet(spynet_weights),
        num_channels=num_channels,
        residual_blocks=residual_blocks,
    )
