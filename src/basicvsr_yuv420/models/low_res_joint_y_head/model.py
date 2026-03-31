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
    downsample_luma_to_uv,
    resize_flow,
    warp_feature,
)


class LowResJointYHeadModel(nn.Module):
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
        self.low_res_backward = RecurrentPropagation(3, num_channels, residual_blocks)
        self.low_res_forward = RecurrentPropagation(3, num_channels, residual_blocks)
        self.low_res_fusion = nn.Sequential(
            nn.Conv2d(num_channels * 2, num_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualStack(num_channels, max(1, residual_blocks // 2)),
        )
        self.y_shallow = nn.Sequential(
            nn.Conv2d(1, num_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.y_merge = nn.Sequential(
            nn.Conv2d(num_channels * 2, num_channels, 3, stride=1, padding=1),
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
        y_low = downsample_luma_to_uv(y.reshape(-1, 1, height, width)).reshape(batch, steps, 1, *uv_size)

        state = uv.new_zeros((batch, self.num_channels, uv_size[0], uv_size[1]))
        backward_features: List[torch.Tensor] = []
        for step in range(steps - 1, -1, -1):
            if step != steps - 1:
                state = warp_feature(state, backward_uv[:, step, :, :, :])
            joint_input = torch.cat((y_low[:, step, :, :, :], uv[:, step, :, :, :]), dim=1)
            state = self.low_res_backward(joint_input, state)
            backward_features.append(state)
        backward_features.reverse()

        state = uv.new_zeros((batch, self.num_channels, uv_size[0], uv_size[1]))
        outputs_y: List[torch.Tensor] = []
        outputs_uv: List[torch.Tensor] = []
        for step in range(steps):
            current_y = y[:, step, :, :, :]
            current_uv = uv[:, step, :, :, :]
            if step != 0:
                state = warp_feature(state, forward_uv[:, step - 1, :, :, :])
            joint_input = torch.cat((y_low[:, step, :, :, :], current_uv), dim=1)
            state = self.low_res_forward(joint_input, state)
            joint_feature = self.low_res_fusion(torch.cat((state, backward_features[step]), dim=1))

            y_context = F.interpolate(joint_feature, scale_factor=2.0, mode="bilinear", align_corners=False)
            y_feature = self.y_merge(torch.cat((self.y_shallow(current_y), y_context), dim=1))

            outputs_y.append(
                self.y_head(y_feature) + F.interpolate(current_y, scale_factor=4.0, mode="bilinear", align_corners=False)
            )
            outputs_uv.append(
                self.uv_head(joint_feature)
                + F.interpolate(current_uv, scale_factor=4.0, mode="bilinear", align_corners=False)
            )

        return {
            "y": torch.stack(outputs_y, dim=1),
            "uv": torch.stack(outputs_uv, dim=1),
        }


def build_low_res_joint_y_head(
    *,
    spynet_weights: Optional[Union[str, Path]] = None,
    num_channels: int = 64,
    residual_blocks: int = 7,
) -> LowResJointYHeadModel:
    return LowResJointYHeadModel(
        spynet=build_shared_spynet(spynet_weights),
        num_channels=num_channels,
        residual_blocks=residual_blocks,
    )
