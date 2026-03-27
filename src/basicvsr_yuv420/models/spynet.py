from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from .common import flow_warp


class BasicModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.basic_module = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 2, kernel_size=7, stride=1, padding=3),
        )

    def forward(self, tensor_input: torch.Tensor) -> torch.Tensor:
        return self.basic_module(tensor_input)


class SpyNet(nn.Module):
    def __init__(self, load_path: Optional[Union[str, Path]] = None) -> None:
        super().__init__()
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if load_path:
            self.load_pretrained(load_path)

    def load_pretrained(self, load_path: Union[str, Path]) -> None:
        try:
            checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(load_path, map_location="cpu")
        state_dict = checkpoint["params"] if isinstance(checkpoint, dict) and "params" in checkpoint else checkpoint
        self.load_state_dict(state_dict, strict=False)

    def preprocess(self, tensor_input: torch.Tensor) -> torch.Tensor:
        return (tensor_input - self.mean) / self.std

    def process(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        ref_pyramid = [self.preprocess(ref)]
        supp_pyramid = [self.preprocess(supp)]

        for _ in range(5):
            ref_pyramid.insert(
                0,
                F.avg_pool2d(ref_pyramid[0], kernel_size=2, stride=2, count_include_pad=False),
            )
            supp_pyramid.insert(
                0,
                F.avg_pool2d(supp_pyramid[0], kernel_size=2, stride=2, count_include_pad=False),
            )

        flow = ref_pyramid[0].new_zeros(
            [
                ref_pyramid[0].size(0),
                2,
                int(math.floor(ref_pyramid[0].size(2) / 2.0)),
                int(math.floor(ref_pyramid[0].size(3) / 2.0)),
            ]
        )

        for level in range(len(ref_pyramid)):
            upsampled_flow = F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0

            if upsampled_flow.size(2) != ref_pyramid[level].size(2):
                upsampled_flow = F.pad(upsampled_flow, pad=[0, 0, 0, 1], mode="replicate")
            if upsampled_flow.size(3) != ref_pyramid[level].size(3):
                upsampled_flow = F.pad(upsampled_flow, pad=[0, 1, 0, 0], mode="replicate")

            warped = flow_warp(
                supp_pyramid[level],
                upsampled_flow.permute(0, 2, 3, 1),
                interp_mode="bilinear",
                padding_mode="border",
            )
            flow = self.basic_module[level](torch.cat([ref_pyramid[level], warped, upsampled_flow], dim=1))
            flow = flow + upsampled_flow

        return flow

    def forward(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        if ref.shape != supp.shape:
            raise ValueError(f"Mismatched input shapes: {ref.shape} vs {supp.shape}")

        height, width = ref.size(2), ref.size(3)
        width_floor = math.floor(math.ceil(width / 32.0) * 32.0)
        height_floor = math.floor(math.ceil(height / 32.0) * 32.0)

        ref = F.interpolate(ref, size=(height_floor, width_floor), mode="bilinear", align_corners=False)
        supp = F.interpolate(supp, size=(height_floor, width_floor), mode="bilinear", align_corners=False)

        flow = F.interpolate(self.process(ref, supp), size=(height, width), mode="bilinear", align_corners=False)
        flow[:, 0, :, :] *= float(width) / float(width_floor)
        flow[:, 1, :, :] *= float(height) / float(height_floor)
        return flow
