from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch.nn as nn

from .basicvsr_rgb_baseline import build_generator as build_basicvsr_rgb_baseline
from .frequency_domain_low_frequency_fusion import build_frequency_domain_low_frequency_fusion
from .frequency_domain_low_frequency_fusion_v2 import build_frequency_domain_low_frequency_fusion_v2
from .low_res_joint_y_head import build_low_res_joint_y_head
from .uv_conditioned_film import build_uv_conditioned_film


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    input_format: str
    output_format: str
    metric_domain: str
    builder: Callable[..., nn.Module]


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "basicvsr_rgb_baseline": ModelSpec(
        model_id="basicvsr_rgb_baseline",
        input_format="rgb",
        output_format="rgb",
        metric_domain="rgb",
        builder=build_basicvsr_rgb_baseline,
    ),
    "low_res_joint_y_head": ModelSpec(
        model_id="low_res_joint_y_head",
        input_format="yuv420",
        output_format="yuv420",
        metric_domain="yuv420",
        builder=build_low_res_joint_y_head,
    ),
    "frequency_domain_low_frequency_fusion": ModelSpec(
        model_id="frequency_domain_low_frequency_fusion",
        input_format="yuv420",
        output_format="yuv420",
        metric_domain="yuv420",
        builder=build_frequency_domain_low_frequency_fusion,
    ),
    "frequency_domain_low_frequency_fusion_v2": ModelSpec(
        model_id="frequency_domain_low_frequency_fusion_v2",
        input_format="yuv420",
        output_format="yuv420",
        metric_domain="yuv420",
        builder=build_frequency_domain_low_frequency_fusion_v2,
    ),
    "uv_conditioned_film": ModelSpec(
        model_id="uv_conditioned_film",
        input_format="yuv420",
        output_format="yuv420",
        metric_domain="rgb",
        builder=build_uv_conditioned_film,
    ),
}


def get_model_spec(model_id: str) -> ModelSpec:
    if model_id not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{model_id}'. Available models: {available}")
    return MODEL_REGISTRY[model_id]


def build_model(model_id: str, **kwargs) -> nn.Module:
    return get_model_spec(model_id).builder(**kwargs)


def list_model_ids() -> Tuple[str, ...]:
    return tuple(sorted(MODEL_REGISTRY))
