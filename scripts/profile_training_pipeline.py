from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from basicvsr_yuv420.data import DEFAULT_VALIDATION_CLIPS, REDSVSRDataset
from basicvsr_yuv420.losses import charbonnier_loss
from basicvsr_yuv420.models import build_model, get_model_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile fetch, transfer, and compute times for a training pipeline.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-lr-dir", required=True)
    parser.add_argument("--train-hr-dir", required=True)
    parser.add_argument("--spynet-weights")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--amp-modes", nargs="+", default=["off", "on"])
    parser.add_argument("--sequence-length", type=int, default=15)
    parser.add_argument("--sequence-stride", type=int, default=15)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--num-channels", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=7)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=6)
    return parser.parse_args()


def reset_cuda_state(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def build_dataset(args: argparse.Namespace, color_mode: str) -> REDSVSRDataset:
    return REDSVSRDataset(
        args.train_lr_dir,
        args.train_hr_dir,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        patch_size=args.patch_size,
        scale=args.scale,
        train=True,
        exclude_clips=DEFAULT_VALIDATION_CLIPS,
        color_mode=color_mode,
    )


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device, color_mode: str):
    prepared = {
        key: value.float().contiguous().to(device, non_blocking=True)
        for key, value in batch.items()
    }
    if color_mode == "rgb":
        return prepared["lr_rgb"], prepared["hr_rgb"]
    return {"y": prepared["lr_y"], "uv": prepared["lr_uv"]}, {"y": prepared["hr_y"], "uv": prepared["hr_uv"]}


def compute_loss(prediction, target) -> torch.Tensor:
    if isinstance(prediction, dict):
        return 0.5 * (
            charbonnier_loss(prediction["y"], target["y"]) +
            charbonnier_loss(prediction["uv"], target["uv"])
        )
    return charbonnier_loss(prediction, target)


def profile_configuration(
    args: argparse.Namespace,
    *,
    device: torch.device,
    dataset: REDSVSRDataset,
    color_mode: str,
    num_workers: int,
    amp_enabled: bool,
) -> Dict[str, float]:
    reset_cuda_state(device)
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model(
        args.model,
        spynet_weights=args.spynet_weights,
        num_channels=args.num_channels,
        residual_blocks=args.residual_blocks,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scaler = GradScaler(enabled=amp_enabled and device.type == "cuda")
    model.train()

    fetch_times: List[float] = []
    h2d_times: List[float] = []
    compute_times: List[float] = []
    step_times: List[float] = []

    try:
        iterator = iter(loader)
        for step in range(args.warmup_steps + args.measure_steps):
            start_fetch = time.perf_counter()
            batch = next(iterator)
            after_fetch = time.perf_counter()
            inputs, targets = move_batch(batch, device, color_mode)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            after_h2d = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            start_compute = time.perf_counter()
            with autocast(enabled=amp_enabled and device.type == "cuda"):
                prediction = model(inputs)
                loss = compute_loss(prediction, targets)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            end_compute = time.perf_counter()

            if step >= args.warmup_steps:
                fetch_times.append(after_fetch - start_fetch)
                h2d_times.append(after_h2d - after_fetch)
                compute_times.append(end_compute - start_compute)
                step_times.append(end_compute - start_fetch)
    finally:
        peak_memory_gb = None
        if device.type == "cuda":
            peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        del model, optimizer, loader
        if "iterator" in locals():
            del iterator
        reset_cuda_state(device)

    mean_step = sum(step_times) / len(step_times)
    return {
        "num_workers": num_workers,
        "amp": amp_enabled,
        "mean_fetch_s": round(sum(fetch_times) / len(fetch_times), 4),
        "mean_h2d_s": round(sum(h2d_times) / len(h2d_times), 4),
        "mean_compute_s": round(sum(compute_times) / len(compute_times), 4),
        "mean_step_s": round(mean_step, 4),
        "samples_per_sec": round(args.train_batch_size / mean_step, 4),
        "peak_memory_gb": None if peak_memory_gb is None else round(peak_memory_gb, 2),
    }


def main() -> None:
    args = parse_args()
    model_spec = get_model_spec(args.model)
    color_mode = model_spec.input_format
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.zeros(1, device=device)

    dataset = build_dataset(args, color_mode=color_mode)
    results = []
    for num_workers in args.num_workers:
        for amp_mode in args.amp_modes:
            amp_enabled = amp_mode.lower() == "on"
            results.append(
                profile_configuration(
                    args,
                    device=device,
                    dataset=dataset,
                    color_mode=color_mode,
                    num_workers=num_workers,
                    amp_enabled=amp_enabled,
                )
            )

    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
