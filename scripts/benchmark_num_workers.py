from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from basicvsr_yuv420.data import DEFAULT_VALIDATION_CLIPS, REDSVSRDataset
from basicvsr_yuv420.data.colorspace import yuv420_to_rgb_bt709_full_range
from basicvsr_yuv420.losses import charbonnier_loss
from basicvsr_yuv420.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark training-step speed for different num_workers values.")
    parser.add_argument("--model", default="low_res_joint_y_head")
    parser.add_argument("--train-lr-dir", required=True)
    parser.add_argument("--train-hr-dir", required=True)
    parser.add_argument("--spynet-weights")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=15)
    parser.add_argument("--sequence-stride", type=int, default=15)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--num-channels", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=7)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--objective-domain", choices=("rgb", "yuv420"), default="yuv420")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=12)
    parser.add_argument("--num-workers", type=int, nargs="+", default=[0, 1, 2, 4, 6])
    return parser.parse_args()


def build_loader(args: argparse.Namespace, num_workers: int, color_mode: str) -> DataLoader:
    dataset = REDSVSRDataset(
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
    return DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch(batch: dict, device: torch.device, color_mode: str):
    if color_mode == "rgb":
        inputs = batch["lr_rgb"].float().contiguous().to(device, non_blocking=True)
        targets = batch["hr_rgb"].float().contiguous().to(device, non_blocking=True)
        return inputs, targets

    inputs = {
        "y": batch["lr_y"].float().contiguous().to(device, non_blocking=True),
        "uv": batch["lr_uv"].float().contiguous().to(device, non_blocking=True),
    }
    targets = {
        "y": batch["hr_y"].float().contiguous().to(device, non_blocking=True),
        "uv": batch["hr_uv"].float().contiguous().to(device, non_blocking=True),
    }
    if "hr_rgb" in batch:
        targets["rgb"] = batch["hr_rgb"].float().contiguous().to(device, non_blocking=True)
    return inputs, targets


def compute_loss(prediction, targets, objective_domain: str):
    if isinstance(prediction, dict):
        if objective_domain == "rgb":
            return charbonnier_loss(yuv420_to_rgb_bt709_full_range(prediction["y"], prediction["uv"]), targets["rgb"])
        loss_y = charbonnier_loss(prediction["y"], targets["y"])
        loss_uv = charbonnier_loss(prediction["uv"], targets["uv"])
        return 0.5 * (loss_y + loss_uv)
    return charbonnier_loss(prediction, targets)


def benchmark_workers(
    args: argparse.Namespace,
    *,
    num_workers: int,
    color_mode: str,
    device: torch.device,
) -> Tuple[int, float]:
    loader = build_loader(args, num_workers=num_workers, color_mode=color_mode)
    model = build_model(
        args.model,
        spynet_weights=args.spynet_weights,
        num_channels=args.num_channels,
        residual_blocks=args.residual_blocks,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    iterator = iter(loader)
    timings: List[float] = []

    for step in range(args.warmup_steps + args.measure_steps):
        batch = next(iterator)
        inputs, targets = move_batch(batch, device, color_mode)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        loss = compute_loss(prediction, targets, args.objective_domain)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if step >= args.warmup_steps:
            timings.append(elapsed)

    mean_step = sum(timings) / len(timings)
    print(f"num_workers={num_workers} mean_step={mean_step:.4f}s it_per_sec={1.0 / mean_step:.4f}", flush=True)
    del model, optimizer, loader, iterator
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return num_workers, mean_step


def resolve_color_mode(model_id: str) -> str:
    if model_id == "basicvsr_rgb_baseline":
        return "rgb"
    return "yuv420"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    color_mode = resolve_color_mode(args.model)
    print(f"device={device}", flush=True)

    results = [
        benchmark_workers(args, num_workers=num_workers, color_mode=color_mode, device=device)
        for num_workers in args.num_workers
    ]
    best_workers, best_mean_step = min(results, key=lambda item: item[1])
    print(f"summary={results}", flush=True)
    print(f"best_num_workers={best_workers} best_mean_step={best_mean_step:.4f}s", flush=True)


if __name__ == "__main__":
    main()
