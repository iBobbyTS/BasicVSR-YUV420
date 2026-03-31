from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
from basicvsr_yuv420.models import build_model, get_model_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark train and eval batch sizes for a selected model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-lr-dir", required=True)
    parser.add_argument("--train-hr-dir", required=True)
    parser.add_argument("--val-lr-dir")
    parser.add_argument("--val-hr-dir")
    parser.add_argument("--spynet-weights")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=15)
    parser.add_argument("--sequence-stride", type=int, default=15)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--num-channels", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=7)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--objective-domain", choices=("rgb", "yuv420"), default="yuv420")
    parser.add_argument("--train-batch-sizes", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--eval-batch-sizes", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=8)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def build_dataset(
    *,
    lr_dir: str,
    hr_dir: str,
    train: bool,
    color_mode: str,
    args: argparse.Namespace,
) -> REDSVSRDataset:
    include_clips = None
    exclude_clips = None
    if train:
        exclude_clips = DEFAULT_VALIDATION_CLIPS
    elif not args.val_lr_dir or not args.val_hr_dir:
        include_clips = DEFAULT_VALIDATION_CLIPS

    return REDSVSRDataset(
        lr_dir,
        hr_dir,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        patch_size=args.patch_size if train else None,
        scale=args.scale,
        train=train,
        include_clips=include_clips,
        exclude_clips=exclude_clips,
        color_mode=color_mode,
    )


def build_loader(dataset: REDSVSRDataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device, color_mode: str):
    prepared = {
        key: value.float().contiguous().to(device, non_blocking=True)
        for key, value in batch.items()
    }
    if color_mode == "rgb":
        return prepared["lr_rgb"], prepared["hr_rgb"]
    targets = {"y": prepared["hr_y"], "uv": prepared["hr_uv"]}
    if "hr_rgb" in prepared:
        targets["rgb"] = prepared["hr_rgb"]
    return {"y": prepared["lr_y"], "uv": prepared["lr_uv"]}, targets


def compute_loss(prediction, target, objective_domain: str) -> torch.Tensor:
    if isinstance(prediction, dict):
        if objective_domain == "rgb":
            return charbonnier_loss(yuv420_to_rgb_bt709_full_range(prediction["y"], prediction["uv"]), target["rgb"])
        loss_y = charbonnier_loss(prediction["y"], target["y"])
        loss_uv = charbonnier_loss(prediction["uv"], target["uv"])
        return 0.5 * (loss_y + loss_uv)
    return charbonnier_loss(prediction, target)


def reset_cuda_state(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        if torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)


def benchmark_train(
    args: argparse.Namespace,
    *,
    batch_size: int,
    device: torch.device,
    color_mode: str,
    dataset: REDSVSRDataset,
) -> Tuple[bool, Optional[float], Optional[float], Optional[float], Optional[str]]:
    reset_cuda_state(device)
    loader = build_loader(dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=True)
    model = build_model(
        args.model,
        spynet_weights=args.spynet_weights,
        num_channels=args.num_channels,
        residual_blocks=args.residual_blocks,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    model.train()

    timings: List[float] = []
    peak_memory_gb = None
    try:
        iterator = iter(loader)
        for step in range(args.warmup_steps + args.measure_steps):
            batch = next(iterator)
            inputs, targets = move_batch(batch, device, color_mode)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = compute_loss(prediction, targets, args.objective_domain)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            if step >= args.warmup_steps:
                timings.append(elapsed)
        if device.type == "cuda":
            peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    except RuntimeError as error:
        message = str(error)
        if "out of memory" in message.lower():
            return False, None, None, None, message
        raise
    finally:
        del model, optimizer, loader
        if "iterator" in locals():
            del iterator
        reset_cuda_state(device)

    mean_step = sum(timings) / len(timings)
    samples_per_sec = batch_size / mean_step
    return True, mean_step, samples_per_sec, peak_memory_gb, None


@torch.no_grad()
def benchmark_eval(
    args: argparse.Namespace,
    *,
    batch_size: int,
    device: torch.device,
    color_mode: str,
    dataset: REDSVSRDataset,
) -> Tuple[bool, Optional[float], Optional[float], Optional[float], Optional[str]]:
    reset_cuda_state(device)
    loader = build_loader(dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=False)
    model = build_model(
        args.model,
        spynet_weights=args.spynet_weights,
        num_channels=args.num_channels,
        residual_blocks=args.residual_blocks,
    ).to(device)
    model.eval()

    timings: List[float] = []
    peak_memory_gb = None
    try:
        iterator = iter(loader)
        for step in range(args.warmup_steps + args.measure_steps):
            batch = next(iterator)
            inputs, targets = move_batch(batch, device, color_mode)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            prediction = model(inputs)
            _ = compute_loss(prediction, targets, args.objective_domain)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            if step >= args.warmup_steps:
                timings.append(elapsed)
        if device.type == "cuda":
            peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    except RuntimeError as error:
        message = str(error)
        if "out of memory" in message.lower():
            return False, None, None, None, message
        raise
    finally:
        del model, loader
        if "iterator" in locals():
            del iterator
        reset_cuda_state(device)

    mean_step = sum(timings) / len(timings)
    samples_per_sec = batch_size / mean_step
    return True, mean_step, samples_per_sec, peak_memory_gb, None


def main() -> None:
    args = parse_args()
    model_spec = get_model_spec(args.model)
    color_mode = model_spec.input_format
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"num_workers={args.num_workers}", flush=True)

    train_dataset = build_dataset(
        lr_dir=args.train_lr_dir,
        hr_dir=args.train_hr_dir,
        train=True,
        color_mode=color_mode,
        args=args,
    )
    if args.val_lr_dir and args.val_hr_dir:
        eval_lr_dir = args.val_lr_dir
        eval_hr_dir = args.val_hr_dir
    else:
        eval_lr_dir = args.train_lr_dir
        eval_hr_dir = args.train_hr_dir
    eval_dataset = build_dataset(
        lr_dir=eval_lr_dir,
        hr_dir=eval_hr_dir,
        train=False,
        color_mode=color_mode,
        args=args,
    )

    if not args.skip_train:
        print("train_benchmark", flush=True)
        for batch_size in args.train_batch_sizes:
            ok, mean_step, samples_per_sec, peak_memory_gb, error = benchmark_train(
                args,
                batch_size=batch_size,
                device=device,
                color_mode=color_mode,
                dataset=train_dataset,
            )
            if ok:
                peak_str = "n/a" if peak_memory_gb is None else f"{peak_memory_gb:.2f}GB"
                print(
                    f"train batch_size={batch_size} mean_step={mean_step:.4f}s samples_per_sec={samples_per_sec:.4f} peak_memory={peak_str}",
                    flush=True,
                )
            else:
                print(f"train batch_size={batch_size} OOM error={error}", flush=True)
                break

    if not args.skip_eval:
        print("eval_benchmark", flush=True)
        for batch_size in args.eval_batch_sizes:
            ok, mean_step, samples_per_sec, peak_memory_gb, error = benchmark_eval(
                args,
                batch_size=batch_size,
                device=device,
                color_mode=color_mode,
                dataset=eval_dataset,
            )
            if ok:
                peak_str = "n/a" if peak_memory_gb is None else f"{peak_memory_gb:.2f}GB"
                print(
                    f"eval batch_size={batch_size} mean_step={mean_step:.4f}s samples_per_sec={samples_per_sec:.4f} peak_memory={peak_str}",
                    flush=True,
                )
            else:
                print(f"eval batch_size={batch_size} OOM error={error}", flush=True)
                break


if __name__ == "__main__":
    main()
