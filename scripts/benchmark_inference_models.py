from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from basicvsr_yuv420.inference import load_frame_sequence
from basicvsr_yuv420.models import build_model, get_model_spec, list_model_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark forward inference speed and peak memory for all models.")
    parser.add_argument("--input-dir", required=True, help="Directory containing a sequence of LR frames.")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to benchmark from the start of the clip.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--spynet-weights")
    parser.add_argument("--num-channels", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=7)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measure-runs", type=int, default=5)
    parser.add_argument("--amp", action="store_true", help="Benchmark inference with autocast enabled.")
    parser.add_argument("--models", nargs="+", default=list(list_model_ids()))
    return parser.parse_args()


def reset_device_state(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        if torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)


def slice_sequence(sequence: Union[torch.Tensor, Dict[str, torch.Tensor]], frames: int):
    if isinstance(sequence, dict):
        return {key: value[:, :frames].contiguous() for key, value in sequence.items()}
    return sequence[:, :frames].contiguous()


def move_sequence_to_device(sequence: Union[torch.Tensor, Dict[str, torch.Tensor]], device: torch.device):
    if isinstance(sequence, dict):
        return {key: value.to(device, non_blocking=True) for key, value in sequence.items()}
    return sequence.to(device, non_blocking=True)


@torch.no_grad()
def benchmark_model(
    *,
    model_id: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    spec = get_model_spec(model_id)
    sequence, reference_paths = load_frame_sequence(args.input_dir, color_mode=spec.input_format)
    if args.frames > len(reference_paths):
        raise ValueError(f"Requested {args.frames} frames, but only found {len(reference_paths)} frames in {args.input_dir}.")
    sequence = slice_sequence(sequence, args.frames)
    sequence = move_sequence_to_device(sequence, device)

    reset_device_state(device)
    model = build_model(
        model_id,
        spynet_weights=args.spynet_weights,
        num_channels=args.num_channels,
        residual_blocks=args.residual_blocks,
    ).to(device)
    model.eval()

    timings: List[float] = []
    amp_enabled = args.amp and device.type == "cuda"

    for run_index in range(args.warmup_runs + args.measure_runs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            _ = model(sequence)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if run_index >= args.warmup_runs:
            timings.append(elapsed)

    mean_latency = sum(timings) / len(timings)
    peak_memory_gb = 0.0
    if device.type == "cuda":
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    result = {
        "mean_sequence_latency_seconds": mean_latency,
        "frames_per_second": args.frames / mean_latency,
        "peak_memory_gb": peak_memory_gb,
        "frames": float(args.frames),
        "measure_runs": float(args.measure_runs),
    }

    del model, sequence
    reset_device_state(device)
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(json.dumps({"device": str(device), "frames": args.frames, "amp": args.amp}, indent=2), flush=True)

    results = {}
    for model_id in args.models:
        result = benchmark_model(model_id=model_id, args=args, device=device)
        results[model_id] = result
        print(
            json.dumps(
                {
                    "model": model_id,
                    **result,
                },
                indent=2,
            ),
            flush=True,
        )

    print(json.dumps({"summary": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
