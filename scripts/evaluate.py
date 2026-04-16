from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from basicvsr_yuv420.checkpoints import load_model_weights
from basicvsr_yuv420.data import DEFAULT_VALIDATION_CLIPS, REDSVSRDataset
from basicvsr_yuv420.engine import evaluate, evaluate_with_statistics
from basicvsr_yuv420.models import build_model, get_model_spec, list_model_ids
from basicvsr_yuv420.utils import resolve_device


def count_clip_frames(root: str, clip_names: list[str]) -> int:
    base_dir = Path(root)
    total = 0
    for clip_name in clip_names:
        total += sum(1 for path in (base_dir / clip_name).iterdir() if path.is_file())
    return total


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate BasicVSR from a checkpoint.")
    parser.add_argument("--lr-dir", required=True)
    parser.add_argument("--hr-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="basicvsr_rgb_baseline", choices=list_model_ids())
    parser.add_argument("--spynet-weights")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=15)
    parser.add_argument("--sequence-stride", type=int, default=15)
    parser.add_argument("--num-channels", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=7)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--use-default-reds-split", action="store_true")
    parser.add_argument(
        "--metric-domain",
        choices=("model_default", "rgb", "yuv420"),
        default="model_default",
        help="Primary metric domain for evaluation.",
    )
    parser.add_argument(
        "--rgb-eval-yuv420",
        action="store_true",
        help="For the RGB baseline, pass LR input through RGB->YUV420->RGB before inference.",
    )
    parser.add_argument(
        "--rgb-input-mode",
        choices=("rgb", "rgb_yuv420_rgb"),
        default="rgb",
        help="Apply RGB input preprocessing in the dataset pipeline for RGB-input models.",
    )
    parser.add_argument(
        "--stats-output",
        help="Optional path to write evaluation metadata plus mean/std statistics as JSON.",
    )
    return parser


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    device = resolve_device(args.device)
    model_spec = get_model_spec(args.model)
    metric_domain = model_spec.metric_domain if args.metric_domain == "model_default" else args.metric_domain
    if model_spec.output_format == "rgb" and metric_domain != "rgb":
        raise ValueError("RGB-output models only support rgb metric domain.")
    if model_spec.output_format == "yuv420" and metric_domain not in {"rgb", "yuv420"}:
        raise ValueError("YUV420-output models only support rgb or yuv420 metric domains.")
    if model_spec.input_format != "rgb" and args.rgb_input_mode != "rgb":
        raise ValueError("--rgb-input-mode is only supported for RGB-input models.")
    if args.rgb_eval_yuv420 and args.rgb_input_mode != "rgb":
        raise ValueError("--rgb-eval-yuv420 cannot be combined with --rgb-input-mode rgb_yuv420_rgb.")
    if args.stats_output and args.batch_size != 1:
        raise ValueError("--stats-output requires --batch-size 1 so statistics are computed over evaluation windows.")
    include_clips = DEFAULT_VALIDATION_CLIPS if args.use_default_reds_split else None

    dataset = REDSVSRDataset(
        args.lr_dir,
        args.hr_dir,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        patch_size=None,
        scale=args.scale,
        train=False,
        include_clips=include_clips,
        color_mode=model_spec.input_format,
        rgb_input_mode=args.rgb_input_mode,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        args.model,
        spynet_weights=args.spynet_weights,
        num_channels=args.num_channels,
        residual_blocks=args.residual_blocks,
    ).to(device)
    checkpoint_state = load_model_weights(args.checkpoint, model, map_location="cpu")

    forward_fn = None
    if args.rgb_eval_yuv420:
        if model_spec.input_format != "rgb":
            raise ValueError("--rgb-eval-yuv420 is only supported for RGB-input models.")
        if not hasattr(model, "eval_yuv420"):
            raise ValueError(f"Model '{args.model}' does not implement eval_yuv420().")
        forward_fn = model.eval_yuv420

    if args.stats_output:
        stats = evaluate_with_statistics(
            model,
            dataloader,
            device,
            forward_fn=forward_fn,
            metric_domain=metric_domain,
        )
        results = stats.result
    else:
        stats = None
        results = evaluate(model, dataloader, device, forward_fn=forward_fn, metric_domain=metric_domain)

    first_sample = dataset[0]
    if "lr_rgb" in first_sample:
        lr_shape = tuple(first_sample["lr_rgb"].shape)
        hr_shape = tuple(first_sample["hr_rgb"].shape)
    else:
        lr_shape = tuple(first_sample["lr_y"].shape)
        hr_shape = tuple(first_sample["hr_rgb"].shape)
    lr_frame_files = count_clip_frames(args.lr_dir, dataset.clip_names)
    hr_frame_files = count_clip_frames(args.hr_dir, dataset.clip_names)

    payload = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "epoch": checkpoint_state.get("epoch"),
        "rgb_eval_yuv420": args.rgb_eval_yuv420,
        "rgb_input_mode": args.rgb_input_mode,
        "metric_domain": metric_domain,
        "num_clips": len(dataset.clip_names),
        "num_windows": len(dataset),
        "num_lr_frame_files": lr_frame_files,
        "num_hr_frame_files": hr_frame_files,
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "lr_sequence_shape": lr_shape,
        "hr_sequence_shape": hr_shape,
        **asdict(results),
    }
    if stats is not None:
        payload["metrics_std"] = stats.std
        payload["window_count"] = stats.count
        output_path = Path(args.stats_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
