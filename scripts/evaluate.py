from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from basicvsr_yuv420.checkpoints import load_model_weights
from basicvsr_yuv420.data import DEFAULT_VALIDATION_CLIPS, REDSVSRDataset
from basicvsr_yuv420.engine import evaluate
from basicvsr_yuv420.models import build_model, get_model_spec, list_model_ids
from basicvsr_yuv420.utils import resolve_device


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
    return parser


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    device = resolve_device(args.device)
    model_spec = get_model_spec(args.model)
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

    results = evaluate(model, dataloader, device)
    payload = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "epoch": checkpoint_state.get("epoch"),
        "loss": results.loss,
        "psnr": results.psnr,
        "ssim": results.ssim,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
