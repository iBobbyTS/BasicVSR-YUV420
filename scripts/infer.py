from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from basicvsr_yuv420.checkpoints import load_model_weights
from basicvsr_yuv420.inference import load_frame_sequence, save_frame_sequence
from basicvsr_yuv420.models import build_generator
from basicvsr_yuv420.utils import resolve_device


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Run BasicVSR inference on a folder of LR frames.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--spynet-weights")
    parser.add_argument("--device")
    parser.add_argument("--num-channels", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=7)
    return parser


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    device = resolve_device(args.device)
    model = build_generator(
        spynet_weights=args.spynet_weights,
        num_channels=args.num_channels,
        residual_blocks=args.residual_blocks,
    ).to(device)
    checkpoint_state = load_model_weights(args.checkpoint, model, map_location="cpu")
    model.eval()

    sequence, reference_paths = load_frame_sequence(args.input_dir)
    sequence = sequence.to(device)

    with torch.no_grad():
        prediction = model(sequence)

    save_frame_sequence(prediction, args.output_dir, reference_paths=reference_paths)
    payload = {
        "checkpoint": args.checkpoint,
        "epoch": checkpoint_state.get("epoch"),
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "frames": prediction.size(1),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
