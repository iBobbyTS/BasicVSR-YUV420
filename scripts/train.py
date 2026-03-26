from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from basicvsr_yuv420.checkpoints import build_checkpoint, load_checkpoint, save_checkpoint
from basicvsr_yuv420.data import DEFAULT_VALIDATION_CLIPS, REDSVSRDataset
from basicvsr_yuv420.engine import evaluate, train_one_epoch
from basicvsr_yuv420.models import build_generator
from basicvsr_yuv420.utils import ensure_dir, read_json, resolve_device, set_seed, write_json


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Train BasicVSR from the converted Python project.")
    parser.add_argument("--train-lr-dir", required=True)
    parser.add_argument("--train-hr-dir", required=True)
    parser.add_argument("--val-lr-dir")
    parser.add_argument("--val-hr-dir")
    parser.add_argument("--output-dir", default="outputs/train_run")
    parser.add_argument("--spynet-weights")
    parser.add_argument("--resume")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-7)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=15)
    parser.add_argument("--sequence-stride", type=int, default=15)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--num-channels", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=7)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        help="Enable mixed precision training. Enabled by default to match the notebook.",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable mixed precision training.",
    )
    parser.add_argument(
        "--use-default-reds-split",
        dest="use_default_reds_split",
        action="store_true",
        help="Exclude REDS validation clips from training and use them for validation. Enabled by default.",
    )
    parser.add_argument(
        "--no-default-reds-split",
        dest="use_default_reds_split",
        action="store_false",
        help="Do not apply the default REDS validation clip split.",
    )
    parser.set_defaults(amp=True, use_default_reds_split=True)
    return parser


def build_dataset(
    *,
    lr_dir: str,
    hr_dir: str,
    train: bool,
    args,
) -> REDSVSRDataset:
    include_clips = None
    exclude_clips = None
    if args.use_default_reds_split:
        if train:
            exclude_clips = DEFAULT_VALIDATION_CLIPS
        else:
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
    )


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    history_path = output_dir / "history.json"
    config_path = output_dir / "config.json"
    write_json(vars(args), config_path)

    train_dataset = build_dataset(lr_dir=args.train_lr_dir, hr_dir=args.train_hr_dir, train=True, args=args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = None
    if args.val_lr_dir and args.val_hr_dir:
        val_dataset = REDSVSRDataset(
            args.val_lr_dir,
            args.val_hr_dir,
            sequence_length=args.sequence_length,
            sequence_stride=args.sequence_stride,
            patch_size=None,
            scale=args.scale,
            train=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    elif args.use_default_reds_split:
        val_dataset = build_dataset(lr_dir=args.train_lr_dir, hr_dir=args.train_hr_dir, train=False, args=args)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    model = build_generator(
        spynet_weights=args.spynet_weights,
        num_channels=args.num_channels,
        residual_blocks=args.residual_blocks,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs * max(len(train_loader), 1), 1),
        eta_min=args.min_learning_rate,
    )
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    start_epoch = 0
    best_psnr = None
    history = read_json(history_path, default={"train": [], "val": []})

    if args.resume:
        state = load_checkpoint(
            args.resume,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location="cpu",
        )
        start_epoch = int(state.get("epoch", -1)) + 1
        best_psnr = state.get("best_psnr")

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scheduler=scheduler,
            scaler=scaler,
            amp_enabled=args.amp and device.type == "cuda",
            progress_desc=f"Train {epoch + 1}/{args.epochs}",
        )

        val_stats = None
        if val_loader is not None:
            val_stats = evaluate(
                model,
                val_loader,
                device,
                progress_desc=f"Eval {epoch + 1}/{args.epochs}",
            )

        train_record = {"epoch": epoch + 1, **asdict(train_stats)}
        history.setdefault("train", []).append(train_record)
        if val_stats is not None:
            history.setdefault("val", []).append({"epoch": epoch + 1, **asdict(val_stats)})
        write_json(history, history_path)

        monitor_psnr = val_stats.psnr if val_stats is not None else train_stats.psnr
        if best_psnr is None or monitor_psnr > best_psnr:
            best_psnr = monitor_psnr

        metrics = {"train": train_record}
        if val_stats is not None:
            metrics["val"] = {"epoch": epoch + 1, **asdict(val_stats)}

        checkpoint = build_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_psnr=best_psnr,
            metrics=metrics,
            model_config={
                "num_channels": args.num_channels,
                "residual_blocks": args.residual_blocks,
                "scale": args.scale,
            },
        )

        latest_path = output_dir / "latest.pt"
        save_checkpoint(latest_path, checkpoint)

        if monitor_psnr == best_psnr:
            save_checkpoint(output_dir / "best.pt", checkpoint)

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(output_dir / f"epoch_{epoch + 1:04d}.pt", checkpoint)

        summary = {
            "epoch": epoch + 1,
            "train": train_record,
            "val": None if val_stats is None else {"epoch": epoch + 1, **asdict(val_stats)},
            "best_psnr": best_psnr,
            "latest_checkpoint": str(latest_path),
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
