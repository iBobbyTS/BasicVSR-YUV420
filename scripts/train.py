from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

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

DATASET_PATH_ARGUMENTS = {
    "train_lr_dir",
    "train_hr_dir",
    "val_lr_dir",
    "val_hr_dir",
}


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
        "--auto-resume",
        dest="auto_resume",
        action="store_true",
        help="Automatically resume from output-dir/latest.pt when available. Enabled by default.",
    )
    parser.add_argument(
        "--no-auto-resume",
        dest="auto_resume",
        action="store_false",
        help="Disable automatic checkpoint resume detection.",
    )
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
    parser.set_defaults(amp=True, use_default_reds_split=True, auto_resume=True)
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


def resolve_resume_path(args, output_dir: Path) -> Optional[Path]:
    if args.resume:
        return Path(args.resume)
    latest_path = output_dir / "latest.pt"
    if args.auto_resume and latest_path.exists():
        return latest_path
    return None


def trim_history(history: Dict[str, Any], completed_epochs: int) -> Dict[str, Any]:
    trimmed = {
        "train": [],
        "val": [],
    }
    for key in ("train", "val"):
        records = history.get(key, [])
        trimmed[key] = [record for record in records if int(record.get("epoch", 0)) <= completed_epochs]
    return trimmed


def write_state(
    path: Path,
    *,
    status: str,
    requested_epochs: int,
    last_completed_epoch: int,
    next_epoch: Optional[int],
    latest_checkpoint: Optional[Path],
    best_psnr: Optional[float],
    resumed_from: Optional[Path],
) -> None:
    payload = {
        "status": status,
        "requested_epochs": requested_epochs,
        "last_completed_epoch": last_completed_epoch,
        "next_epoch": next_epoch,
        "latest_checkpoint": None if latest_checkpoint is None else str(latest_checkpoint),
        "best_psnr": best_psnr,
        "resumed_from": None if resumed_from is None else str(resumed_from),
    }
    write_json(payload, path)


def build_recorded_config(args) -> Dict[str, Any]:
    return {
        key: value
        for key, value in sorted(vars(args).items())
        if key not in DATASET_PATH_ARGUMENTS
    }

def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    history_path = output_dir / "history.json"
    config_path = output_dir / "config.json"
    state_path = output_dir / "state.json"
    current_config = build_recorded_config(args)
    resume_path = resolve_resume_path(args, output_dir)

    if config_path.exists():
        recorded_config = read_json(config_path, default={})
        if recorded_config != current_config:
            payload = {
                "status": "config_mismatch",
                "recorded_config": recorded_config,
                "current_config": current_config,
            }
            print(json.dumps(payload, indent=2), flush=True)
            raise SystemExit(1)

    write_json(current_config, config_path)

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
    history = {"train": [], "val": []}
    resumed_state = None

    if resume_path is not None:
        resumed_state = load_checkpoint(
            resume_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location="cpu",
        )
        start_epoch = int(resumed_state.get("epoch", -1)) + 1
        best_psnr = resumed_state.get("best_psnr")
        history = trim_history(
            read_json(history_path, default={"train": [], "val": []}),
            completed_epochs=start_epoch,
        )

    latest_path = output_dir / "latest.pt"

    if start_epoch >= args.epochs:
        write_state(
            state_path,
            status="completed",
            requested_epochs=args.epochs,
            last_completed_epoch=start_epoch,
            next_epoch=None,
            latest_checkpoint=resume_path if resume_path is not None else latest_path,
            best_psnr=best_psnr,
            resumed_from=resume_path,
        )
        summary = {
            "status": "completed",
            "requested_epochs": args.epochs,
            "completed_epochs": start_epoch,
            "best_psnr": best_psnr,
            "latest_checkpoint": None if resume_path is None else str(resume_path),
            "last_metrics": None if resumed_state is None else resumed_state.get("metrics"),
        }
        print(json.dumps(summary, indent=2), flush=True)
        return

    write_state(
        state_path,
        status="running",
        requested_epochs=args.epochs,
        last_completed_epoch=start_epoch,
        next_epoch=start_epoch + 1,
        latest_checkpoint=resume_path if resume_path is not None else None,
        best_psnr=best_psnr,
        resumed_from=resume_path,
    )

    for epoch in range(start_epoch, args.epochs):
        write_state(
            state_path,
            status="running",
            requested_epochs=args.epochs,
            last_completed_epoch=epoch,
            next_epoch=epoch + 1,
            latest_checkpoint=latest_path if latest_path.exists() else resume_path,
            best_psnr=best_psnr,
            resumed_from=resume_path,
        )
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
        monitor_psnr = val_stats.psnr if val_stats is not None else train_stats.psnr
        is_best = best_psnr is None or monitor_psnr > best_psnr
        if is_best:
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

        save_checkpoint(latest_path, checkpoint)

        if is_best:
            save_checkpoint(output_dir / "best.pt", checkpoint)

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(output_dir / f"epoch_{epoch + 1:04d}.pt", checkpoint)

        history.setdefault("train", []).append(train_record)
        if val_stats is not None:
            history.setdefault("val", []).append({"epoch": epoch + 1, **asdict(val_stats)})
        write_json(history, history_path)

        is_completed = epoch + 1 >= args.epochs
        write_state(
            state_path,
            status="completed" if is_completed else "running",
            requested_epochs=args.epochs,
            last_completed_epoch=epoch + 1,
            next_epoch=None if is_completed else epoch + 2,
            latest_checkpoint=latest_path,
            best_psnr=best_psnr,
            resumed_from=resume_path,
        )

        summary = {
            "epoch": epoch + 1,
            "train": train_record,
            "val": None if val_stats is None else {"epoch": epoch + 1, **asdict(val_stats)},
            "best_psnr": best_psnr,
            "latest_checkpoint": str(latest_path),
        }
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
