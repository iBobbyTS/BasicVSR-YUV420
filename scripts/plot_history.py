from __future__ import annotations

import json
import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Plot training loss and PSNR curves from a history.json file.")
    parser.add_argument("--history", required=True, help="Path to history.json produced by train.py")
    parser.add_argument("--output", help="Path to the output image. Defaults to <history_dir>/training_curves.png")
    parser.add_argument("--title", default="Training Curves")
    parser.add_argument("--dpi", type=int, default=150)
    return parser


def _load_history(path: Path) -> Dict[str, List[Dict[str, float]]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "train": payload.get("train", []),
        "val": payload.get("val", []),
    }


def _series(records: Sequence[Dict[str, float]], key: str) -> Tuple[List[int], List[float]]:
    epochs: List[int] = []
    values: List[float] = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        numeric_value = float(value)
        if math.isnan(numeric_value):
            continue
        epochs.append(int(record["epoch"]))
        values.append(numeric_value)
    return epochs, values


def _plot_metric(ax, train_records, val_records, key: str, label: str) -> None:
    train_epochs, train_values = _series(train_records, key)
    val_epochs, val_values = _series(val_records, key)

    if train_epochs:
        ax.plot(train_epochs, train_values, marker="o", linewidth=1.8, markersize=3, label=f"Train {label}")
    if val_epochs:
        ax.plot(val_epochs, val_values, marker="s", linewidth=1.8, markersize=3, label=f"Val {label}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    if train_epochs or val_epochs:
        ax.legend()


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    history_path = Path(args.history)
    output_path = Path(args.output) if args.output else history_path.parent / "training_curves.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history = _load_history(history_path)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    figure.suptitle(args.title)

    _plot_metric(axes[0], history["train"], history["val"], key="loss", label="Loss")
    _plot_metric(axes[1], history["train"], history["val"], key="psnr", label="PSNR")

    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    print(
        json.dumps(
            {
                "history": str(history_path),
                "output": str(output_path),
                "train_epochs": len(history["train"]),
                "val_epochs": len(history["val"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
