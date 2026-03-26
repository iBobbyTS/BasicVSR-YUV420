from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def load_frame_sequence(frames_dir: Union[str, Path]) -> Tuple[torch.Tensor, List[Path]]:
    input_dir = Path(frames_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    frame_paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not frame_paths:
        raise RuntimeError(f"No image frames found in {input_dir}")

    frames = []
    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGB")
        frame = np.asarray(image, dtype=np.float32) / 255.0
        frames.append(torch.from_numpy(frame).permute(2, 0, 1))

    sequence = torch.stack(frames, dim=0).unsqueeze(0)
    return sequence, frame_paths


def save_frame_sequence(
    sequence: torch.Tensor,
    output_dir: Union[str, Path],
    *,
    reference_paths: Optional[Sequence[Path]] = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if sequence.ndim != 5 or sequence.size(0) != 1:
        raise ValueError(f"Expected a tensor shaped [1, T, C, H, W], but found {sequence.shape}")

    frames = sequence.squeeze(0).detach().cpu().clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()
    for index, frame in enumerate(frames):
        file_stem = reference_paths[index].stem if reference_paths is not None else f"{index:08d}"
        image = Image.fromarray((frame * 255.0).round().astype(np.uint8))
        image.save(output_path / f"{file_stem}.png")
