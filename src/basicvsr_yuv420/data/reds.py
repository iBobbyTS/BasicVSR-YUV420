from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_VALIDATION_CLIPS = ("000", "011", "015", "020")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def _list_frame_paths(clip_dir: Path) -> List[Path]:
    return sorted(path for path in clip_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


class REDSVSRDataset(Dataset):
    def __init__(
        self,
        lr_dir: Union[str, Path],
        hr_dir: Union[str, Path],
        *,
        sequence_length: int = 15,
        sequence_stride: Optional[int] = None,
        patch_size: Optional[int] = 64,
        scale: int = 4,
        train: bool = True,
        include_clips: Optional[Sequence[str]] = None,
        exclude_clips: Optional[Sequence[str]] = None,
    ) -> None:
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride or sequence_length
        self.patch_size = patch_size
        self.scale = scale
        self.train = train
        self.include_clips = set(include_clips or [])
        self.exclude_clips = set(exclude_clips or [])

        if not self.lr_dir.exists():
            raise FileNotFoundError(f"LR directory not found: {self.lr_dir}")
        if not self.hr_dir.exists():
            raise FileNotFoundError(f"HR directory not found: {self.hr_dir}")

        lr_clip_dirs = {path.name: path for path in self.lr_dir.iterdir() if path.is_dir()}
        hr_clip_dirs = {path.name: path for path in self.hr_dir.iterdir() if path.is_dir()}
        clip_names = sorted(set(lr_clip_dirs).intersection(hr_clip_dirs))

        if self.include_clips:
            clip_names = [name for name in clip_names if name in self.include_clips]
        if self.exclude_clips:
            clip_names = [name for name in clip_names if name not in self.exclude_clips]
        if not clip_names:
            raise RuntimeError("No clip folders matched the provided dataset configuration.")

        self.clip_names = clip_names
        self.lr_clip_dirs = lr_clip_dirs
        self.hr_clip_dirs = hr_clip_dirs
        self.samples = self._build_samples()

        if not self.samples:
            raise RuntimeError("No training or evaluation samples could be built from the dataset.")

    def _build_samples(self) -> List[Tuple[str, int]]:
        samples = []  # type: List[Tuple[str, int]]
        for clip_name in self.clip_names:
            lr_frames = _list_frame_paths(self.lr_clip_dirs[clip_name])
            hr_frames = _list_frame_paths(self.hr_clip_dirs[clip_name])

            if len(lr_frames) != len(hr_frames):
                raise RuntimeError(f"Mismatched frame counts for clip {clip_name}.")
            if len(lr_frames) < self.sequence_length:
                continue

            max_start = len(lr_frames) - self.sequence_length + 1
            for start in range(0, max_start, self.sequence_stride):
                samples.append((clip_name, start))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sequence(self, frame_paths: Sequence[Path]) -> np.ndarray:
        frames = []
        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            frames.append(np.asarray(image, dtype=np.uint8))
        return np.stack(frames, axis=0)

    def _random_crop(self, lr: np.ndarray, hr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.patch_size is None:
            return lr, hr

        lr_height, lr_width = lr.shape[1:3]
        if self.patch_size > lr_height or self.patch_size > lr_width:
            raise ValueError(
                f"Patch size {self.patch_size} is larger than LR frame size {(lr_height, lr_width)}."
            )

        hr_patch_size = self.patch_size * self.scale
        hr_height, hr_width = hr.shape[1:3]
        expected_hr_shape = (lr_height * self.scale, lr_width * self.scale)
        if (hr_height, hr_width) != expected_hr_shape:
            raise ValueError(
                f"Expected HR frame size {expected_hr_shape}, but found {(hr_height, hr_width)}."
            )

        x = random.randint(0, lr_width - self.patch_size)
        y = random.randint(0, lr_height - self.patch_size)

        lr_crop = lr[:, y : y + self.patch_size, x : x + self.patch_size, :]
        hr_crop = hr[
            :,
            y * self.scale : y * self.scale + hr_patch_size,
            x * self.scale : x * self.scale + hr_patch_size,
            :,
        ]
        return lr_crop, hr_crop

    def _augment(self, lr: torch.Tensor, hr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[2])
            hr = torch.flip(hr, dims=[2])
        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[1])
            hr = torch.flip(hr, dims=[1])
        return lr, hr

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clip_name, start = self.samples[index]
        end = start + self.sequence_length

        lr_frames = _list_frame_paths(self.lr_clip_dirs[clip_name])[start:end]
        hr_frames = _list_frame_paths(self.hr_clip_dirs[clip_name])[start:end]

        lr = self._load_sequence(lr_frames)
        hr = self._load_sequence(hr_frames)

        if self.train:
            lr, hr = self._random_crop(lr, hr)

        lr_tensor = torch.from_numpy(lr).float() / 255.0
        hr_tensor = torch.from_numpy(hr).float() / 255.0

        if self.train:
            lr_tensor, hr_tensor = self._augment(lr_tensor, hr_tensor)

        return lr_tensor, hr_tensor
