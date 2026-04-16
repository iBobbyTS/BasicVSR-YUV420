from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .colorspace import rgb_through_yuv420_bt709_full_range, rgb_to_yuv420_bt709_full_range

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
        color_mode: str = "rgb",
        rgb_input_mode: str = "rgb",
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
        self.color_mode = color_mode
        self.rgb_input_mode = rgb_input_mode

        if self.color_mode not in {"rgb", "yuv420"}:
            raise ValueError(f"Unsupported color mode: {self.color_mode}")
        if self.rgb_input_mode not in {"rgb", "rgb_yuv420_rgb"}:
            raise ValueError(f"Unsupported RGB input mode: {self.rgb_input_mode}")
        if self.color_mode != "rgb" and self.rgb_input_mode != "rgb":
            raise ValueError("rgb_input_mode is only supported when color_mode='rgb'.")
        if self.color_mode == "yuv420" and self.patch_size is not None and self.patch_size % 2 != 0:
            raise ValueError("YUV420 training requires an even patch size.")

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
        samples: List[Tuple[str, int]] = []
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

        max_x = lr_width - self.patch_size
        max_y = lr_height - self.patch_size
        if self.color_mode == "yuv420":
            x = random.randrange(0, max_x + 1, 2)
            y = random.randrange(0, max_y + 1, 2)
        else:
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

        lr_crop = lr[:, y : y + self.patch_size, x : x + self.patch_size, :]
        hr_crop = hr[
            :,
            y * self.scale : y * self.scale + hr_patch_size,
            x * self.scale : x * self.scale + hr_patch_size,
            :,
        ]
        return lr_crop, hr_crop

    def _augment(self, lr: np.ndarray, hr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            lr = np.flip(lr, axis=2).copy()
            hr = np.flip(hr, axis=2).copy()
        if random.random() < 0.5:
            lr = np.flip(lr, axis=1).copy()
            hr = np.flip(hr, axis=1).copy()
        return lr, hr

    @staticmethod
    def _to_chw_tensor(sequence: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(sequence).permute(0, 3, 1, 2).float() / 255.0

    def _build_rgb_sample(self, lr_tensor: torch.Tensor, hr_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.rgb_input_mode == "rgb_yuv420_rgb":
            lr_tensor = rgb_through_yuv420_bt709_full_range(lr_tensor)
        return {
            "lr_rgb": lr_tensor,
            "hr_rgb": hr_tensor,
        }

    def _build_yuv420_sample(self, lr_tensor: torch.Tensor, hr_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        lr_y, lr_uv = rgb_to_yuv420_bt709_full_range(lr_tensor)
        hr_y, hr_uv = rgb_to_yuv420_bt709_full_range(hr_tensor)
        return {
            "lr_y": lr_y,
            "lr_uv": lr_uv,
            "hr_y": hr_y,
            "hr_uv": hr_uv,
            "hr_rgb": hr_tensor,
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        clip_name, start = self.samples[index]
        end = start + self.sequence_length

        lr_frames = _list_frame_paths(self.lr_clip_dirs[clip_name])[start:end]
        hr_frames = _list_frame_paths(self.hr_clip_dirs[clip_name])[start:end]

        lr = self._load_sequence(lr_frames)
        hr = self._load_sequence(hr_frames)

        if self.train:
            lr, hr = self._random_crop(lr, hr)
            lr, hr = self._augment(lr, hr)

        lr_tensor = self._to_chw_tensor(lr)
        hr_tensor = self._to_chw_tensor(hr)

        if self.color_mode == "rgb":
            return self._build_rgb_sample(lr_tensor, hr_tensor)
        return self._build_yuv420_sample(lr_tensor, hr_tensor)
