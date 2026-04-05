from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from basicvsr_yuv420.data.colorspace import rgb_through_yuv420_bt709_full_range
from basicvsr_yuv420.data.reds import REDSVSRDataset


class RGBInputModeDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.lr_dir = root / "lr"
        self.hr_dir = root / "hr"
        (self.lr_dir / "000").mkdir(parents=True)
        (self.hr_dir / "000").mkdir(parents=True)

        self.lr_frame = np.array(
            [
                [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
                [[0, 255, 255], [255, 0, 255], [64, 64, 64], [192, 192, 192]],
                [[32, 128, 224], [224, 128, 32], [16, 240, 96], [240, 16, 96]],
                [[12, 48, 96], [96, 48, 12], [180, 40, 220], [40, 220, 180]],
            ],
            dtype=np.uint8,
        )
        self.hr_frame = np.repeat(np.repeat(self.lr_frame, 4, axis=0), 4, axis=1)

        Image.fromarray(self.lr_frame).save(self.lr_dir / "000" / "00000000.png")
        Image.fromarray(self.hr_frame).save(self.hr_dir / "000" / "00000000.png")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _to_tensor(frame: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

    def test_rgb_mode_keeps_original_lr_and_hr(self) -> None:
        dataset = REDSVSRDataset(
            self.lr_dir,
            self.hr_dir,
            sequence_length=1,
            patch_size=None,
            train=False,
            color_mode="rgb",
            rgb_input_mode="rgb",
        )

        sample = dataset[0]
        expected_lr = self._to_tensor(self.lr_frame).unsqueeze(0)
        expected_hr = self._to_tensor(self.hr_frame).unsqueeze(0)

        self.assertTrue(torch.allclose(sample["lr_rgb"], expected_lr, atol=1e-6))
        self.assertTrue(torch.allclose(sample["hr_rgb"], expected_hr, atol=1e-6))

    def test_rgb_yuv420_rgb_mode_round_trips_lr_only(self) -> None:
        dataset = REDSVSRDataset(
            self.lr_dir,
            self.hr_dir,
            sequence_length=1,
            patch_size=None,
            train=False,
            color_mode="rgb",
            rgb_input_mode="rgb_yuv420_rgb",
        )

        sample = dataset[0]
        original_lr = self._to_tensor(self.lr_frame).unsqueeze(0)
        expected_lr = rgb_through_yuv420_bt709_full_range(original_lr)
        expected_hr = self._to_tensor(self.hr_frame).unsqueeze(0)

        self.assertTrue(torch.allclose(sample["lr_rgb"], expected_lr, atol=1e-6))
        self.assertFalse(torch.allclose(sample["lr_rgb"], original_lr, atol=1e-5))
        self.assertTrue(torch.allclose(sample["hr_rgb"], expected_hr, atol=1e-6))

    def test_invalid_rgb_input_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            REDSVSRDataset(
                self.lr_dir,
                self.hr_dir,
                sequence_length=1,
                patch_size=None,
                train=False,
                color_mode="rgb",
                rgb_input_mode="invalid",
            )

    def test_non_rgb_dataset_rejects_rgb_input_mode(self) -> None:
        with self.assertRaises(ValueError):
            REDSVSRDataset(
                self.lr_dir,
                self.hr_dir,
                sequence_length=1,
                patch_size=None,
                train=False,
                color_mode="yuv420",
                rgb_input_mode="rgb_yuv420_rgb",
            )


if __name__ == "__main__":
    unittest.main()
