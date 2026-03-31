from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from basicvsr_yuv420.data.colorspace import yuv420_to_rgb_bt709_full_range
from basicvsr_yuv420.engine import _compute_batch_statistics
from basicvsr_yuv420.losses import charbonnier_loss
from basicvsr_yuv420.models import build_model, get_model_spec


class RGBObjectiveModelSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.inputs = {
            "y": torch.rand(1, 3, 1, 64, 64),
            "uv": torch.rand(1, 3, 2, 32, 32),
        }

    def _build_target(self) -> dict[str, torch.Tensor]:
        target_y = torch.rand(1, 3, 1, 256, 256)
        target_uv = torch.rand(1, 3, 2, 128, 128)
        target_rgb = yuv420_to_rgb_bt709_full_range(target_y, target_uv)
        return {
            "y": target_y,
            "uv": target_uv,
            "rgb": target_rgb,
        }

    def _assert_rgb_objective_stats(self, model_id: str) -> None:
        model = build_model(model_id, num_channels=8, residual_blocks=1)
        model.eval()
        with torch.no_grad():
            prediction = model(self.inputs)
        target = self._build_target()
        stats = _compute_batch_statistics(
            prediction,
            target,
            loss_fn=charbonnier_loss,
            compute_ssim=False,
            metric_domain="rgb",
        )

        self.assertEqual(tuple(prediction["y"].shape), (1, 3, 1, 256, 256))
        self.assertEqual(tuple(prediction["uv"].shape), (1, 3, 2, 128, 128))
        self.assertIsNotNone(stats["loss"])
        self.assertIsNotNone(stats["psnr"])
        self.assertIsNotNone(stats["loss_rgb"])
        self.assertIsNotNone(stats["psnr_rgb"])
        self.assertIsNotNone(stats["loss_y"])
        self.assertIsNotNone(stats["loss_uv"])
        self.assertIsNotNone(stats["psnr_y"])
        self.assertIsNotNone(stats["psnr_uv"])
        self.assertTrue(torch.isfinite(stats["loss"]))
        self.assertTrue(torch.isfinite(stats["psnr"]))
        self.assertTrue(torch.isfinite(stats["loss_rgb"]))
        self.assertTrue(torch.isfinite(stats["psnr_rgb"]))
        self.assertAlmostEqual(float(stats["loss"]), float(stats["loss_rgb"]), places=7)
        self.assertAlmostEqual(float(stats["psnr"]), float(stats["psnr_rgb"]), places=7)

    def test_uv_conditioned_film_uses_rgb_objective_path(self) -> None:
        self.assertEqual(get_model_spec("uv_conditioned_film").metric_domain, "rgb")
        self._assert_rgb_objective_stats("uv_conditioned_film")

    def test_low_res_joint_y_head_supports_rgb_objective_path(self) -> None:
        self._assert_rgb_objective_stats("low_res_joint_y_head")

    def test_frequency_domain_fusion_supports_rgb_objective_path(self) -> None:
        self._assert_rgb_objective_stats("frequency_domain_low_frequency_fusion")


if __name__ == "__main__":
    unittest.main()
