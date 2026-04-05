from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from basicvsr_yuv420.engine import summarize_metric_history


class EvaluateStatisticsTests(unittest.TestCase):
    def test_summarize_metric_history_returns_mean_and_population_std(self) -> None:
        history = {
            "loss": [1.0, 3.0],
            "psnr": [20.0, 24.0],
            "y_psnr": [21.0, 25.0],
            "uv420_psnr": [40.0, 44.0],
        }

        stats = summarize_metric_history(history, lr=0.0)

        self.assertEqual(stats.count, 2)
        self.assertAlmostEqual(stats.result.loss, 2.0, places=7)
        self.assertAlmostEqual(stats.result.psnr, 22.0, places=7)
        self.assertAlmostEqual(stats.result.y_psnr, 23.0, places=7)
        self.assertAlmostEqual(stats.result.uv420_psnr, 42.0, places=7)
        self.assertAlmostEqual(stats.std["loss"], 1.0, places=7)
        self.assertAlmostEqual(stats.std["psnr"], 2.0, places=7)
        self.assertAlmostEqual(stats.std["y_psnr"], 2.0, places=7)
        self.assertAlmostEqual(stats.std["uv420_psnr"], 2.0, places=7)

    def test_single_window_statistics_report_zero_std(self) -> None:
        stats = summarize_metric_history({"loss": [2.5], "psnr": [30.0]}, lr=0.0)

        self.assertEqual(stats.count, 1)
        self.assertAlmostEqual(stats.result.loss, 2.5, places=7)
        self.assertAlmostEqual(stats.result.psnr, 30.0, places=7)
        self.assertAlmostEqual(stats.std["loss"], 0.0, places=7)
        self.assertAlmostEqual(stats.std["psnr"], 0.0, places=7)
        self.assertTrue(math.isfinite(stats.std["loss"]))
        self.assertTrue(math.isfinite(stats.std["psnr"]))


if __name__ == "__main__":
    unittest.main()
