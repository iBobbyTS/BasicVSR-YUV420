from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import scripts.benchmark_inference_models as benchmark_script


class BenchmarkInferenceModelsTests(unittest.TestCase):
    def make_args(self) -> Namespace:
        return Namespace(
            input_dir="data/REDS_downloads/val_sharp_bicubic/val/val_sharp_bicubic/X4/000",
            frames=15,
            device="cuda:0",
            spynet_weights="models/checkpoints/spynet_sintel_final-3d2a1287.pth",
            num_channels=64,
            residual_blocks=7,
            warmup_runs=2,
            measure_runs=10,
            amp=False,
            models=["basicvsr_rgb_baseline"],
            single_model_output=None,
        )

    def test_build_child_command_includes_isolation_output_and_model(self) -> None:
        args = self.make_args()

        command = benchmark_script.build_child_command(args, "basicvsr_rgb_baseline", "temp/output.json")

        self.assertIn("--single-model-output", command)
        self.assertIn("temp/output.json", command)
        self.assertIn("--models basicvsr_rgb_baseline", command)
        self.assertIn("--frames 15", command)

    def test_benchmark_model_in_subprocess_reads_child_output(self) -> None:
        args = self.make_args()

        def fake_runner(command: str) -> int:
            marker = "--single-model-output "
            start = command.index(marker) + len(marker)
            remainder = command[start:]
            if " --" in remainder:
                output_path = remainder.split(" --", 1)[0].strip().strip('"')
            else:
                output_path = remainder.strip().strip('"')
            Path(output_path).write_text(
                json.dumps({"result": {"frames_per_second": 12.5, "peak_memory_gb": 1.25}}, indent=2),
                encoding="utf-8",
            )
            return 0

        result = benchmark_script.benchmark_model_in_subprocess(
            "basicvsr_rgb_baseline",
            args,
            command_runner=fake_runner,
        )

        self.assertEqual(result["frames_per_second"], 12.5)
        self.assertEqual(result["peak_memory_gb"], 1.25)

    def test_single_model_output_requires_exactly_one_model(self) -> None:
        args = self.make_args()
        args.models = ["basicvsr_rgb_baseline", "low_res_joint_y_head"]
        args.single_model_output = tempfile.mktemp(suffix=".json")

        with patch.object(benchmark_script, "parse_args", return_value=args):
            with self.assertRaises(ValueError):
                benchmark_script.main()


if __name__ == "__main__":
    unittest.main()
