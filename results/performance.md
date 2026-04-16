| Model                                   |   FPS |   Sequence Latency | 10Hz Peak GPU Memory | Peak GPU Memory Delta | `torch.cuda.max_memory_allocated` |
|-----------------------------------------|------:|-------------------:|---------------------:|----------------------:|----------------------------------:|
| `basicvsr_rgb_baseline`                 | 21.54 | 0.697s / 15 frames |              4011 MB |               1791 MB |                         0.919 GiB |
| `low_res_joint_y_head`                  | 32.23 | 0.465s / 15 frames |              3549 MB |               1329 MB |                         0.770 GiB |
| `frequency_domain_low_frequency_fusion` | 28.80 | 0.521s / 15 frames |              3549 MB |               1329 MB |                         0.788 GiB |
Notes:

- Input clip: the first 15 frames of `data/REDS_downloads/val_sharp_bicubic/val/val_sharp_bicubic/X4/000`.
- Input size: LR `15 x 320 x 180`, corresponding HR output size `15 x 1280 x 720`.
- FPS, sequence latency, and `torch.cuda.max_memory_allocated` command:
  `python scripts/benchmark_inference_models.py --input-dir data/REDS_downloads/val_sharp_bicubic/val/val_sharp_bicubic/X4/000 --frames 15 --measure-runs 10 --device cuda:0 --spynet-weights models/checkpoints/spynet_sintel_final-3d2a1287.pth --models basicvsr_rgb_baseline low_res_joint_y_head frequency_domain_low_frequency_fusion`
- FPS is the mean over `10` measured runs, each run using the same `15`-frame input sequence after `2` warmup runs.
- The launcher script benchmarks each model through `os.system(...)` in a fresh Python subprocess, so the reported `torch.cuda.max_memory_allocated` value is isolated per model instead of reusing the previous model's CUDA cache.
- The 10Hz GPU memory columns came from separate single-model runs of the same 15-frame command while sampling GPU memory for 20 seconds at 100 ms intervals with:
  `powershell -ExecutionPolicy Bypass -File .skill/gpu-utilization-monitor/scripts/sample_gpu_utilization.ps1 -DurationSeconds 20 -IntervalMilliseconds 100 -CsvPath <output.csv>`
- The sampled inference process also uses `15` frames and `10` measured runs per model.
- `Peak GPU Memory Delta` uses an idle baseline of `2220 MB` measured immediately before the 10Hz sampling batch.
