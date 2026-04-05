| Model                                   |   FPS |   Sequence Latency | 10Hz Peak GPU Memory | Peak GPU Memory Delta | `torch.cuda.max_memory_allocated` |
|-----------------------------------------|------:|-------------------:|---------------------:|----------------------:|----------------------------------:|
| `basicvsr_rgb_baseline`                 | 22.16 | 4.514s / 100 frames |              8618 MB |               6558 MB |                         4.676 GiB |
| `low_res_joint_y_head`                  | 31.96 | 3.129s / 100 frames |              8718 MB |               6658 MB |                         4.255 GiB |
| `frequency_domain_low_frequency_fusion` | 28.34 | 3.528s / 100 frames |              8738 MB |               6678 MB |                         4.257 GiB |
Notes:

- Input clip: the first 100 frames of `data/REDS_downloads/val_sharp_bicubic/val/val_sharp_bicubic/X4/000`.
- Input size: LR `100 x 320 x 180`, corresponding HR output size `100 x 1280 x 720`.
- FPS, sequence latency, and `torch.cuda.max_memory_allocated` command:
  `python scripts/benchmark_inference_models.py --input-dir data/REDS_downloads/val_sharp_bicubic/val/val_sharp_bicubic/X4/000 --frames 100 --device cuda:0 --spynet-weights models/checkpoints/spynet_sintel_final-3d2a1287.pth --models basicvsr_rgb_baseline low_res_joint_y_head frequency_domain_low_frequency_fusion`
- The 10Hz GPU memory columns came from separate single-model runs of the same 100-frame command while sampling GPU memory for 20 seconds at 100 ms intervals with:
  `powershell -ExecutionPolicy Bypass -File .skill/gpu-utilization-monitor/scripts/sample_gpu_utilization.ps1 -DurationSeconds 20 -IntervalMilliseconds 100 -CsvPath <output.csv>`
- `Peak GPU Memory Delta` uses an idle baseline of `2060 MB` measured immediately before the 10Hz sampling batch.
