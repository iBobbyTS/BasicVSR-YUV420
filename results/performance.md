| Model | FPS | Sequence Latency | 10Hz Peak GPU Memory | Peak GPU Memory Delta | `torch.cuda.max_memory_allocated` |
|-------|-----:|-----------------:|---------------------:|----------------------:|----------------------------------:|
| `basicvsr_rgb_baseline` | 21.80 | 2.294s / 50 frames | 4506 MB | 3644 MB | 2.325 GiB |
| `low_res_joint_y_head` | 32.30 | 1.548s / 50 frames | 4494 MB | 3632 MB | 2.118 GiB |
| `frequency_domain_low_frequency_fusion` | 28.88 | 1.732s / 50 frames | 4506 MB | 3644 MB | 2.119 GiB |
| `uv_conditioned_film` | 22.92 | 2.181s / 50 frames | 5253 MB | 4391 MB | 2.117 GiB |

Notes:

- All measurements used the first 50 frames of `data/REDS_downloads/val_sharp_bicubic/val/val_sharp_bicubic/X4/000`.
- FPS and sequence latency were measured in a pure forward benchmark without `nvidia-smi` sampling.
- The 10Hz GPU memory columns came from a separate 20-second `nvidia-smi` run with 100 ms sampling intervals.
- The `torch.cuda.max_memory_allocated` column came from the pure forward benchmark and reflects PyTorch allocator peak memory during measured inference.
