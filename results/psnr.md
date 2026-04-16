| Model                                    | RGB-PSNR (mean ± std) | Y-PSNR (mean ± std) | UV420-PSNR (mean ± std) |
|------------------------------------------|----------------------:|--------------------:|------------------------:|
| `baseline_rgb (rgb input)`               |    `28.0913 ± 2.7028` |  `28.0972 ± 2.7178` |      `53.6855 ± 2.1163` |
| `baseline_rgb (yuv420 input)`            |    `27.7906 ± 2.6547` |  `28.0385 ± 2.7096` |      `44.2885 ± 3.2082` |
| `baseline_rgb (trained on yuv420 input)` |    `27.7225 ± 2.5896` |  `27.8923 ± 2.6477` |      `45.3433 ± 2.5018` |
| `low_res_joint_y_head (yuv420 loss)`     |    `27.7765 ± 2.6381` |  `27.9041 ± 2.6662` |      `46.7719 ± 3.3062` |
| `low_res_joint_y_head`                   |    `27.8092 ± 2.6414` |  `27.9103 ± 2.6674` |      `47.2815 ± 3.1441` |
| `frequency_domain_lf_fusion`             |    `27.9083 ± 2.6782` |  `28.0224 ± 2.7055` |      `46.8942 ± 3.1721` |

Notes:

- Evaluation dataset: `data/REDS_downloads/val_sharp_bicubic/val/val_sharp_bicubic/X4` and `data/REDS_downloads/val_sharp/val/val_sharp`
- Files used per run: `30` clips, `3000` LR frame files, `3000` HR frame files
- Evaluation windows per run: `180`
- LR frame size: `320 x 180`
- HR frame size: `1280 x 720`
- Evaluation window configuration: `sequence_length=15`, `sequence_stride=15`, `batch_size=1`
- RGB-input models use LR tensors of shape `15 x 3 x 180 x 320`
- YUV420-input models use luma tensors of shape `15 x 1 x 180 x 320` and chroma tensors of shape `15 x 2 x 90 x 160`
- `mean` and `std` are computed over the `180` evaluation windows returned by `scripts/evaluate.py`

Commands:

```bash
python scripts/evaluate.py --model basicvsr_rgb_baseline --checkpoint outputs/baseline_rgb_v3_noamp_lr1e4_bs2_acc2/best.pt --metric-domain rgb --rgb-input-mode rgb --stats-output outputs/eval_stats_20260405/baseline_rgb_input.json --lr-dir data/REDS_downloads/val_sharp_bicubic/val/val_sharp_bicubic/X4 --hr-dir data/REDS_downloads/val_sharp/val/val_sharp --spynet-weights models/checkpoints/spynet_sintel_final-3d2a1287.pth --device cuda:0 --batch-size 1 --sequence-length 15 --sequence-stride 15
python scripts/evaluate.py --model basicvsr_rgb_baseline --checkpoint outputs/baseline_rgb_v3_noamp_lr1e4_bs2_acc2/best.pt --metric-domain rgb --rgb-input-mode rgb_yuv420_rgb --stats-output outputs/eval_stats_20260405/baseline_rgb_yuv420_input.json --lr-dir data/REDS_downloads/val_sharp_bicubic/val/val_sharp_bicubic/X4 --hr-dir data/REDS_downloads/val_sharp/val/val_sharp --spynet-weights models/checkpoints/spynet_sintel_final-3d2a1287.pth --device cuda:0 --batch-size 1 --sequence-length 15 --sequence-stride 15
python scripts/evaluate.py --model basicvsr_rgb_baseline --checkpoint outputs/baseline_rgb_v7_yuv420_input_tuned_bs4_vb1_nw1/best.pt --metric-domain rgb --rgb-input-mode rgb_yuv420_rgb --stats-output outputs/eval_stats_20260405/baseline_rgb_trained_on_yuv420_input.json --lr-dir data/REDS_downloads/val_sharp_bicubic/val/val_sharp_bicubic/X4 --hr-dir data/REDS_downloads/val_sharp/val/val_sharp --spynet-weights models/checkpoints/spynet_sintel_final-3d2a1287.pth --device cuda:0 --batch-size 1 --sequence-length 15 --sequence-stride 15
python scripts/evaluate.py --model low_res_joint_y_head --checkpoint outputs/low_res_joint_y_head_v2_bs8_vb1_nw4/best.pt --metric-domain rgb --rgb-input-mode rgb --stats-output outputs/eval_stats_20260405/low_res_joint_y_head_yuv420_loss.json --lr-dir data/REDS_downloads/val_sharp_bicubic/val/val_sharp_bicubic/X4 --hr-dir data/REDS_downloads/val_sharp/val/val_sharp --spynet-weights models/checkpoints/spynet_sintel_final-3d2a1287.pth --device cuda:0 --batch-size 1 --sequence-length 15 --sequence-stride 15
python scripts/evaluate.py --model low_res_joint_y_head --checkpoint outputs/low_res_joint_y_head_v3_rgb_objective_bs8_vb1_nw4/best.pt --metric-domain rgb --rgb-input-mode rgb --stats-output outputs/eval_stats_20260405/low_res_joint_y_head_rgb_loss.json --lr-dir data/REDS_downloads/val_sharp_bicubic/val/val_sharp_bicubic/X4 --hr-dir data/REDS_downloads/val_sharp/val/val_sharp --spynet-weights models/checkpoints/spynet_sintel_final-3d2a1287.pth --device cuda:0 --batch-size 1 --sequence-length 15 --sequence-stride 15
python scripts/evaluate.py --model frequency_domain_low_frequency_fusion --checkpoint outputs/frequency_domain_low_frequency_fusion_v2_rgb_objective_bs4_vb1_nw6/best.pt --metric-domain rgb --rgb-input-mode rgb --stats-output outputs/eval_stats_20260405/frequency_domain_lf_fusion_rgb_loss.json --lr-dir data/REDS_downloads/val_sharp_bicubic/val/val_sharp_bicubic/X4 --hr-dir data/REDS_downloads/val_sharp/val/val_sharp --spynet-weights models/checkpoints/spynet_sintel_final-3d2a1287.pth --device cuda:0 --batch-size 1 --sequence-length 15 --sequence-stride 15
```
