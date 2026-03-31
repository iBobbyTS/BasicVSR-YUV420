| Model                                    | RGB-PSNR | Y-PSNR  | UV420-PSNR |
|------------------------------------------|----------|---------|------------|
| baseline_rgb                             | 28.0913  | 28.0972 | 53.6855    |
| uv_conditioned_film (rgb loss)           | 27.8463  | 27.9722 | 46.6212    |
| low_res_joint_y_head (yuv420 loss)       | 27.7765  | 27.9041 | 46.7719    |
| low_res_joint_y_head (rgb loss)          | 27.8092  | 27.9103 | 47.2815    |
| frequency_domain_lf_fusion (rgb loss)    | 27.9083  | 28.0224 | 46.8942    |
| frequency_domain_lf_fusion_v2 (rgb loss) | 27.7904  | 27.9197 | 46.1408    |
