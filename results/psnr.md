| Model                                  | RGB-PSNR | Y-PSNR  | UV420-PSNR |
|----------------------------------------|----------|---------|------------|
| baseline_rgb (rgb input)               | 28.0913  | 28.0972 | 53.6855    |
| baseline_rgb (yuv420 input)            | 27.7906  | 28.0385 | 44.2885    |
| baseline_rgb (trained on yuv420 input) | 27.3635  | 27.8700 | 40.8040    |
| uv_conditioned_film (rgb loss)         | 27.8463  | 27.9722 | 46.6212    |
| low_res_joint_y_head (yuv420 loss)     | 27.7765  | 27.9041 | 46.7719    |
| low_res_joint_y_head                   | 27.8092  | 27.9103 | 47.2815    |
| frequency_domain_lf_fusion             | 27.9083  | 28.0224 | 46.8942    |
| frequency_domain_lf_fusion_v2          | 27.7904  | 27.9197 | 46.1408    |
