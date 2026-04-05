### [baseline_rgb_v3_noamp_lr1e4_bs2_acc2](baseline_rgb_v3_noamp_lr1e4_bs2_acc2)
Baseline RGB model, this is not the official BasicVSR model. It's derived from a community implemented notebook. 

### [baseline_rgb_v7_yuv420_input_tuned_bs4_vb1_nw1](baseline_rgb_v7_yuv420_input_tuned_bs4_vb1_nw1)
Change input from LR RGB to LR RGB -> YUV420 -> RGB, this ensures our following expieriments are only affected by the model, not both model and input type. But the result shows using YUV420 input will not increase the performance of the baseline. 

### [low_res_joint_y_head_v2_bs8_vb1_nw4](low_res_joint_y_head_v2_bs8_vb1_nw4)
Improves the model by moving most temporal modeling into a low-resolution joint Y+UV space and using a lightweight high-resolution Y head only at the end, which reduces computation while preserving a dedicated luminance reconstruction path.
### [low_res_joint_y_head_v3_rgb_objective_bs8_vb1_nw4](low_res_joint_y_head_v3_rgb_objective_bs8_vb1_nw4)
We realized that our target is not a good YUV420 output, because it will be converted to RGB to display, so we converted the predicted output to RGB before computing the loss. 

### [frequency_domain_low_frequency_fusion_v2_rgb_objective_bs7_vb1_nw2](frequency_domain_low_frequency_fusion_v2_rgb_objective_bs7_vb1_nw2)
Improves the model by splitting luminance Y into a low-frequency temporal branch and a high-frequency detail branch, so most computation stays in low-resolution Y+UV fusion while preserving more Y detail at reconstruction time.

### [uv_conditioned_film_v3_rgb_objective_bs6_vb6_nw1](uv_conditioned_film_v3_rgb_objective_bs6_vb6_nw1)
### [uv_conditioned_film_v4_rgb_objective_resume_bs4_vb1_nw1](uv_conditioned_film_v4_rgb_objective_resume_bs4_vb1_nw1)
Improves the model by keeping a full-resolution recurrent Y backbone and using a low-resolution UV branch to generate conditioning signals that dynamically modulate Y feature propagation and reconstruction.

### [inference_benchmarks_20260331_50frames](inference_benchmarks_20260331_50frames)
Not a training result, it's memory and speed benchmark. 
