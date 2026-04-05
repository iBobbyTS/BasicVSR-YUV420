### [baseline_rgb_v3_noamp_lr1e4_bs2_acc2](baseline_rgb_v3_noamp_lr1e4_bs2_acc2)
Baseline RGB model, this is not the official BasicVSR model. It's derived from a community implemented notebook. 

### [baseline_rgb_v7_yuv420_input_tuned_bs4_vb1_nw1](baseline_rgb_v7_yuv420_input_tuned_bs4_vb1_nw1)
Change input from LR RGB to LR RGB -> YUV420 -> RGB, this ensures our following expieriments are only affected by the model, not both model and input type. But the result shows using YUV420 input will not increase the performance of the baseline. 

### [low_res_joint_y_head_v2_bs8_vb1_nw4](low_res_joint_y_head_v2_bs8_vb1_nw4)
Improves the model by moving most temporal modeling into a low-resolution joint Y+UV space and using a lightweight high-resolution Y head only at the end, which reduces computation while preserving a dedicated luminance reconstruction path.
### [low_res_joint_y_head_v3_rgb_objective_bs8_vb1_nw4](low_res_joint_y_head_v3_rgb_objective_bs8_vb1_nw4)
We realized that our target is not a good YUV420 output, because it will be converted to RGB to display, so we converted the predicted output to RGB before computing the loss. 

### [inference_benchmarks_20260331_50frames](inference_benchmarks_20260331_50frames)
Not a training result, it's memory and speed benchmark. 
