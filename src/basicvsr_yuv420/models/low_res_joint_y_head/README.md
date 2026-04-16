# low_res_joint_y_head

This variant pushes the main temporal backbone into low-resolution joint YUV space and uses a dedicated full-resolution Y reconstruction head.

## Report Method Mapping

This implementation corresponds to the low-resolution joint backbone with a higher-resolution Y head.

## Implementation in This Repository

- LR Y is downsampled to UV resolution and concatenated with UV to form a 3-channel low-resolution recurrent input
- Bidirectional temporal propagation happens entirely in that low-resolution joint space
- UV is reconstructed directly from the propagated low-resolution features
- Y is reconstructed by combining shallow full-resolution Y features with upsampled low-resolution temporal context

## Simplifications

- The low-resolution backbone keeps the same recurrent structure in both directions
- Y reconstruction uses a lightweight fusion head instead of a deeper refinement network
- Losses remain purely weighted YUV420 reconstruction terms

## Tensor Shapes

- Input Y: `[B, T, 1, H, W]`
- Input UV: `[B, T, 2, H/2, W/2]`
- Output Y: `[B, T, 1, 4H, 4W]`
- Output UV: `[B, T, 2, 2H, 2W]`

## Loss and Metrics

- `loss = 0.5 * loss_y + 0.5 * loss_uv`
- `psnr = 0.5 * psnr_y + 0.5 * psnr_uv`
- Optional SSIM follows the same weighted rule
