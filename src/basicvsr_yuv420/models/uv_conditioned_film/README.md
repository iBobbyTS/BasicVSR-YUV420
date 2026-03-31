# uv_conditioned_film

This variant keeps a full-resolution recurrent Y backbone and introduces a separate low-resolution UV branch that modulates the Y features through FiLM-style conditioning.

## Report Method Mapping

This implementation corresponds to the UV-conditioned FiLM / gating idea from the research report.

## Implementation in This Repository

- Motion is estimated from the Y plane only through the shared SpyNet module
- A full-resolution Y recurrent branch performs the main temporal propagation
- A low-resolution UV recurrent branch runs in parallel on native YUV420 chroma resolution
- UV features generate FiLM parameters that modulate Y features during both backward and forward propagation
- Y and UV are reconstructed with separate output heads

## Simplifications

- The UV branch uses the same recurrent propagation pattern as the Y branch instead of a larger dedicated chroma encoder
- FiLM is applied through a single convolutional conditioner per direction
- No extra perceptual or cross-domain loss is added

## Tensor Shapes

- Input Y: `[B, T, 1, H, W]`
- Input UV: `[B, T, 2, H/2, W/2]`
- Output Y: `[B, T, 1, 4H, 4W]`
- Output UV: `[B, T, 2, 2H, 2W]`

## Loss and Metrics

- `loss = 0.5 * loss_y + 0.5 * loss_uv`
- `psnr = 0.5 * psnr_y + 0.5 * psnr_uv`
- Optional SSIM follows the same weighted rule
