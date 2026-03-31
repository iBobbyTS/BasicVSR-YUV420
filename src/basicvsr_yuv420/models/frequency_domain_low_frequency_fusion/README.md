# frequency_domain_low_frequency_fusion

This variant approximates the report's frequency-domain idea with a fixed low/high decomposition on the luma signal and a low-resolution fusion path shared with chroma.

## Report Method Mapping

This implementation corresponds to the frequency-inspired low-frequency fusion method from the research report.

## Implementation in This Repository

- The luma input is decomposed into a low-frequency component and a high-frequency residual through a fixed pooling-plus-reconstruction split
- The low-frequency luma component is fused with UV in a low-resolution recurrent branch
- UV is reconstructed from the fused low-resolution temporal features
- Y is reconstructed from the combination of upsampled low-frequency context and a dedicated high-frequency luma encoder

## Simplifications

- The frequency decomposition is fixed and convolutional rather than a learned wavelet or FFT pipeline
- No additional spectral loss is introduced
- The model keeps a lightweight reconstruction head for practical training

## Tensor Shapes

- Input Y: `[B, T, 1, H, W]`
- Input UV: `[B, T, 2, H/2, W/2]`
- Output Y: `[B, T, 1, 4H, 4W]`
- Output UV: `[B, T, 2, 2H, 2W]`

## Loss and Metrics

- `loss = 0.5 * loss_y + 0.5 * loss_uv`
- `psnr = 0.5 * psnr_y + 0.5 * psnr_uv`
- Optional SSIM follows the same weighted rule
