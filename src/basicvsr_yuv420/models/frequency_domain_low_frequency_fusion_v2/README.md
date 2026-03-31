# frequency_domain_low_frequency_fusion_v2

This variant keeps the original low-frequency recurrent fusion path but upgrades the luma high-frequency branch from a per-frame encoder to a lightweight bidirectional temporal branch.

## Motivation

The first version preserved luma high-frequency detail only through frame-local encoding. That kept the model fast, but it also limited how much temporal detail recovery the Y branch could learn.

## Implementation in This Repository

- The low-frequency luma component and UV are still fused in the same low-resolution bidirectional recurrent backbone
- UV reconstruction is unchanged and still comes from the low-resolution fused feature
- The high-frequency luma residual now has its own narrow bidirectional recurrent branch at full luma resolution
- The high-frequency branch uses fewer channels than the main low-resolution backbone so the extra temporal modeling stays lightweight
- Y reconstruction uses both:
  - upsampled low-frequency temporal context
  - temporally propagated high-frequency luma context

## Simplifications

- The high-frequency branch is intentionally narrow rather than matching the full backbone width
- The model still uses the same fixed pooling-plus-residual low/high split
- No additional spectral or perceptual loss is introduced

## Tensor Shapes

- Input Y: `[B, T, 1, H, W]`
- Input UV: `[B, T, 2, H/2, W/2]`
- Output Y: `[B, T, 1, 4H, 4W]`
- Output UV: `[B, T, 2, 2H, 2W]`

## Expected Tradeoff

- Better Y temporal detail modeling than `frequency_domain_low_frequency_fusion`
- Slightly higher inference cost than v1
- Still intended to remain clearly lighter than the RGB baseline
