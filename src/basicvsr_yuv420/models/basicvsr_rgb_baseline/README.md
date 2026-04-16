# basicvsr_rgb_baseline

This model keeps the original notebook-derived RGB recurrent super-resolution implementation used by the repository before the YUV420 extension work.

## Intended Role

- Serves as the default RGB baseline
- Provides the reference training pipeline before any YUV420-specific architectural changes
- Preserves the original repository behavior for backward compatibility

## Inputs and Outputs

- Input: `lr_rgb` tensor shaped `[B, T, 3, H, W]`
- Output: RGB tensor shaped `[B, T, 3, 4H, 4W]`

## Notes

- This is not an official OpenMMLab BasicVSR reproduction
- It remains the baseline model used for fair comparison against the YUV420 variants
- Loss and PSNR for this model are still computed in RGB space
