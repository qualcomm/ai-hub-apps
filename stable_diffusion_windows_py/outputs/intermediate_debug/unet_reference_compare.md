# QNN UNet vs PyTorch UNet Reference Compare

## Environment

```json
{
  "python": "3.11.15 | packaged by Anaconda, Inc. | (main, Jun 11 2026, 15:12:53) [MSC v.1942 64 bit (AMD64)]",
  "platform_machine": "AMD64",
  "onnxruntime_version": "1.24.1",
  "onnxruntime_providers": [
    "QNNExecutionProvider",
    "AzureExecutionProvider",
    "CPUExecutionProvider"
  ],
  "prompt": "A girl taking a walk at sunset",
  "seed": 47,
  "num_steps": 5,
  "timestep_index": 0,
  "local_files_only": true,
  "timestep": 801.0,
  "reference_unet": "sd2-community/stable-diffusion-2-1"
}
```

## Key Statistics

| tensor | shape | dtype | min | max | mean | std | zero_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| text_emb.cond_minus_uncond | [1, 77, 1024] | float32 | -9.3341761 | 8.0494576 | -0.0015618909 | 0.69729782 | 0.013278713 |
| qnn.noise_cond | [1, 4, 64, 64] | float32 | -4.1717653 | 3.8935988 | 0.0023330584 | 0.99883529 | 0.00012207031 |
| qnn.noise_uncond | [1, 4, 64, 64] | float32 | -4.1704769 | 3.8877077 | 0.0031742033 | 0.99741883 | 0.00012207031 |
| qnn.noise_cond_minus_uncond | [1, 4, 64, 64] | float32 | -0.052651167 | 0.049153253 | -0.00084114489 | 0.0091155551 | 0.0087890625 |
| reference.noise_cond | [1, 4, 64, 64] | float32 | -0.7857486 | 0.77493656 | -0.028182452 | 0.18424929 | 0 |
| reference.noise_uncond | [1, 4, 64, 64] | float32 | -0.84489 | 1.2161658 | -0.026866755 | 0.2401635 | 0 |
| reference.noise_cond_minus_uncond | [1, 4, 64, 64] | float32 | -0.88647223 | 0.67739689 | -0.0013156969 | 0.17506623 | 0 |
| compare.delta_absdiff | [1, 4, 64, 64] | float32 | 4.9576163e-05 | 0.89402008 | 0.12608736 | 0.12008488 | 0 |

## Delta Ratio

- QNN delta std: `0.0091155551`
- PyTorch reference delta std: `0.17506623`
- QNN / reference std ratio: `0.05206918`

## Conclusion

QNN UNet text-conditioning delta is much smaller than the PyTorch baseline (std ratio 0.0520692). Treat `unet_qairt_context.bin` or UNet conversion settings as the primary suspect.
