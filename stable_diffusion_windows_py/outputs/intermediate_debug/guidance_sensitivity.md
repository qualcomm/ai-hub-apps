# Guidance Sensitivity Debug

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
  "guidance_scales": [
    0.0,
    1.0,
    3.0,
    7.5,
    15.0,
    30.0
  ]
}
```

## Scale Comparison

| guidance_scale | noise_pred_std | cond_uncond_delta_std | final_latent_std | image_std | noise_pred_absdiff_vs_0_mean | final_latent_absdiff_vs_0_mean | image_absdiff_vs_0_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.99741883 | 0.0091155551 | 0.12110498 | 12.963764 | 0 | 0 | 0 |
| 1.0 | 0.99883529 | 0.0091155551 | 0.12022568 | 13.072959 | 0.0073005874 | 0.0018734032 | 0.34219615 |
| 3.0 | 1.001911 | 0.0091155551 | 0.11863965 | 13.302754 | 0.021901764 | 0.0054803123 | 0.69231669 |
| 7.5 | 1.0100011 | 0.0091155551 | 0.1170036 | 13.800269 | 0.054754406 | 0.013224191 | 1.628376 |
| 15.0 | 1.0269899 | 0.0091155551 | 0.13454428 | 15.066388 | 0.10950881 | 0.041350424 | 5.0663185 |
| 30.0 | 1.073296 | 0.0091155551 | 0.20763276 | 17.810982 | 0.21901762 | 0.096148282 | 8.7554131 |

## Conclusion

Guidance scale changes do affect noise_pred, final latent, and image output. This argues against a totally disconnected guidance path. If the generated image still looks low-information, the remaining suspicion is weak UNet text-conditioning sensitivity inside the QNN context rather than Python-side guidance_scale being ignored.

## Output Images

- `guidance_0p0.png`
- `guidance_1p0.png`
- `guidance_3p0.png`
- `guidance_7p5.png`
- `guidance_15p0.png`
- `guidance_30p0.png`
