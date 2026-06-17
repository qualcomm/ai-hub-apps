# Intermediate Debug Summary

## Environment

- Python architecture: `AMD64`
- ONNX Runtime: `1.24.1`
- Providers: `['QNNExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']`
- Prompt: `A girl taking a walk at sunset`
- Alt prompt: `A red sports car parked in a snowy mountain at night`
- Steps: `5`
- Seed: `47`

## Key Statistics

| tensor | shape | dtype | min | max | mean | std | zero_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| text_encoder.output.cond_embedding | [1, 77, 1024] | float32 | -6.6315475 | 12.992723 | -0.16970948 | 1.0241217 | 0.00011414367 |
| text_encoder.output.uncond_embedding | [1, 77, 1024] | float32 | -5.7696242 | 12.992723 | -0.16814759 | 1.0403697 | 0.00020292208 |
| text_encoder.output.cond_minus_uncond | [1, 77, 1024] | float32 | -9.3341761 | 8.0494576 | -0.0015618909 | 0.69729782 | 0.013278713 |
| step_1.unet.text_emb.cond_minus_uncond.float32_before_quant | [1, 77, 1024] | float32 | -9.3341761 | 8.0494576 | -0.0015618909 | 0.69729782 | 0.013278713 |
| step_1.unet.text_emb.cond_minus_uncond.uint16_after_quant_signed | [1, 77, 1024] | int32 | -26952 | 23243 | -4.5100954 | 2013.4483 | 0.013278713 |
| step_1.unet.text_emb.cond_minus_uncond.dequantized_after_quant | [1, 77, 1024] | float32 | -9.3340292 | 8.0495262 | -0.0015619383 | 0.69729833 | 0.013278713 |
| unet.prompt_compare.prompt_a_minus_b.noise_nchw | [1, 4, 64, 64] | float32 | -0.16292369 | 0.046575904 | -0.0042211618 | 0.017968048 | 0.0054931641 |
| latent.initial.scaled | [1, 4, 64, 64] | float32 | -4.1334796 | 3.86008 | 0.0074700304 | 0.993912 | 0 |
| step_1.unet.output.noise_cond_nhwc | [1, 64, 64, 4] | float32 | -4.1717653 | 3.8935988 | 0.0023330584 | 0.99883529 | 0.00012207031 |
| step_1.noise_pred.nchw | [1, 4, 64, 64] | float32 | -4.18014 | 3.931891 | -0.0031343835 | 1.0100011 | 0 |
| step_5.latent.after_scheduler | [1, 4, 64, 64] | float32 | -0.50596952 | 0.66309458 | 0.0048467099 | 0.1170036 | 0 |
| vae.input.latent_nchw | [1, 4, 64, 64] | float32 | -0.50596952 | 0.66309458 | 0.0048467099 | 0.1170036 | 0 |
| vae.output.image_float | [1, 512, 512, 3] | float32 | 0.26594949 | 0.5824216 | 0.44006879 | 0.054100643 | 0 |
| image.before_save.uint8 | [512, 512, 3] | uint8 | 68 | 149 | 112.21487 | 13.800269 | 0 |

## Likely Collapse Point

UNet conditional-unconditional noise delta is very small versus noise prediction (step 1 std ratio 0.0090); UNet text_emb cond/uncond delta survives uint16 quantization (zero ratio 0.0133); very different prompts change UNet noise with std 0.017968

## Notes

- v0.56.0 was not used.
- The script uses the existing v0.48.0 `model/` directory and `onnxruntime-qnn==1.24.1` environment.
- `diagnostic_image.png` is produced from the same 5-step diagnostic path.

## CPU Comparison

CPU comparison was requested but skipped.

Reason: these v0.48.0 assets are precompiled QNN context-wrapper ONNX files; they are intended for QNNExecutionProvider and are not a portable CPU ONNX baseline. Use non-context ONNX exports for a meaningful CPUExecutionProvider comparison.
