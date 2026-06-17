# Snapdragon X Elite Stable Diffusion QNN Intermediate Debug Report

## Status

- Target: Snapdragon X Elite / Windows ARM64 / Qualcomm QNN Stable Diffusion diagnosis.
- Fixed stack: `qai_hub_models==0.48.0`, `onnxruntime-qnn==1.24.1`.
- `v0.56.0` was not used.
- Existing `demo.py` was not modified.
- QNN 5-step intermediate diagnosis completed successfully.
- CPU comparison was requested, but skipped because the available ONNX files are precompiled QNN context-wrapper ONNX assets, not portable CPU baseline exports.
- Existing `demo.py` smoke test was not run after diagnosis because command execution approval was not granted.
- Draft PR creation is still pending.

## Added / Generated Files

- `stable_diffusion_windows_py/diagnose_intermediate_qnn.py`
- `stable_diffusion_windows_py/outputs/intermediate_debug/tensor_stats.json`
- `stable_diffusion_windows_py/outputs/intermediate_debug/tensor_stats.md`
- `stable_diffusion_windows_py/outputs/intermediate_debug/onnx_io.md`
- `stable_diffusion_windows_py/outputs/intermediate_debug/model_files.md`
- `stable_diffusion_windows_py/outputs/intermediate_debug/summary.md`
- `stable_diffusion_windows_py/outputs/intermediate_debug/unet_text_emb.md`
- `stable_diffusion_windows_py/outputs/intermediate_debug/guidance_sensitivity.md`
- `stable_diffusion_windows_py/outputs/intermediate_debug/guidance_sensitivity.json`
- `stable_diffusion_windows_py/outputs/intermediate_debug/unet_reference_compare.md`
- `stable_diffusion_windows_py/outputs/intermediate_debug/unet_reference_compare.json`
- `stable_diffusion_windows_py/outputs/intermediate_debug/cpu_compare.md`
- `stable_diffusion_windows_py/outputs/intermediate_debug/diagnostic_image.png`
- `stable_diffusion_windows_py/outputs/intermediate_debug/report.md`

## Command Run

```powershell
& 'C:\Users\hirok\miniconda3\envs\AI_Hub_SD\python.exe' diagnose_intermediate_qnn.py --num-steps 5 --seed 47 --output-dir outputs/intermediate_debug --cpu-compare
```

The command completed successfully after changing CPU comparison to a safe skip/report path.

## Environment Observed

- Python: `3.11.15`
- Python architecture: `AMD64`
- ONNX Runtime import version: `1.24.1`
- Available providers: `QNNExecutionProvider`, `AzureExecutionProvider`, `CPUExecutionProvider`
- Prompt: `A girl taking a walk at sunset`
- Steps: `5`
- Seed: `47`
- Guidance scale: `7.5`

## Model File Check

`model/metadata.yaml` reports:

- runtime: `precompiled_qnn_onnx`
- precision: `w8a16`
- QAIRT: `2.42.0.251225135753_193295`
- ONNX Runtime: `1.24.1`

Required files were present:

- `metadata.yaml`: 1,480 bytes
- `text_encoder.onnx`: 733 bytes
- `text_encoder_qairt_context.bin`: 396,058,624 bytes
- `unet.onnx`: 1,714 bytes
- `unet_qairt_context.bin`: 882,139,136 bytes
- `vae.onnx`: 873 bytes
- `vae_qairt_context.bin`: 64,622,592 bytes

Metadata matched the expected v0.48.0-style precompiled QNN model layout.

## ONNX I/O

- `text_encoder.onnx`
  - input: `tokens`, `[1, 77]`, `INT32`
  - output: `text_embedding`, `[1, 77, 1024]`, `UINT16`
- `unet.onnx`
  - input: `latent`, `[1, 64, 64, 4]`, `UINT16`
  - input: `timestep`, `[1, 1]`, `UINT16`
  - input: `text_emb`, `[1, 77, 1024]`, `UINT16`
  - output: `output_latent`, `[1, 64, 64, 4]`, `UINT16`
- `vae.onnx`
  - input: `latent`, `[1, 64, 64, 4]`, `UINT16`
  - output: `image`, `[1, 512, 512, 3]`, `UINT16`

## Key Tensor Statistics

| Tensor | Shape | Dtype | Min | Max | Mean | Std | NaN | Inf | Zero ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TextEncoder cond embedding | `[1, 77, 1024]` | `float32` | -6.6315 | 12.9927 | -0.1697 | 1.0241 | 0 | 0 | 0.000114 |
| TextEncoder uncond embedding | `[1, 77, 1024]` | `float32` | -5.7696 | 12.9927 | -0.1681 | 1.0404 | 0 | 0 | 0.000203 |
| TextEncoder cond - uncond | `[1, 77, 1024]` | `float32` | -9.3342 | 8.0495 | -0.00156 | 0.6973 | 0 | 0 | 0.0133 |
| Initial latent scaled | `[1, 4, 64, 64]` | `float32` | -4.1335 | 3.8601 | 0.00747 | 0.9939 | 0 | 0 | 0 |
| Step 1 UNet noise cond | `[1, 64, 64, 4]` | `float32` | -4.1718 | 3.8936 | 0.00233 | 0.9988 | 0 | 0 | 0.000122 |
| Step 1 noise pred | `[1, 4, 64, 64]` | `float32` | -4.1801 | 3.9319 | -0.00313 | 1.0100 | 0 | 0 | 0 |
| Step 1 noise cond - uncond | `[1, 4, 64, 64]` | `float32` | -0.0527 | 0.0492 | -0.000841 | 0.00912 | 0 | 0 | 0.00879 |
| Step 5 latent after scheduler | `[1, 4, 64, 64]` | `float32` | -0.5060 | 0.6631 | 0.00485 | 0.1170 | 0 | 0 | 0 |
| VAE output image float | `[1, 512, 512, 3]` | `float32` | 0.2659 | 0.5824 | 0.4401 | 0.0541 | 0 | 0 | 0 |
| Image before save uint8 | `[512, 512, 3]` | `uint8` | 68 | 149 | 112.2149 | 13.8003 | 0 | 0 | 0 |

## Diagnosis

The strongest suspect is the UNet guidance path.

TextEncoder itself does not look collapsed:

- Conditional and unconditional embeddings have healthy variance.
- `cond - uncond` also has significant variance.
- No NaN or Inf was observed.

Initial latent also looks normal:

- Initial scaled latent std is about `0.9939`.
- No NaN, Inf, or unexpected zeroing was observed.

UNet output is numerically active, but the prompt-conditioning delta is extremely small:

- Step 1 `noise_pred` std: `1.0100`
- Step 1 `noise_cond_minus_uncond` std: `0.00912`
- Ratio: about `0.9%`

This suggests that UNet is producing noise-like output, but conditional vs unconditional text conditioning is barely changing that output. If the final image is low-information or weakly prompt-dependent, the likely breakage is between TextEncoder output and UNet conditioning consumption, not in raw TextEncoder generation.

VAE does not appear to be a hard collapse point:

- VAE output std is `0.0541`.
- Saved uint8 image std is `13.8003`.
- Values are compressed, but not single-valued.

## UNet text_emb Follow-up

Additional instrumentation was added around `OnnxModelTorchWrapper._prepare_inputs()`.

The installed `qai_hub_models==0.48.0` wrapper quantizes float inputs as:

```text
uint_value = round(float_value / scale) + zero_point
uint_value = clip(uint_value, dtype_min, dtype_max).astype(dtype)
```

For `unet.text_emb`, the metadata/wrapper QDQ parameters are:

- dtype: `uint16`
- scale: `0.00034632044844329357`
- zero_point: `23638`

The cond/uncond text embedding difference survives quantization:

- before quant std: `0.69729782`
- uint16 signed delta std: `2013.4483`
- dequantized-after-quant std: `0.69729833`
- quantized delta zero ratio: `0.013278713`

This means the cond/uncond difference is not being erased by the float32-to-uint16 conversion.

Two very different prompts were also compared at the same initial latent and timestep:

- prompt A: `A girl taking a walk at sunset`
- prompt B: `A red sports car parked in a snowy mountain at night`
- prompt A/B text delta after dequant std: `0.66667875`
- prompt A/B UNet noise delta std: `0.017968048`

The UNet output does change when the prompt changes, so `text_emb` is reaching the UNet in some form. However, the output response is still very small compared with the full UNet noise scale near `~1.0`.

## Guidance Scale Sensitivity Follow-up

Additional guidance sensitivity testing was run with the same seed and the same initial latent:

```text
guidance_scale = 0, 1, 3, 7.5, 15, 30
```

The sweep measured step-1 `noise_pred`, step-1 `cond/uncond` delta, final latent, and final image differences against guidance scale `0`.

| guidance_scale | noise_pred_std | cond_uncond_delta_std | final_latent_std | image_std | noise_pred_absdiff_vs_0_mean | final_latent_absdiff_vs_0_mean | image_absdiff_vs_0_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.99741883 | 0.0091155551 | 0.12110498 | 12.963764 | 0 | 0 | 0 |
| 1.0 | 0.99883529 | 0.0091155551 | 0.12022568 | 13.072959 | 0.0073005874 | 0.0018734032 | 0.34219615 |
| 3.0 | 1.001911 | 0.0091155551 | 0.11863965 | 13.302754 | 0.021901764 | 0.0054803123 | 0.69231669 |
| 7.5 | 1.0100011 | 0.0091155551 | 0.1170036 | 13.800269 | 0.054754406 | 0.013224191 | 1.628376 |
| 15.0 | 1.0269899 | 0.0091155551 | 0.13454428 | 15.066388 | 0.10950881 | 0.041350424 | 5.0663185 |
| 30.0 | 1.073296 | 0.0091155551 | 0.20763276 | 17.810982 | 0.21901762 | 0.096148282 | 8.7554131 |

Guidance scale is not ignored: increasing it does move `noise_pred`, final latent, and the final image. However, the underlying `cond/uncond` delta remains small (`0.0091155551`) relative to the full noise scale (`~1.0`). This points away from a Python-side guidance-scale bug and toward weak text-conditioning sensitivity inside the QNN UNet/context path.

## PyTorch UNet Reference Follow-up

A PyTorch `UNet2DConditionModel` baseline from `sd2-community/stable-diffusion-2-1` was loaded with `diffusers` and compared against the QNN UNet using the same:

- prompt: `A girl taking a walk at sunset`
- seed: `47`
- timestep: `801`
- initial latent
- QNN TextEncoder `cond` and `uncond` embeddings

Key result:

| tensor | std |
| --- | --- |
| QNN `noise_cond_minus_uncond` | `0.0091155551` |
| PyTorch reference `noise_cond_minus_uncond` | `0.17506623` |
| QNN / PyTorch ratio | `0.05206918` |

The same text embedding delta that reaches the QNN UNet produces a much larger conditioning response in the PyTorch baseline. QNN is only about `5.2%` of the reference delta magnitude for this one-step comparison.

This strongly supports the suspicion that the issue is inside `unet_qairt_context.bin` or the UNet QNN conversion settings, rather than TextEncoder output, Python-side `text_emb` quantization, or guidance scale application.

## Most Likely Cause Area

Priority suspicion:

1. UNet text conditioning is weak after the input reaches the QNN context, rather than lost during Python-side quantization or ignored guidance scaling. The PyTorch reference comparison confirms QNN `cond/uncond` delta is only about `5.2%` of the baseline.
2. UNet model/context may be insensitive to text conditioning because of conversion/export/context-bin issue.
3. Scheduler/timestep scaling may be dominating or compressing the guidance effect.
4. Conditional/unconditional batching or input ordering mismatch is less likely because direct prompt A/B calls do change UNet output, but it should still be checked against a known-good reference.
5. VAE is lower priority because it receives a non-constant latent and produces a non-constant image.

## Next Fixes To Try

1. Rebuild or reacquire `unet_qairt_context.bin` and compare whether the QNN `cond/uncond` delta approaches the PyTorch baseline.
2. Verify UNet context generation settings for text embedding input quantization, cross-attention blocks, and any graph partition/fallback behavior.
3. Obtain non-context ONNX exports for UNet/TextEncoder/VAE, then run an ONNXRuntime CPU/QNN-style comparison outside the precompiled QNN context wrapper files.
4. Feed synthetic extreme `text_emb` inputs into UNet and confirm output sensitivity.
5. Run the original `demo.py` smoke test once command approval is available.

## Remaining Work

- Run existing `demo.py` smoke test.
- Stage and commit changes.
- Push the work branch.
- Open Draft PR.
