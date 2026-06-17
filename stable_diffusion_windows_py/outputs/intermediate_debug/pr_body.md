## Summary

Adds Snapdragon X Elite / Windows ARM64 / Qualcomm QNN diagnostic tooling for the v0.48.0 Stable Diffusion demo path without modifying the existing `demo.py`.

This PR is a cause report rather than a fix. The main finding is that QNN UNet text-conditioning sensitivity is much weaker than a PyTorch UNet baseline under the same inputs.

## Reproduction Conditions

- Target: Snapdragon X Elite / Windows ARM64 / Qualcomm QNN
- Fixed stack: `qai_hub_models==0.48.0`, `onnxruntime-qnn==1.24.1`
- `v0.56.0` was not used
- Existing `demo.py` was not modified
- Prompt: `A girl taking a walk at sunset`
- Seed: `47`
- Step count: `5`
- Reference comparison timestep: `801`
- Model layout: v0.48.0-style `precompiled_qnn_onnx` / `w8a16` context-wrapper files

## What Changed

- Added intermediate tensor diagnostics for TextEncoder, initial latents, UNet steps, VAE decode, and image tensor output.
- Added UNet `text_emb` quantization instrumentation around the installed `OnnxModelTorchWrapper._prepare_inputs()` behavior.
- Added guidance scale sensitivity sweep for `0, 1, 3, 7.5, 15, 30` using the same seed and initial latent.
- Added QNN UNet vs PyTorch UNet reference comparison using the same QNN TextEncoder embeddings, latent, and timestep.
- Added ONNX I/O, model file, metadata, tensor stats, and summary reports under `outputs/intermediate_debug/`.

## Commands Run

```powershell
& 'C:\Users\hirok\miniconda3\envs\AI_Hub_SD\python.exe' diagnose_intermediate_qnn.py --num-steps 5 --seed 47 --output-dir outputs/intermediate_debug --cpu-compare
& 'C:\Users\hirok\miniconda3\envs\AI_Hub_SD\python.exe' diagnose_guidance_sensitivity_qnn.py --num-steps 5 --seed 47 --output-dir outputs/intermediate_debug --guidance-scales 0 1 3 7.5 15 30
& 'C:\Users\hirok\miniconda3\envs\AI_Hub_SD\python.exe' compare_qnn_unet_reference.py --num-steps 5 --seed 47 --timestep-index 0 --output-dir outputs/intermediate_debug --local-files-only
```

CPU comparison against the available ONNX files is documented as skipped because those files are precompiled QNN context-wrapper ONNX assets, not portable CPU ONNX baselines.

## Comparison Method

The QNN UNet and a PyTorch `UNet2DConditionModel` baseline from `sd2-community/stable-diffusion-2-1` were run with the same:

- QNN TextEncoder conditional and unconditional `text_emb`
- initial latent
- scheduler timestep
- prompt and seed

The comparison measured conditional vs unconditional `noise_pred` delta from each UNet path.

## Numerical Result

| path | `noise_cond_minus_uncond` std |
| --- | --- |
| QNN UNet | `0.0091155551` |
| PyTorch reference UNet | `0.17506623` |
| QNN / PyTorch ratio | `0.05206918` |

The QNN UNet conditioning response is only about `5.2%` of the PyTorch reference response under the same text embedding, latent, and timestep.

## Causes That Look Unlikely

- TextEncoder hard collapse: conditional and unconditional embeddings have healthy variance and no NaN/Inf.
- Python-side `text_emb` quantization loss: cond/uncond differences survive float32 to uint16 conversion and dequantization.
- Guidance scale being ignored: changing guidance scale from `0` to `30` changes `noise_pred`, final latent, and final image.
- VAE hard collapse: VAE output and saved uint8 image are not single-valued, although the image is low-information.

## Remaining Primary Suspect

The primary suspect is weak text-conditioning sensitivity inside the QNN UNet context path, especially:

- `unet_qairt_context.bin`
- UNet QNN conversion settings
- cross-attention handling during conversion or context generation
- text embedding input quantization/scale handling inside the precompiled context

## Questions For Qualcomm

1. Is the v0.48.0 `unet_qairt_context.bin` expected to preserve Stable Diffusion v2.1 cross-attention sensitivity at roughly the same magnitude as the PyTorch UNet baseline?
2. Are there known issues in the `precompiled_qnn_onnx` / `w8a16` UNet context for Stable Diffusion v2.1 where text conditioning becomes weak while noise output remains active?
3. What exact UNet conversion command, calibration data, quantization settings, and QAIRT options were used to generate `unet_qairt_context.bin`?
4. Should `text_emb` be supplied as UINT16 using scale `0.00034632044844329357` and zero point `23638`, or is there any additional preprocessing expected by the QNN context?
5. Is there a portable non-context ONNX UNet for v0.48.0 that can be used as an ONNXRuntime CPU baseline against the QNN context-wrapper ONNX?
6. Are cross-attention blocks fully executed on QNN in this context, or can graph partitioning/fallback change the behavior?

## Key Files

- `stable_diffusion_windows_py/diagnose_intermediate_qnn.py`
- `stable_diffusion_windows_py/diagnose_guidance_sensitivity_qnn.py`
- `stable_diffusion_windows_py/compare_qnn_unet_reference.py`
- `stable_diffusion_windows_py/outputs/intermediate_debug/report.md`
- `stable_diffusion_windows_py/outputs/intermediate_debug/unet_reference_compare.md`
