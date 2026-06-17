# Snapdragon X Elite Stable Diffusion QNN Runbook

This document fixes the known-working Qualcomm AI Hub Stable Diffusion Windows sample setup for HP EliteBook Ultra G1q / Snapdragon X Elite / Windows ARM64.

## Baseline

Use the `v0.48.0` Snapdragon X Elite `precompiled_qnn_onnx` Stable Diffusion v2.1 model.

Recommended runtime:

- Conda env: `AI_Hub_SD`
- Python: x64 Python 3.11 under emulation
- `qai_hub_models[stable-diffusion-v2-1]~=0.48.0`
- `onnxruntime-qnn==1.24.1`

`onnxruntime-qnn==1.24.4` also completed the demo in earlier tests, but `1.24.1` matches the model metadata and is the preferred fixed version.

## Why Not v0.56.0

The latest `v0.56.0` Snapdragon X Elite model metadata expects:

```text
qairt: 2.45.0...
onnx_runtime: 1.25.0
```

PyPI currently does not provide `onnxruntime-qnn==1.25.0`; available versions skip from the `1.24.x` line to the `2.x` Plugin EP line. Testing `v0.56.0` with `onnxruntime-qnn==1.24.4` failed with QNN context load error 5000.

For this baseline, do not force `v0.56.0`.

## Install Sequence

The install order matters because `qai_hub_models` depends on regular `onnxruntime`, while this sample must use `onnxruntime-qnn`.

```powershell
& C:\Users\hirok\miniconda3\envs\AI_Hub_SD\python.exe -m pip install "qai_hub_models[stable-diffusion-v2-1]~=0.48.0"
& C:\Users\hirok\miniconda3\envs\AI_Hub_SD\python.exe -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml
& C:\Users\hirok\miniconda3\envs\AI_Hub_SD\python.exe -m pip install onnxruntime-qnn==1.24.1
& C:\Users\hirok\miniconda3\envs\AI_Hub_SD\python.exe -m pip install protobuf==6.31.1
```

Verify:

```powershell
& C:\Users\hirok\miniconda3\envs\AI_Hub_SD\python.exe -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

Expected:

```text
1.24.1
['QNNExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']
```

## Model

Expected `model/` contents:

```text
metadata.yaml
text_encoder.onnx
text_encoder_qairt_context.bin
unet.onnx
unet_qairt_context.bin
vae.onnx
vae_qairt_context.bin
```

The adopted model metadata reports:

```text
runtime: precompiled_qnn_onnx
precision: w8a16
qairt: 2.42.0.251225135753_193295
onnx_runtime: 1.24.1
```

## Run

Single default run:

```powershell
.\scripts\run_sd_v048_qnn.ps1
```

Manual equivalent:

```powershell
& C:\Users\hirok\miniconda3\envs\AI_Hub_SD\python.exe demo.py `
  --prompt "A girl taking a walk at sunset" `
  --num-steps 20 `
  --seed 47 `
  --output-dir generated_default_v048_20steps
```

Step comparison:

```powershell
.\scripts\compare_sd_v048_steps.ps1
```

Expected output directories:

```text
generated_v048_5steps_seed47
generated_v048_10steps_seed47
generated_v048_20steps_seed47
```

Verified on 2026-06-17:

```text
generated_default_v048_20steps/image.png
generated_v048_5steps_seed47/image.png
generated_v048_10steps_seed47/image.png
generated_v048_20steps_seed47/image.png
```

## Known Status

Successful:

- `v0.48.0` + `onnxruntime-qnn==1.24.1`
- `v0.48.0` + `onnxruntime-qnn==1.24.4`

Failed:

- `v0.56.0` + `onnxruntime-qnn==1.24.4`
- Failure: QNN context load error 5000

Current quality note:

- The demo completes and writes images.
- Earlier generated images were mostly flat/monochrome, so output quality still needs follow-up investigation.
- This runbook intentionally focuses on freezing the reproducible QNN execution baseline before further quality debugging.

## Do Not Mix

- Do not mix regular `onnxruntime` with `onnxruntime-qnn`.
- Do not mix ARM64 Python experiments with this x64 Conda sample environment.
- Do not upgrade the working `AI_Hub_SD` environment casually.
- Do not force `v0.56.0` until a compatible `onnxruntime-qnn` path is available or the sample is ported to the 2.x Plugin EP API.
