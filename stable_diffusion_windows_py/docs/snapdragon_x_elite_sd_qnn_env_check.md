# Snapdragon X Elite Stable Diffusion QNN Environment Check

Date: 2026-06-17

Repository:

- `quic/ai-hub-apps`
- Branch: `codex/fix-sd-qnn-v048-snapdragon-x-elite`
- Base branch at clone time: `release`
- Base commit observed earlier: `34e088e`

Application:

- `stable_diffusion_windows_py`

Python environment:

- Conda env: `AI_Hub_SD`
- Python executable: `C:\Users\hirok\miniconda3\envs\AI_Hub_SD\python.exe`
- Python version:

```text
3.11.15 | packaged by Anaconda, Inc. | (main, Jun 11 2026, 15:12:53) [MSC v.1942 64 bit (AMD64)]
```

- `platform.machine()`:

```text
AMD64
```

- `platform.architecture()`:

```text
('64bit', 'WindowsPE')
```

ONNX Runtime:

```text
onnxruntime.version= 1.24.1
onnxruntime.providers= ['QNNExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']
```

Packages:

```text
onnxruntime-qnn= 1.24.1
qai-hub-models= 0.48.0
qai-hub= 0.50.0
protobuf= 6.31.1
numpy= 2.4.6
```

QNN backend:

```text
C:\Users\hirok\miniconda3\envs\AI_Hub_SD\Lib\site-packages\onnxruntime\capi\QnnHtp.dll
```

Adopted model:

- Release: `v0.48.0`
- Chipset asset: `qualcomm-snapdragon-x-elite`
- Runtime: `precompiled_qnn_onnx`
- Precision: `w8a16`
- Metadata tool versions:

```text
qairt: 2.42.0.251225135753_193295
onnx_runtime: 1.24.1
```

Model files:

```text
metadata.yaml
text_encoder.onnx
text_encoder_qairt_context.bin
unet.onnx
unet_qairt_context.bin
vae.onnx
vae_qairt_context.bin
```

Notes:

- Regular `onnxruntime` must not be installed alongside `onnxruntime-qnn`.
- `pip show qai-hub-models` can hit a Windows console encoding issue because package metadata contains the registered trademark symbol. Version checks above were collected with `importlib.metadata`.

Verification runs:

```text
scripts/run_sd_v048_qnn.ps1
  ORT: 1.24.1
  Providers: ['QNNExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']
  Output: generated_default_v048_20steps/image.png

scripts/compare_sd_v048_steps.ps1
  Output: generated_v048_5steps_seed47/image.png
  Output: generated_v048_10steps_seed47/image.png
  Output: generated_v048_20steps_seed47/image.png
```
