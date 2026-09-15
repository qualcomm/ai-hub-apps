# SAM3 on Snapdragon X Elite

A Windows Python app that runs SAM3 (Segment Anything Model 3) text-prompted instance segmentation on-device using ONNX Runtime QNN Execution Provider on Snapdragon X Elite.

## Requirements

- Windows on Snapdragon X Elite
- PowerShell

## Setup

SAM3 is **export-only** — its pre-compiled weights are not downloadable from AI Hub due to licensing restrictions, so you must export the ONNX bundle yourself first:

1. Accept the SAM3 model license at [facebook/sam3](https://huggingface.co/facebook/sam3) (sign in to HuggingFace and agree to the terms). Exporting downloads the source weights from there, which fails until the license is accepted.

2. Follow the [SAM3 model export instructions](https://github.com/qualcomm/ai-hub-models/tree/main/src/qai_hub_models/models/sam3) to export the `precompiled_qnn_onnx` bundle for Snapdragon X Elite.

Then set up the app with the exported bundle using one of the options below.

### Option A: Using the CLI (Recommended)

Install the CLI and fetch the app with the model:

```powershell
pip install qai-hub-apps
qai-hub-apps fetch sam3_segmentation_windows_py --model <path\to\exported_bundle> --output-dir ~
cd ~\sam3_segmentation_windows_py
```

This downloads the app source and places the model binaries in the correct location automatically:

- `models/vision_backbone.onnx`
- `models/head.onnx`

> [!NOTE]
> SAM3 is export-only, so `--model` takes the path to a bundle you exported yourself rather than a model ID. The CLI places the exported assets into the app automatically.

### Option B: Cloning the Repo

If you cloned the release branch, the app directory is already self-contained — but **model weights are not included**. Export the `precompiled_qnn_onnx` bundle and copy its files into `models/` so the layout matches what the CLI produces:

- `models/vision_backbone.onnx`
- `models/head.onnx`

See [Setup](#setup) above for export instructions.

> [!NOTE]
> The export is flat: each `*.onnx` sits next to its `*_qairt_context.bin` sidecar (which holds the precompiled weights the graph references by relative path), so keep every file in `models/` together. This matches the default `--backbone models\vision_backbone.onnx --head models\head.onnx` paths, so the plain `--image`/`--text-prompts` examples work without extra flags.

## Install Dependencies

Allow PowerShell scripts to run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser Unrestricted -Force
```

Install all dependencies:

> [!WARNING]
> This script installs packages at the user scope (Python). These will be available system-wide for the current user after installation.

```powershell
.\install_runtime.ps1
```

This installs:
- Python (native ARM64 on Snapdragon — every dependency below ships an ARM64 wheel)
- `numpy`, `pillow`, `tokenizers` (for the CLIP tokenizer)
- `onnxruntime-qnn` (ONNX Runtime with QNN Execution Provider for the Snapdragon NPU)

> [!NOTE]
> To skip the automatic Python install, comment out the `Install-WingetPackage` line in `install_runtime.ps1` and update the `-Python` argument to point to your own Python executable.

## Run

Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

Run with a text prompt:

```powershell
python main.py --image photo.jpg --text-prompts "cup"
```

Multiple prompts (one detection pass per prompt):

```powershell
python main.py --image photo.jpg --text-prompts "cup,bowl,person"
```

All options:

```powershell
python main.py --backbone models\vision_backbone.onnx `
               --head     models\head.onnx `
               --image    photo.jpg `
               --text-prompts "cup,bowl" `
               --confidence 0.5 `
               --output sam3_output.png
```

> [!NOTE]
> The backbone and head are precompiled QNN context binaries, so there's no first-run HTP compile step -- both stages load and run directly.

Example output:
![Example output](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-apps/apps/sam3_segmentation_windows_py/test/sam3_output.png)
