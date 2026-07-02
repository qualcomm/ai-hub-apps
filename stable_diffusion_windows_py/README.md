# Stable Diffusion on Snapdragon X Elite

A Windows Python app that runs Stable Diffusion v2.1 on-device using ONNX Runtime QNN on Snapdragon X Elite.

## Requirements

- Windows on Snapdragon X Elite
- PowerShell

## Setup

### Option A: Using the CLI (Recommended)

Install the CLI and fetch the app with the model:

```powershell
pip install qai-hub-apps
qai-hub-apps fetch stable_diffusion_windows_py --model stable_diffusion_v2_1 --chipset qualcomm-snapdragon-x-elite --output-dir ~
cd ~\stable_diffusion_windows_py
```

> [!NOTE]
> To use a model you exported yourself with [AI Hub Models](https://github.com/qualcomm/ai-hub-models),
> pass the exported model path to `--model` in place of a model ID. The CLI places the exported
> assets into the app automatically:
>
> ```powershell
> qai-hub-apps fetch stable_diffusion_windows_py --model <path\to\exported_model>
> ```

### Option B: Cloning the Repo

If you cloned the release branch, the app directory is already self-contained — but **model weights are not included**. Download the `PRECOMPILED_QNN_ONNX` model files from [Qualcomm HuggingFace Repo](https://huggingface.co/qualcomm/Stable-Diffusion-v2.1) for your target device (e.g. `Snapdragon® X Elite`), extract the zip to `<APP ROOT>/model/`. The expected directory structure is:

```
model/
  |_ metadata.yaml
  |_ text_encoder.onnx
  |_ text_encoder_qairt_context.bin
  |_ unet.onnx
  |_ unet_qairt_context.bin
  |_ vae.onnx
  |_ vae_qairt_context.bin
```

## Install Dependencies

First, allow PowerShell scripts to run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser Unrestricted -Force
```

Then install all platform and Python dependencies:

> [!WARNING]
> This script installs packages at the user scope (Python, Git). These will be available system-wide for the current user after installation.

```powershell
.\install_runtime.ps1
```

This installs:
- x64 Python (required — ARM64 Python is not supported by all dependencies)
- Git for Windows
- Python packages including `qai-hub-models` and `onnxruntime-qnn`

> [!NOTE]
> If you don't want us to install Python for you via winget, open `install_runtime.ps1` and
> comment out the `Install-WingetPackage -Id "Python.Python.*"` line. Then update the `-Python`
> argument in the subsequent `Install-PipDeps` calls to point to your own x64 Python executable
> (e.g. `-Python "C:\Python311\python.exe"`).

## Run

```powershell
.venv\Scripts\Activate.ps1
python demo.py --prompt "A girl taking a walk at sunset" --num-steps 20
```
