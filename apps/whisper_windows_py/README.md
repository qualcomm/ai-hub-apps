# Whisper on Snapdragon X Elite

A Windows Python app that runs OpenAI's Whisper speech-to-text on-device using ONNX Runtime QNN on Snapdragon X Elite.

## Requirements

- Windows on Snapdragon X Elite
- PowerShell

## Setup

### Option A: Using the CLI (Recommended)

Install the CLI and fetch the app with the model:

```powershell
pip install qai-hub-apps
qai-hub-apps fetch whisper_windows_py --model whisper_base --chipset qualcomm-snapdragon-x-elite --output-dir ~
cd ~\whisper_windows_py
```

> [!NOTE]
> To use a model you exported yourself with [AI Hub Models](https://github.com/qualcomm/ai-hub-models),
> pass the exported model path to `--model` in place of a model ID. The CLI places the exported
> assets into the app automatically:
>
> ```powershell
> qai-hub-apps fetch whisper_windows_py --model <path\to\exported_model>
> ```

### Option B: Cloning the Repo

If you cloned the release branch, the app directory is already self-contained — but **model weights are not included**. Download a compatible model from [AI Hub Models](https://aihub.qualcomm.com/mobile/models), and place the ONNX models at:
- `models/encoder.onnx`
- `models/decoder.onnx`

## Install Dependencies

First, allow PowerShell scripts to run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser Unrestricted -Force
```

Then install all platform and Python dependencies:

> [!WARNING]
> This script installs packages at the user scope (Python, Git, ffmpeg). These will be available system-wide for the current user after installation.

```powershell
.\install_runtime.ps1
```

This installs:
- ARM64 Python (native ARM64 is required — running x64 Python under emulation breaks NPU access via ONNX Runtime QNN)
- Git for Windows
- ffmpeg for reading audio files
- Python packages listed in `requirements.txt` (Transformers, ONNX Runtime QNN, etc.)

> [!NOTE]
> To skip the automatic Python install, comment out the `Install-Python` line in
> `install_runtime.ps1` and add a `-Python` argument to the `Install-PipDeps` calls pointing
> to your own ARM64 Python executable (e.g. `-Python "C:\Python311-arm64\python.exe"`).

## Run

Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

### List microphone devices

```powershell
python demo.py --list-audio-devices
```

### Stream from a microphone

```powershell
python demo.py --stream-audio-device <device_number>
```

### Transcribe an audio file

```powershell
python demo.py --audio-file <path-or-url>
```

For example, with a [sample audio file](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/hf_whisper_asr_shared/v1/audio/fox.wav):

```powershell
python demo.py --audio-file fox.wav
```
