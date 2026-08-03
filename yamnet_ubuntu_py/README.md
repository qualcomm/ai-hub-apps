# YamNet Audio Classification app

A Python app using LiteRT that performs on-device audio event classification.
It reads an audio file, computes log-mel features on the CPU, runs the YamNet
model on the Qualcomm NPU via the QAIRT TFLite delegate, and prints the top
predicted [AudioSet](https://research.google.com/audioset/) sound classes.

## Requirements

- ARM64 Ubuntu 22.04+ or compatible Linux
- Docker

## Setup

### Option A: Using the CLI (Recommended)

Install the CLI and fetch the app with the model:

```bash
pip install qai-hub-apps
qai-hub-apps fetch yamnet_ubuntu_py --model yamnet --output-dir ~
cd ~/yamnet_ubuntu_py
```

> [!NOTE]
> To use a model you exported yourself with [AI Hub Models](https://github.com/qualcomm/ai-hub-models),
> pass the exported model path to `--model` in place of a model ID. The CLI places the exported
> assets into the app automatically:
>
> ```bash
> qai-hub-apps fetch yamnet_ubuntu_py --model <path\to\exported_model>
> ```

### Option B: Cloning the Repo

If you cloned the release branch, the app directory is already self-contained — but **model weights are not included**. Download a compatible model from [AI Hub Models](https://aihub.qualcomm.com/iot/models), unzip the bundle and copy the following files into place before building:
- `models/yamnet.tflite`
- `models/labels.txt`

## Build

### Install Docker

Follow [these instructions](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) to install Docker.

### Install Ubuntu host packages (Dragonwing devices)

Add the Qualcomm PPA and install the required host packages:

```bash
sudo apt-add-repository -y ppa:ubuntu-qcom-iot/qcom-ppa
sudo apt-get update
sudo apt-get install qcom-fastrpc1 qcom-fastrpc-dev
```

After installing, reboot the device.

### Using Docker
From the app directory, build our Docker image with all required runtime dependencies, including the supported QAIRT SDK.
```bash
docker build --build-arg BUILD_TYPE=runtime -t aiha-yamnet .
```

## Run

```bash
./run_docker.sh --interactive
```
Inside the container:
```bash
bash test.sh
```

`test.sh` downloads a test audio clip and runs the app against it using the QAIRT runtime.

### Classify your own audio

```bash
./run_docker.sh --hexagon-version <HEX_VER> --audio-file /path/to/audio.wav
```

The app prints the top predicted classes, e.g.:

```
Top 5 predictions: Speech | Whistling | Music | Inside, small room | Silence
```

> [!IMPORTANT]
> You must provide `--hexagon-version` matching your device's Hexagon DSP version. For example, the [Dragonwing RB3 Gen 2](https://www.qualcomm.com/developer/hardware/rb3-gen-2-development-kit) uses Hexagon v68. To find the Hexagon version for your device, visit the [AI Hub device catalogue](https://workbench.aihub.qualcomm.com/devices/).

> [!NOTE]
> Audio of any sample rate, channel count, or duration is accepted: it is
> resampled to 16 kHz mono and split into ~1 s segments, whose scores are
> averaged for the final prediction.
