# Portrait Segmentation app

A Python app using GStreamer, OpenCV, and LiteRT that performs real-time
portrait (person) segmentation on a live camera stream, overlaying the
segmentation mask on each frame.

## Requirements

- ARM64 Ubuntu 22.04+ or compatible Linux
- Docker

## Setup

### Option A: Using the CLI (Recommended)

Install the CLI and fetch the app with the model:

```bash
pip install qai-hub-apps
qai-hub-apps fetch portrait_segmentation_ubuntu_py --model sinet --output-dir ~
cd ~/portrait_segmentation_ubuntu_py
```

> [!NOTE]
> To use a model you exported yourself with [AI Hub Models](https://github.com/qualcomm/ai-hub-models),
> pass the exported model path to `--model` in place of a model ID. The CLI places the exported
> assets into the app automatically:
>
> ```bash
> qai-hub-apps fetch portrait_segmentation_ubuntu_py --model <path/to/exported_model>
> ```

### Option B: Cloning the Repo

If you cloned the release branch, the app directory is already self-contained — but **model weights are not included**. Download a compatible model from [AI Hub Models](https://aihub.qualcomm.com/iot/models), unzip the bundle and copy both files into the `models/` directory before building:
- `models/<model>.tflite` — the model weights
- `models/metadata.json` — the model I/O metadata that ships in the bundle

## Build

### Install Docker

Follow [these instructions](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) to install Docker.

> [!IMPORTANT]
> To use the Docker path, you currently need to build the base image that carries the
> toolchain, then point the app at it. From this app directory:
>
> ```bash
> docker build -f base.Dockerfile -t qai-hub-apps-ubuntu-base:local .
> export QAIHA_BASE_IMAGE=qai-hub-apps-ubuntu-base:local
> ```
>
> The image is reused by every later build, and the same one works for every Ubuntu app
> on the same CLI version. If the CLI is upgraded (or downgraded), make sure to rebuild
> the base. Alternatively, run the app natively on the host with `--no-docker` and skip
> Docker entirely.

### Install the camera server (Dragonwing RB3 built-in camera)

If you are using the built-in camera on a Dragonwing RB3, install
`qcom-camera-server`, then reboot the device:

```bash
sudo apt-add-repository -y ppa:ubuntu-qcom-iot/qcom-ppa
sudo apt-get update
sudo apt-get install qcom-camera-server
```

## Run

`./launch.sh` builds the app's Docker image — with all required runtime
dependencies, including the supported QAIRT SDK — on first use, installs the
app's dependencies inside the container, then runs the app. App arguments go
after `--`; add `--no-docker` to run natively on the host instead.

Start with the app's self-test:

```bash
./launch.sh --test -- --hexagon-version <HEX_VER>
```

`test.sh` downloads a test video and runs the app against it using the QAIRT runtime.

### List available cameras

```bash
./launch.sh -- --list-devices
```

### Run with a specific camera

```bash
./launch.sh -- --hexagon-version <HEX_VER> --video-device /dev/video0
```

> [!IMPORTANT]
> You must provide `--hexagon-version` matching your device's Hexagon DSP version. For example, the [Dragonwing RB3 Gen 2](https://www.qualcomm.com/developer/hardware/rb3-gen-2-development-kit) uses Hexagon v68. To find the Hexagon version for your device, visit the [AI Hub device catalogue](https://workbench.aihub.qualcomm.com/devices/).

> [!NOTE]
> To use the integrated camera of a Dragonwing RB3, the `qtiqmmfsrc` GStreamer plugin must be used.
> `./launch.sh -- --hexagon-version v68 --video-gstreamer-source "qtiqmmfsrc name=camsrc camera=0"`.

This serves the camera feed on port 8080. Open a browser and navigate to
`http://<device-ip>:8080` to view the stream.
