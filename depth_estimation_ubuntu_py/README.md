# Depth Estimation app

A Python app using GStreamer, OpenCV, and LiteRT that performs real-time
monocular depth estimation on a live camera stream, streaming a color depth
heatmap for each frame.

The app is model-agnostic: it reads the model file name and input geometry from
the `metadata.json` that ships with the AI Hub model asset, so the same code
runs any of the depth-estimation models exported from
[AI Hub Models](https://github.com/qualcomm/ai-hub-models) (Depth-Anything,
Depth-Anything-V2/V3, Midas, ...).

> [!WARNING]
> The container runs `--privileged` for DSP/camera access and serves the web
> UI on `0.0.0.0:8080` with **no authentication**. Only run this on a
> trusted network.

## Requirements

- ARM64 Ubuntu 22.04+ or compatible Linux
- Docker

## Setup

### Option A: Using the CLI (Recommended)

Install the CLI and fetch the app with the model:

```bash
pip install qai-hub-apps
qai-hub-apps fetch depth_estimation_ubuntu_py --model depth_anything_v3 --output-dir ~
cd ~/depth_estimation_ubuntu_py
```
> [!NOTE]
> This app is model-agnostic — swap `--model` for any of `depth_anything_v3`,
> `depth_anything_v2`, `depth_anything`, or `midas` and the app runs
> unchanged; the model's I/O geometry is read from `metadata.json` at startup.

> [!NOTE]
> To use a model you exported yourself with [AI Hub Models](https://github.com/qualcomm/ai-hub-models),
> pass the exported model path to `--model` in place of a model ID. The CLI places the exported
> assets into the app automatically:
>
> ```bash
> qai-hub-apps fetch depth_estimation_ubuntu_py --model <path/to/exported_model>
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

`test.sh` downloads a test image, streams it as a still video, and runs the app against it using the QAIRT runtime.

### List available cameras

```bash
./launch.sh -- --list-devices
```

### Run with a specific camera

```bash
./launch.sh -- --hexagon-version <HEX_VER> --video-device /dev/video0
```

> [!IMPORTANT]
> `--hexagon-version` must match your device's Hexagon DSP version unless it can be detected from the configured target device. To find the Hexagon version for your device, visit the [AI Hub device catalogue](https://workbench.aihub.qualcomm.com/devices/).

This serves the live stream (left) and the depth heatmap (right) on port 8080.
Open a browser and navigate to `http://<device-ip>:8080` to view them.
