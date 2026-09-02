# Mediapipe Hand Gesture app

A Python app using GStreamer, OpenCV, and LiteRT that performs hand detection
and gesture analysis on a live camera stream.

## Requirements

- ARM64 Ubuntu 22.04+ or compatible Linux
- Docker

## Setup

### Option A: Using the CLI (Recommended)

Install the CLI and fetch the app with the model:

```bash
pip install qai-hub-apps
qai-hub-apps fetch mediapipe_hand_gesture_ubuntu_py --model mediapipe_hand_gesture --output-dir ~
cd ~/mediapipe_hand_gesture_ubuntu_py
```

> [!NOTE]
> To use a model you exported yourself with [AI Hub Models](https://github.com/qualcomm/ai-hub-models),
> pass the exported model path to `--model` in place of a model ID. The CLI places the exported
> assets into the app automatically:
>
> ```bash
> qai-hub-apps fetch mediapipe_hand_gesture_ubuntu_py --model <path/to/exported_model>
> ```

### Option B: Cloning the Repo

If you cloned the release branch, the app directory is already self-contained — but **model weights are not included**. Download a compatible model from [AI Hub Models](https://aihub.qualcomm.com/mobile/models), unzip the bundle and copy the tflite model to the following paths before building:
- `models/palm_detector.tflite`
- `models/hand_landmark_detector.tflite`
- `models/canned_gesture_classifier.tflite`

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

If you are using a built-in camera on the Dragonwing RB3, also install `qcom-camera-server`:

```bash
sudo apt-get install qcom-camera-server
```

After installing, reboot the device.

### Using Docker
From the app directory, build our Docker image with all required runtime dependencies, including the supported QAIRT SDK.
```bash
docker build --build-arg BUILD_TYPE=runtime -t aiha-gesture .
```

## Run

```bash
./run_docker.sh --interactive
```
Inside the container:
```bash
bash test.sh
```

`test.sh` downloads a test video and runs the app against it using the QAIRT runtime.


### List available cameras

```bash
./run_docker.sh --list-devices
```

### Run with a specific camera

```bash
./run_docker.sh --hexagon-version <HEX_VER> --video-device /dev/video0
```

> [!IMPORTANT]
> You must provide `--hexagon-version` matching your device's Hexagon DSP version. For example, the [Dragonwing RB3 Gen 2](https://www.qualcomm.com/developer/hardware/rb3-gen-2-development-kit) uses Hexagon v68. To find the Hexagon version for your device, visit the [AI Hub device catalogue](https://workbench.aihub.qualcomm.com/devices/).

> [!NOTE]
> To use the integrated camera of a Dragonwing RB3, the `qtiqmmfsrc` GStreamer plugin must be used.
> `./run_docker.sh --hexagon-version v68 --video-gstreamer-source "qtiqmmfsrc name=camsrc camera=0"`.

This serves the camera feed on port 8080. Open a browser and navigate to
`http://<device-ip>:8080` to view the stream.
