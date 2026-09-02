# 3D Object Detection app

A Python app using GStreamer, OpenCV, and LiteRT that detects objects in a live
camera stream and estimates a 3D bounding box for each one, drawing the projected
box on every frame.

The model bundle ships two networks that run back to back: a 2D detector locates
objects in the frame, and a second network takes each detected crop and predicts
the object's orientation and dimensions. Those predictions are solved against the
2D box to place an oriented 3D box in camera space.

## Requirements

- ARM64 Ubuntu 22.04+ or compatible Linux
- Docker

## Setup

### Option A: Using the CLI (Recommended)

Install the CLI and fetch the app with the model:

```bash
pip install qai-hub-apps
qai-hub-apps fetch object_detection_3d_ubuntu_py --model deepbox --output-dir ~
cd ~/object_detection_3d_ubuntu_py
```

> [!NOTE]
> To use a model you exported yourself with [AI Hub Models](https://github.com/qualcomm/ai-hub-models),
> pass the exported model path to `--model` in place of a model ID. The CLI places the exported
> assets into the app automatically:
>
> ```bash
> qai-hub-apps fetch object_detection_3d_ubuntu_py --model <path/to/exported_model>
> ```

### Option B: Cloning the Repo

If you cloned the release branch, the app directory is already self-contained — but **model weights are not included**. Download a compatible model from [AI Hub Models](https://aihub.qualcomm.com/iot/models), unzip the bundle and copy its contents into the `models/` directory before building:
- `models/<detector>.tflite` and `models/<3d_head>.tflite` — the model weights
- `models/metadata.json` — the model I/O metadata that ships in the bundle
- `models/labels.txt` — the class labels that ship in the bundle

> [!NOTE]
> The app resolves each model file, its input resolution, and its layout from
> `metadata.json`, matching the two networks by their output names rather than
> their file names.

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
docker build --build-arg BUILD_TYPE=runtime -t aiha-object-detection-3d .
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

### Camera field of view

Placing a box in 3D requires the camera's intrinsics. The app assumes a pinhole
camera with the principal point at the frame center and derives the focal length
from `--hfov`, the horizontal field of view in degrees. The default approximates
the camera the model was trained on; pass your camera's own value for accurate
distances.

```bash
./run_docker.sh --video-device /dev/video0 --hfov 70
```

Also pass `--video-source-width`/`--video-source-height` matching your source's
aspect ratio. The defaults are 4:3, so a 16:9 source is stretched, which skews the
box geometry.

### Recognized classes

The 3D estimate is a residual on top of per-class average dimensions measured on
driving data, so only classes with a known average are lifted to 3D: people,
cyclists, cars, trucks, buses, and trains. Other detections are ignored.
