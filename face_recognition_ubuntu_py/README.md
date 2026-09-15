# Face Recognition app

A Python app using GStreamer, OpenCV, and LiteRT that performs on-device face
recognition with any face-embedding model. You enroll identities from a
directory of face images (the "gallery"); the app then reads a live camera
stream, detects each face, embeds it on-device, and draws a bounding box labeled
with the matched identity — or `Unknown` when no gallery identity is close
enough. Output frames are served over a minimal web interface for
headless/embedded use.

Face **detection** uses OpenCV's bundled Haar cascade on the CPU; face
**recognition** (the face embedding) runs on the NPU via the QAIRT runtime.

## Supported models

The app is model-agnostic — it works with any TFLite face-embedding model that
takes a cropped face image and emits a fixed-length embedding vector. The
default (and currently supported) model is:

- **[CavaFace](https://aihub.qualcomm.com/iot/models)** — 112x112 RGB input,
  512-dimensional L2-normalized embedding.

As more face-embedding models are onboarded they will be listed here; select a
model at fetch time with `--model <model-id>` (see [Setup](#setup)).

## Requirements

- ARM64 Ubuntu 24.04+ or compatible Linux
- Docker

## Setup

### Option A: Using the CLI (Recommended)

Install the CLI and fetch the app with the model:

```bash
pip install qai-hub-apps
qai-hub-apps fetch face_recognition_ubuntu_py --model cavaface --output-dir ~
cd ~/face_recognition_ubuntu_py
```

> [!NOTE]
> To use a model you exported yourself with [AI Hub Models](https://github.com/qualcomm/ai-hub-models),
> pass the exported model path to `--model` in place of a model ID. The CLI places the exported
> assets into the app automatically:
>
> ```bash
> qai-hub-apps fetch face_recognition_ubuntu_py --model <path/to/exported_model>
> ```

### Option B: Cloning the Repo

If you cloned the release branch, the app directory is already self-contained — but **model weights are not included**. Download a compatible model from [AI Hub Models](https://aihub.qualcomm.com/iot/models), unzip the bundle and copy the tflite model to the following path before building:
- `models/cavaface.tflite`

## Prepare a gallery

Create a directory of the people you want the app to recognize. Two layouts are
supported and may be mixed:

```
gallery/
├── Alice/          # one folder per identity; all images are averaged
│   ├── 1.jpg
│   └── 2.jpg
├── Bob/
│   └── portrait.jpg
└── Carol.jpg       # a single top-level image; the filename is the identity
```

Enrolling several photos per identity (the folder layout) generally improves
accuracy. Each enrollment image should contain a single, roughly front-facing
face; if the app detects no face in an image it falls back to embedding the whole
image (so tightly pre-cropped faces still enroll).

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
docker build --build-arg BUILD_TYPE=runtime -t aiha-face-recognition .
```

## Run

```bash
./run_docker.sh --interactive
```
Inside the container:
```bash
bash test.sh
```

`test.sh` downloads two public reference photos of the same person, enrolls the
first as identity `reference_person`, then recognizes the second against that
gallery using the QAIRT runtime, expecting a `reference_person` match.

### List available cameras

```bash
./run_docker.sh --list-devices
```

### Run with a live camera

```bash
./run_docker.sh --hexagon-version <HEX_VER> --gallery-dir /path/to/gallery --video-device /dev/video0
```

`run_docker.sh` resolves the gallery path and bind-mounts it into the container,
so the gallery can live anywhere on the host. This serves the annotated camera
feed on port 8080 — open a browser and navigate to `http://<device-ip>:8080` to
view the stream.

> [!IMPORTANT]
> You must provide `--hexagon-version` matching your device's Hexagon DSP version. For example, the [Dragonwing IQ-9075 EVK](https://www.qualcomm.com/developer/hardware/iq-9075-evk) uses Hexagon v73. To find the Hexagon version for your device, visit the [AI Hub device catalogue](https://workbench.aihub.qualcomm.com/devices/).

> [!NOTE]
> To use the integrated camera of a Dragonwing RB3, the `qtiqmmfsrc` GStreamer plugin must be used.
> `./run_docker.sh --hexagon-version v68 --gallery-dir /path/to/gallery --video-gstreamer-source "qtiqmmfsrc name=camsrc camera=0"`.

### Recognize a single image (no camera)

To identify the faces in one image against the gallery — useful for testing or
on machines without a camera — use `--image`:

```bash
./run_docker.sh --hexagon-version <HEX_VER> --gallery-dir /path/to/gallery --image /path/to/photo.jpg --output /path/to/annotated.jpg
```

The app prints one `name: score` line per detected face and, with `--output`,
writes a copy of the image with the labeled bounding boxes drawn on it.

### Tuning

- `--recognition-threshold` (default `0.5`): a detected face is labeled with the
  best-matching identity only if its cosine similarity is at least this value;
  otherwise it is labeled `Unknown`. Because the Haar crop is not landmark-aligned,
  you may need to tune this on your device and gallery.

> [!NOTE]
> The Haar cascade detects roughly front-facing faces only; strongly rotated,
> profile, or very small faces may be missed. CavaFace expects a reasonably tight
> face crop; the app resizes and pads each detected crop to the model's 112x112
> input while preserving aspect ratio.
