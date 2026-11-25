# OCR Text Recognition app

A Python EasyOCR app using GStreamer, OpenCV, and LiteRT. It streams a live
camera feed (or a still image/video file) to a web browser, showing the live
feed and the most recent OCR result side by side; pressing 's' in the browser
captures the current frame and runs detection + recognition on it, drawing
the recognized text boxes onto the captured frame in the "Detected" panel
until the next capture.

The model is **EasyOCR** (AI Hub slug `easyocr`), a two-stage pipeline: a CRAFT
**text detector** locates text regions and a CTC **text recognizer** reads each
region.

## Requirements

- ARM64 Ubuntu 22.04+ or compatible Linux
- Docker

## Setup

### Option A: Using the CLI (Recommended)

```bash
pip install qai-hub-apps
qai-hub-apps fetch easyocr_ubuntu_py --model easyocr --output-dir ~
cd ~/easyocr_ubuntu_py
```

`easyocr` is a collection model, so the fetch downloads **two** TFLite files (a
detector and a recognizer). Confirm the produced filenames match the paths in
`info.yaml` / `main.py`:
- `models/easyocr_detector.tflite`
- `models/easyocr_recognizer.tflite`

### Option B: Cloning the Repo

Model weights are not included. Download the `easyocr` w8a8 TFLite bundle from
[AI Hub Models](https://aihub.qualcomm.com/iot/models), unzip, and copy the two
`.tflite` files to the `models/` paths listed above.

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

`test.sh` downloads a test image and runs the app against it using the QAIRT
runtime.

### Run against a still image / document

```bash
./launch.sh -- --hexagon-version <HEX_VER> \
    --video-gstreamer-source "filesrc location=doc.png ! decodebin ! imagefreeze"
```

### Run with a camera

```bash
./launch.sh -- --hexagon-version <HEX_VER> --video-device /dev/video0
```

> [!IMPORTANT]
> Provide `--hexagon-version` matching your device's Hexagon DSP version. Find it
> in the [AI Hub device catalogue](https://workbench.aihub.qualcomm.com/devices/).

Open `http://<device-ip>:8080` in a browser to view the live stream, shown
side by side with a "Detected" panel (empty until the first capture). Press
**'s'** to capture the current frame and run OCR on it — the detected
boxes/text appear in the "Detected" panel and stay there until you press
's' again.

> [!WARNING]
> The web UI has no authentication and streams the live scanned document/text
> content, which may contain PII (IDs, invoices, medical forms). Only run this
> app on a trusted network; the container publishes port 8080 on all interfaces,
> not just localhost. Restrict access at the network/firewall level, or tunnel
> it instead (e.g. `ssh -L 8080:localhost:8080`) if you need to view it from
> another machine.

## Performance per capture

The video stream itself always runs at full camera/source framerate — OCR only
runs once per capture (pressing 's' in the browser), so it never affects
streaming smoothness. Within a single capture, the two stages have very
different costs, and the recognizer runs **once per detected text line**:

| Stage | Runs | w8a8 TFLite latency (Dragonwing IQ-9075 EVK) |
|---|---|---|
| Detector (CRAFT) | once per capture | ~10 ms |
| Recognizer (CTC) | once per text line | ~30–96 ms per line (device-dependent) |

So total capture latency ≈ `detector + (num_text_lines × recognizer)` plus
CPU-side post-processing — from under 100 ms for sparse text to a second or
more for a text-dense document, since the recognizer's input is fixed batch
size 1 and lines are processed sequentially.

**Delegate note:** the detector runs on the Hexagon HTP via the QNN TFLite
delegate, but EasyOCR's recognizer graph has dynamic-shaped tensors (its
LSTM/CTC path) that the static-shape HTP delegate rejects, so `main.py` falls
back to running the recognizer **on CPU**. This makes recognition slower than
the HTP figures above. Exporting the recognizer with an unrolled LSTM (static
shapes) would let it run on the HTP — a possible future optimization.
