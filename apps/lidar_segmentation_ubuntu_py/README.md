# LiDAR Semantic Segmentation app

A Python app using OpenCV and LiteRT that assigns a semantic class to every
point of a LiDAR scan, and renders the result as a color-coded bird's-eye view
and range image — served to a browser, or written to an image file.

The scan is projected into a 64-row spherical range image — one row per laser
beam, columns spanning 360° of azimuth — carrying range, x, y, z and remission
as its five channels. The model classifies that image, and the per-pixel
prediction is read back onto the original points.

## Requirements

- ARM64 Ubuntu 22.04+ or compatible Linux
- Docker

## Setup

### Option A: Using the CLI (Recommended)

Install the CLI and fetch the app with the model:

```bash
pip install qai-hub-apps
qai-hub-apps fetch lidar_segmentation_ubuntu_py --model salsanext --output-dir ~
cd ~/lidar_segmentation_ubuntu_py
```

> [!NOTE]
> To use a model you exported yourself with [AI Hub Models](https://github.com/qualcomm/ai-hub-models),
> pass the exported model path to `--model` in place of a model ID. The CLI places the exported
> assets into the app automatically:
>
> ```bash
> qai-hub-apps fetch lidar_segmentation_ubuntu_py --model <path/to/exported_model>
> ```

### Option B: Cloning the Repo

If you cloned the release branch, the app directory is already self-contained — but **model weights are not included**. Download a compatible model from [AI Hub Models](https://aihub.qualcomm.com/iot/models), unzip the bundle and copy its contents into the `models/` directory before building:
- `models/salsanext.tflite` — the model weights
- `models/metadata.json` — the model I/O metadata that ships in the bundle

> [!NOTE]
> The app resolves the model file name, its projection size and its channel
> layout from `metadata.json`, so a renamed or re-exported asset still runs.

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

## Run

`./launch.sh` builds the app's Docker image — with all required runtime
dependencies, including the supported QAIRT SDK — on first use, installs the
app's dependencies inside the container, then runs the app. App arguments go
after `--`; add `--no-docker` to run natively on the host instead.

Start with the app's self-test:

```bash
./launch.sh --test -- --hexagon-version <HEX_VER>
```

`test.sh` downloads a sample LiDAR scan, segments it with the QAIRT runtime and
writes the result to `000000.png`.

### Run against your own scan

```bash
./launch.sh -- --hexagon-version <HEX_VER> --lidar-source /path/to/scan.bin
```

> [!IMPORTANT]
> You must provide `--hexagon-version` matching your device's Hexagon DSP version. For example, the [Dragonwing RB3 Gen 2](https://www.qualcomm.com/developer/hardware/rb3-gen-2-development-kit) uses Hexagon v68. To find the Hexagon version for your device, visit the [AI Hub device catalogue](https://workbench.aihub.qualcomm.com/devices/).

`--lidar-source` is one KITTI-format Velodyne scan: a flat `float32` array of
`(x, y, z, remission)` records, with x forward, y left and z up in meters,
relative to the sensor.

This serves the segmented output on port 8080. Open a browser and navigate to
`http://<device-ip>:8080` to view it, then press Enter in the terminal to
exit.

### Writing the output to a file

`--output` writes the rendered view to that image path instead of serving it,
and the app exits without waiting.

```bash
./launch.sh -- --hexagon-version <HEX_VER> --output segmented.png
```

Pass `--profile` to print the per-stage latency breakdown.

### Sensor geometry

The projection assumes the 64-beam sensor the model was trained on, spanning
+3° to -25° of elevation. A scan from a sensor with a different vertical field
of view projects into the wrong rows and segments poorly; adjust `FOV_UP_DEG` and
`FOV_DOWN_DEG` in `utils/constants.py` to match your sensor.

### Recognized classes

The model predicts the 19 SemanticKITTI classes drawn in the on-screen key —
road, sidewalk, parking, other-ground, building, fence, vegetation, trunk,
terrain, pole, traffic-sign, car, truck, bicycle, motorcycle, other-vehicle,
person, bicyclist and motorcyclist — plus an ignored class for points it cannot
label, which are left unpainted.
