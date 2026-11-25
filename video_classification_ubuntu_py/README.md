# Video Classification app

A Python app using OpenCV and LiteRT that performs on-device video
classification. It samples a clip of frames spanning the input video, runs the
ResNet-2Plus1D model on the Qualcomm NPU via the QAIRT TFLite delegate, and
prints the top predicted [Kinetics-400](https://arxiv.org/abs/1705.06950)
action classes.

## Requirements

- ARM64 Ubuntu 22.04+ or compatible Linux
- Docker

## Setup

### Option A: Using the CLI (Recommended)

Install the CLI and fetch the app with the model:

```bash
pip install qai-hub-apps
qai-hub-apps fetch video_classification_ubuntu_py --model resnet_2plus1d --output-dir ~
cd ~/video_classification_ubuntu_py
```

> [!NOTE]
> To use a model you exported yourself with [AI Hub Models](https://github.com/qualcomm/ai-hub-models),
> pass the exported model path to `--model` in place of a model ID. The CLI places the exported
> assets into the app automatically:
>
> ```bash
> qai-hub-apps fetch video_classification_ubuntu_py --model <path/to/exported_model>
> ```

### Option B: Cloning the Repo

If you cloned the release branch, the app directory is already self-contained — but **model weights are not included**. Download ResNet-2Plus1D from [AI Hub Models](https://aihub.qualcomm.com/iot/models), unzip the bundle, and copy its contents into `models/`, renaming the model file to `models/classifier.tflite`.

Compatible model assets must include their own `labels.txt` file.

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

### Run on a specific video

```bash
./launch.sh -- --hexagon-version <HEX_VER> --video <path/to/video.mp4> --output <path/to/annotated.jpg>
```

`--output` writes a frame from the middle of the video with the predictions
overlaid.

> [!IMPORTANT]
> You must provide `--hexagon-version` matching your device's Hexagon DSP version. For example, the [Dragonwing RB3 Gen 2](https://www.qualcomm.com/developer/hardware/rb3-gen-2-development-kit) uses Hexagon v68. To find the Hexagon version for your device, visit the [AI Hub device catalogue](https://workbench.aihub.qualcomm.com/devices/).

Use `--top-k` to change how many predictions are shown.

`--num-clips` sets how many clips are sampled from the video; each is
classified independently and their scores are averaged. The default of 1 is a
single fast inference. Raising it to 5 samples the video more densely, which is
more accurate on longer or more varied clips but takes proportionally longer.
