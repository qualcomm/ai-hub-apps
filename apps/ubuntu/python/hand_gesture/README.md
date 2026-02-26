# Mediapipe Hand Gesture app

A Python app using GStreamer, OpenCV, and LiteRT that performs hand detection
and gesture analysis on a live camera stream.

## Device setup

Even if run via Docker, there is one requirement needed outside the Docker image:

```
sudo apt install libqnn1
```

## Run via Docker

The easiest way to run the app is via Docker. First, please make sure Docker is
installed by following [these
instructions](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository):

Build the docker image:

```
./build_docker.sh
```
Export a suitable model using `qai_hub_models` choosing a supported device of your choice, here we export for [Dragonwing RB3 Gen 2](https://www.qualcomm.com/developer/hardware/rb3-gen-2-development-kit)
```bash
python -m qai_hub_models.models.mediapipe_hand_gesture.export --target-runtime tflite --device "Dragonwing RB3 Gen 2 Vision Kit" --device-os 1.6 --precision w8a8 --skip-inferencing --skip-profiling
```
And move the model files to `<app_root>/models` directory as
```
models/PalmDetector.tflite
models/HandLandmarkDetector.tflite
models/CannedGestureClassifier.tflite
```

You can run the docker via the `run_docker.sh` command.
List available camera sources (or [Video4Linux2](https://en.wikipedia.org/wiki/Video4Linux) devices):

```
./run_docker.sh --list-devices
```

Pick a device, in this case

```
./run_docker.sh --video-device /dev/video0
```

This will serve the camera feed through port 8080, viewable from a web-browser
if you got to `http://<ip-address>:8080`, where `<ip-address>` is replaced by
the IP address of your IoT device.

For debugging, it can be useful to run the docker interactively:

```
./run_docker.sh --interactive
```

## Run directly

You can also run the app outside Docker. Please refer to `Dockerfile` for all
the necessary installation steps.
