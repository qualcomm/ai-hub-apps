[![Qualcomm® AI Hub Apps](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](https://aihub.qualcomm.com)

# Object Detection Sample App

This sample app performs object detection on live camera input.

The app aims to showcase an example of combining streaming camera, TFLite, and OpenCV.

## Requirements

- Android device with [USB debugging enabled](https://developer.android.com/studio/debug/dev-options) (Android 14+)
- [Android Studio](https://developer.android.com/studio) **2023.1.1 or newer**

## Setup

### Option A: Using the CLI (Recommended)

Install the CLI and fetch the app with the model:

```bash
pip install qai-hub-apps
qai-hub-apps fetch object_detection_android --model yolox --output-dir ~
cd ~/object_detection_android
```

This downloads the app source and places the model asset in the correct location automatically.

> [!NOTE]
> To use a model you exported yourself with [AI Hub Models](https://github.com/qualcomm/ai-hub-models),
> pass the exported model path to `--model` in place of a model ID. The CLI places the exported
> assets into the app automatically:
>
> ```bash
> qai-hub-apps fetch object_detection_android --model <path/to/exported_model>
> ```

### Option B: Cloning the Repo

If you cloned the release branch **with [Git-LFS](https://git-lfs.com) enabled**, the app directory is already self-contained — but **model weights are not included**. Download or export a [compatible model](#compatible-ai-hub-models) from [AI Hub Models](https://aihub.qualcomm.com/mobile/models), then copy the files into place before building:
- Copy the `.tflite` file to `src/main/assets/detector.tflite`
- Copy the labels file to `src/main/assets/labels.txt`

## Build

From the app directory (after either option above):

### Option A: Using Android Studio
To build APK using Android studio:
- Open this folder in Android Studio
- Run gradle sync
- Build and run the `app` target
    - Click on `Build` -> `Generate App Bundles or APKs` -> `Generate APKs`
    - Click on `Run` -> `Run 'app'`

The APKs will be at:
- `build/outputs/apk/debug/app-debug.apk`
- `build/outputs/apk/androidTest/debug/app-debug-androidTest.apk`

### Option B: Using Docker

> [!IMPORTANT]
> **Building on an ARM host?**
> The Android build tools (AAPT2, NDK clang) are x86_64-only binaries. To run them under emulation, register the QEMU x86_64 handler on the **host** before building (run once per boot):
> ```bash
> sudo apt-get update && sudo apt-get install -y qemu-user-static binfmt-support
> sudo update-binfmts --enable qemu-x86_64
> ```

Build our Docker image with all required dependencies, including the supported Android and QAIRT SDKs.
```bash
docker build --build-arg BUILD_TYPE=build -t aiha-detection .
```
Build the APK:
```bash
docker run --name detection-container aiha-detection bash -c "source /app/scripts/android_utils.sh && cd /app && gradle assembleDebug assembleAndroidTest"

mkdir ./build

docker cp detection-container:/app/build/outputs ./build/outputs
```

#### Install & Run

Connect your Android device via USB, then:

```bash
adb install build/outputs/apk/debug/app-debug.apk
```

Launch the app from your device's app drawer.

## Supported Hardware (TF Lite Delegates)

By default, this app supports the following hardware:
* [Qualcomm Hexagon NPU -- via QNN](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
* [GPU -- via GPUv2](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/gpu)
* [CPU -- via XNNPack](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/xnnpack/README.md)

Comments have been left in [TFLiteHelpers.java](src/main/java/com/quicinc/tflite/TFLiteHelpers.java) and [AIHubDefaults.java](src/main/java/com/quicinc/tflite/AIHubDefaults.java) to guide you on how to add support for additional TF Lite delegates that could target other hardware.

## AI Model Requirements

### Model Runtime Formats
- TensorFlow Lite (.tflite)

### I/O Specification

| INPUT | Description | Shape | Data Type
| -- | -- | -- | --
| `image` | An RGB image | [1, Height, Width, 3] | float32 input expecting inputs normalized to [0, 1]

| OUTPUT | Description | Shape | Data Type
| -- | -- | -- | --
| `boxes` | Bounding Boxes | [1, Anchors, 4] | float32 boxes (x0, y0, x1, y1) in pixel space
| `scores` | Class Scores | [1, Anchors] | float32 class logit predictions
| `class_idx` | Class Index | [1, Anchors] | uint8 or int32 class index

The app is developed to work best with a Width/Height ratio of 1.

## Compatible [AI Hub Models](https://aihub.qualcomm.com/mobile/models)

The app is currently compatible with the TFLite `w8a8` variant of these models:

  - [Yolo-v3](https://aihub.qualcomm.com/mobile/models/yolov3)
  - [Yolo-v5](https://aihub.qualcomm.com/mobile/models/yolov5)
  - [Yolo-v6](https://aihub.qualcomm.com/mobile/models/yolov6)
  - [Yolo-v7](https://aihub.qualcomm.com/mobile/models/yolov7)
  - [YOLOv8-Detection](https://aihub.qualcomm.com/mobile/models/yolov8_det)
  - [YOLOv10-Detection](https://aihub.qualcomm.com/mobile/models/yolov10_det)
  - [YOLOv11-Detection](https://aihub.qualcomm.com/mobile/models/yolov11_det)
  - [Yolo-X](https://aihub.qualcomm.com/mobile/models/yolox)
  - [DETR-ResNet50](https://aihub.qualcomm.com/mobile/models/detr_resnet50)
  - [DETR-ResNet101](https://aihub.qualcomm.com/mobile/models/detr_resnet101)

## Replicating an AI Hub Profile / Inference Job

Each AI Hub profile or inference job, once completed, will contain a `Runtime Configuration` section.

Modify [TFLiteHelpers.java](src/main/java/com/qualcomm/tflite/TFLiteHelpers.java) according to the runtime configuration applied to the job. **Comment stubs are included** to help guide you (search for `TO REPLICATE AN AI HUB JOB...`)

Note that if your job uses delegates other than QNN NPU, GPUv2, and TFLite, then you'll also need to add support for those delegates to the app.

## Technologies Used by this App

- [Android SDK](https://developer.android.com/studio)
- [TensorFlow Lite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite)
- [OpenCV](https://opencv.org)
- [QNN SDK (TF Lite Delegate)](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
- [GPUv2 Delegate](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/gpu)
- [XNNPack Delegate](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/xnnpack/README.md)

## License

This app is released under the [BSD-3 License](../../LICENSE) found at the root of this repository.

All models from [AI Hub Models](https://github.com/qualcomm/ai-hub-models) are released under separate license(s). Refer to the [AI Hub Models repository](https://github.com/qualcomm/ai-hub-models) for details on each model.

The QNN SDK dependency is also released under a separate license. Please refer to the LICENSE file downloaded with the SDK for details.
