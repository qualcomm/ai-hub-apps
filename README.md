# Qualcomm® AI Hub Apps

The Qualcomm® AI Hub Apps are a collection of sample apps and tutorials to help deploy machine learning models on Qualcomm® devices.

Each app is designed to work with one or more models from [Qualcomm® AI Hub Models](https://aihub.qualcomm.com/).

With this repository, you can...
* Explore apps optimized for on-device deployment of various machine learning tasks.
* View open-source app recipes for running [Qualcomm® AI Hub Models](https://aihub.qualcomm.com/) on local devices.
* Find tutorials for end-to-end workflows.

## Overview

### Supported runtimes
* [TensorFlow Lite](https://www.tensorflow.org/lite)
* [ONNX](https://onnxruntime.ai/)
* Genie SDK (Generative AI runtime on top of [Qualcomm® AI Engine Direct SDK](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk))

### Supported Deployment Targets
* Android 11 Red Velvet Cake & Newer, API v30+
* Windows 11
* Ubuntu 24.04+

### Supported compute units
* CPU, GPU, NPU (includes [hexagon HTP](https://developer.qualcomm.com/hardware/qualcomm-innovators-development-kit/ai-resources-overview/ai-hardware-cores-accelerators))


### Chipsets supported for NPU Acceleration
* [Snapdragon X2 Elite](https://www.qualcomm.com/laptops/products/snapdragon-x2-elite)
* [Snapdragon X Elite](https://www.qualcomm.com/laptops/products/snapdragon-x-elite)
* [Snapdragon 8 Elite Gen 5](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-gen-5-mobile-platform)
* [Snapdragon 8 Elite](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-elite-mobile-platform)
* [Snapdragon 8 Gen 3](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-gen-3-mobile-platform)
* [Snapdragon 8 Gen 2](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-gen-2-mobile-platform)
* ... and all other [Snapdragon® chipsets supported by the QAIRT SDK](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html#supported-snapdragon-devices)

_Weight and activation type required for NPU Acceleration:_
* Floating Point: FP16 (All Snapdragon® chipsets with Hexagon® Architecture v69 or newer)
* Integer : INT8 or INT16 (All Snapdragon® chipsets)

__NOTE: Some of these apps will run without NPU acceleration on non-Snapdragon® chipsets.__

## Getting Started with Apps

1. Search for your desired OS & app in the tables below.

2. The README of the selected app will contain build & installation instructions.


> **Contributors:** See [CONTRIBUTING.md](https://github.com/qualcomm/ai-hub-apps/blob/main/CONTRIBUTING.md) for dev setup, repo architecture, and how to add new apps.
<!-- ANDROID_APPS -->
## _Android_ App Directory

| Task | Language | Inference API | Special Tags |
| -- | -- | -- | -- |
| [ChatApp](apps/chatapp_android) | Java/C++ | Genie SDK | LLM, GenAI |
| [Image Classification](apps/image_classification_android) | Java | TensorFlow Lite |
| [Object Detection](https://github.com/qualcomm/ai-hub-apps/tree/main/apps/object_detection_android)* | Java | TensorFlow Lite | OpenCV, Live Camera Feed |
| [Semantic Segmentation](apps/semantic_segmentation_android) |  Java | TensorFlow Lite | OpenCV, Live Camera Feed |
| [Super Resolution](apps/super_resolution_android) | Java | TensorFlow Lite |
| [WhisperKit (Speech to Text)](https://github.com/argmaxinc/WhisperKitAndroid)* | Various | TensorFlow Lite |
| [GenieX Chat Android](apps/geniex_chat_android) | Kotlin/Java | GenieX |
<!-- /ANDROID_APPS -->

<!-- WINDOWS_APPS -->
## _Windows_ App Directory

| Task | Language | Inference API | Special Tags |
| -- | -- | -- | -- |
| [ChatApp](apps/chatapp_windows_cpp) | C++ | Genie SDK | LLM, GenAI |
| [Image Classification](apps/image_classification_windows_cpp) | C++ | ONNX | OpenCV |
| [Object Detection](https://github.com/qualcomm/ai-hub-apps/tree/main/apps/object_detection_windows_cpp)* | C++ | ONNX | OpenCV |
| [Super Resolution](apps/super_resolution_windows_cpp) | C++ | ONNX | OpenCV |
| [Whisper Speech-to-Text](apps/whisper_windows_py) | Python | ONNX |
| [Stable Diffusion Image Generation](apps/stable_diffusion_windows_py) | Python | ONNX |
| [GenieX Chat Windows](https://github.com/qualcomm/geniex)* | Go | GenieX |
<!-- /WINDOWS_APPS -->

<!-- UBUNTU_APPS -->
## _Ubuntu_ App Directory

| Task | Language | Inference API | Special Tags |
| -- | -- | -- | -- |
| [Hand Gesture Recognition](apps/mediapipe_hand_gesture_ubuntu_py) | Python | TensorFlow Lite | GStreamer |
| [Posenet Pose Estimation](apps/posenet_ubuntu_py) | Python | TensorFlow Lite | GStreamer |
<!-- /UBUNTU_APPS -->

_\* Source available on GitHub; not included in the CLI release._

<!-- TUTORIALS -->
## _Tutorials_ Directory

| Tutorial | Topic |
| --- | --- |
| [LLM on-device deployment](https://github.com/qualcomm/ai-hub-apps/tree/main/tutorials/llm_on_genie) | Exporting and deploying Large Language Model (LLM) using Genie SDK |
| [LLM on-device deployment with GenieX](https://github.com/qualcomm/ai-hub-apps/tree/main/tutorials/geniex) | Running and exporting Large Language Models (LLMs) with the GenieX runtime |
<!-- /TUTORIALS -->

## LICENSE

Qualcomm® AI Hub Apps is licensed under BSD-3. See the [LICENSE file](LICENSE).
