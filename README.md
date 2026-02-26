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

1. Search for your desired OS & app in [this folder](apps), or in the tables at the bottom of this file.

2. The README of the selected app will contain build & installation instructions.

## _Android_ App Directory

| Task | Language | Inference API | Special Tags |
| -- | -- | -- | -- |
| [ChatApp](apps/android/ChatApp) | Java/C++ | Genie SDK | LLM, GenAI |
| [Image Classification](apps/android/ImageClassification) | Java | TensorFlow Lite |
| [Semantic Segmentation](apps/android/SemanticSegmentation) |  Java | TensorFlow Lite | OpenCV, Live Camera Feed |
| [Super Resolution](apps/android/SuperResolution) | Java | TensorFlow Lite |
| [WhisperKit (Speech to Text)](https://github.com/argmaxinc/WhisperKitAndroid) | Various | TensorFlow Lite |

## _Windows_ App Directory

| Task | Language | Inference API | Special Tags |
| -- | -- | -- | -- |
| [ChatApp](apps/windows/cpp/ChatApp) | C++ | Genie SDK | LLM, GenAI |
| [Image Classification](apps/windows/cpp/Classification) | C++ | ONNX | OpenCV |
| [Object Detection](apps/windows/cpp/ObjectDetection) | C++ | ONNX | OpenCV |
| [Super Resolution](apps/windows/cpp/SuperResolution) | C++ | ONNX | OpenCV |
| [Whisper Speech-to-Text](apps/windows/python/Whisper) | Python | ONNX |
| [Stable Diffusion Image Generation](apps/windows/python/StableDiffusion) | Python | ONNX |

## _Ubuntu_ App Directory

| Task | Language | Inference API | Special Tags |
| -- | -- | -- | -- |
| [Hand Gesture Recognition](apps/ubuntu/python/hand_gesture) | Python | TensorFlow Lite | GStreamer |

## _Tutorials_ Directory

| Tutorial | Topic |
| --- | --- |
| [LLM on-device deployment](tutorials/llm_on_genie) | Exporting and deploying Large Language Model (LLM) using Genie SDK |

## LICENSE

Qualcomm® AI Hub Apps is licensed under BSD-3. See the [LICENSE file](../LICENSE).
