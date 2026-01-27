[![Qualcomm® AI Hub Apps](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](https://aihub.qualcomm.com)

# Image Classification Sample App

This sample app classifies images. The top 3 predicted Imagenet classes are displayed.

The app aims to showcase best practices for using **TF Lite** for model inference on Android devices.

<p align="center" width="100%">
<img src="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-apps/android/ImageClassification/v1/app_screenshot.jpg" height="400" />
</p>

## Prerequisites
1. Clone this repository **with [Git-LFS](https://git-lfs.com) enabled.**
2. Download [Android Studio](https://developer.android.com/studio). **Version 2023.1.1 or newer** is required.
3. [Enable USB debugging](https://developer.android.com/studio/debug/dev-options) on your Android device.

## Build the APK
1. Download or export a [compatible model](#compatible-ai-hub-models) from [AI Hub Models](https://aihub.qualcomm.com/mobile/models).
2. Copy the `.tflite` file to `src/main/assets/classifier.tflite`
3. Open this folder in Android Studio, run gradle sync, and build the `ImageClassification` target.

## Supported Hardware (TF Lite Delegates)

By default, this app supports the following hardware:
* [Qualcomm Hexagon NPU -- via QNN](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
* [GPU -- via GPUv2](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/gpu)
* [CPU -- via XNNPack](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/xnnpack/README.md)

Comments have been left in [TFLiteHelpers.java](../tflite_helpers/TFLiteHelpers.java) and [AIHubDefaults.java](../tflite_helpers/AIHubDefaults.java) to guide you on how to add support for additional TF Lite delegates that could target other hardware.


## AI Model Requirements

### Model Runtime Formats
- TensorFlow Lite (.tflite)

### I/O Specification

| INPUT | Description | Shape | Data Type
| -- | -- | -- | --
| Image | An RGB image | [1, Height, Width, 3] | float32 or uint8 (zero point of 0, scale of 1 / 255)

| OUTPUT | Description | Shape | Data Type
| -- | -- | -- | --
| Classes | **Imagenet** Classes | [ 1, 1000 ] | float32, uint8, or int8 (any quantization params)


## Compatible [AI Hub Models](https://aihub.qualcomm.com/mobile/models)

The below is a non-exhaustive list of [AI Hub Models](https://aihub.qualcomm.com/mobile/models) that should be compatible.

**Not every model has been individually tested with this app**. Please file an issue or reach out on [Slack](http://aihub.qualcomm.com/community/slack) if you find a model in this list with app compatibility issues.

Please download the TFLite asset. Variants `float` and `w8a8` are both supported by the app.

- [ConvNext-Tiny](https://aihub.qualcomm.com/mobile/models/convnext_tiny)
- [DenseNet-121](https://aihub.qualcomm.com/mobile/models/densenet121)
- [EfficientNet-B0](https://aihub.qualcomm.com/mobile/models/efficientnet_b0)
- [GoogLeNet](https://aihub.qualcomm.com/mobile/models/googlenet)
- [Inception-v3](https://aihub.qualcomm.com/mobile/models/inception_v3)
- [MNASNet05](https://aihub.qualcomm.com/mobile/models/mnasnet05)
- [MobileNet-v2](https://aihub.qualcomm.com/mobile/models/mobilenet_v2)
- [MobileNet-v3-Large](https://aihub.qualcomm.com/mobile/models/mobilenet_v3_large)
- [MobileNet-v3-Small](https://aihub.qualcomm.com/mobile/models/mobilenet_v3_small)
- [RegNet](https://aihub.qualcomm.com/mobile/models/regnet)
- [ResNet101](https://aihub.qualcomm.com/mobile/models/resnet101)
- [ResNet18](https://aihub.qualcomm.com/mobile/models/resnet18)
- [ResNet50](https://aihub.qualcomm.com/mobile/models/resnet50)
- [ResNeXt101](https://aihub.qualcomm.com/mobile/models/resnext101)
- [ResNeXt50](https://aihub.qualcomm.com/mobile/models/resnext50)
- [ShuffleNet-v2](https://aihub.qualcomm.com/mobile/models/shufflenet_v2)
- [SqueezeNet-1_1](https://aihub.qualcomm.com/mobile/models/squeezenet1_1)
- [Swin-Base](https://aihub.qualcomm.com/mobile/models/swin_base)
- [Swin-Small](https://aihub.qualcomm.com/mobile/models/swin_small)
- [Swin-Tiny](https://aihub.qualcomm.com/mobile/models/swin_tiny)
- [VIT](https://aihub.qualcomm.com/mobile/models/vit)
- [WideResNet50](https://aihub.qualcomm.com/mobile/models/wideresnet50)

## Replicating an AI Hub Profile / Inference Job

Each AI Hub profile or inference job, once completed, will contain a `Runtime Configuration` section.

Modify [TFLiteHelpers.java](../tflite_helpers/TFLiteHelpers.java) according to the runtime configuration applied to the job. **Comment stubs are included** to help guide you (search for `TO REPLICATE AN AI HUB JOB...`)

Note that if your job uses delegates other than QNN NPU, GPUv2, and TFLite, then you will also need to add support for those delegates to the app.

## Technologies Used by this App

- [Android SDK](https://developer.android.com/studio)
- [TensorFlow Lite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite)
- [TF Lite Support Library](https://github.com/tensorflow/tflite-support)
- [QNN SDK (TF Lite Delegate)](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
- [GPUv2 Delegate](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/gpu)
- [XNNPack Delegate ](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/xnnpack/README.md)

## License

This app is released under the [BSD-3 License](../../../LICENSE) found at the root of this repository.

All models from [AI Hub Models](https://github.com/quic/ai-hub-models) are released under separate license(s). Refer to the [AI Hub Models repository](https://github.com/quic/ai-hub-models) for details on each model.

The QNN SDK dependency is also released under a separate license. Please refer to the LICENSE file downloaded with the SDK for details.
