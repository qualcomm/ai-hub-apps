[![Qualcomm® AI Hub Apps](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](https://aihub.qualcomm.com)

# Object Detection CLI application

This sample app detects objects in an image and writes the result with bounding boxes drawn.

The app aims to showcase best practices for using **ONNX Runtime** with the [QNN execution provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) for model inference on Windows on Snapdragon® devices, accelerated on the Snapdragon® NPU.

## Requirements

- Windows on Snapdragon X Elite
- [Visual Studio 22](https://visualstudio.microsoft.com/vs/) with the **Desktop development with C++** workload installed

## Setup

### Option A: Using the CLI (Recommended)

Install the CLI and fetch the app with the model:

```powershell
pip install qai-hub-apps
qai-hub-apps fetch object_detection_windows_cpp --model yolox --output-dir ~
cd ~\object_detection_windows_cpp
```

This downloads the app source and places the model asset in the correct location automatically.

> [!NOTE]
> To use a model you exported yourself with [AI Hub Models](https://github.com/qualcomm/ai-hub-models),
> pass the exported model path to `--model` in place of a model ID. The CLI places the exported
> assets into the app automatically:
>
> ```powershell
> qai-hub-apps fetch object_detection_windows_cpp --model <path\to\exported_model>
> ```

Browse the full set of compatible models on [AI Hub](https://aihub.qualcomm.com/models?domain=Computer+Vision&useCase=Object+Detection).

### Option B: Cloning the Repo

If you cloned the release branch, the app directory is already self-contained — but **model weights are not included**. Download the **float** ONNX variant of [YOLO-X](https://aihub.qualcomm.com/compute/models/yolox) from AI Hub, unzip the bundle and copy these files into place before building:
- Copy the ONNX model to `assets\models\detection.onnx`
- Copy the class labels file to `assets\models\labels.txt`

## Build

From the app directory (after either option above):

### Option A: Using Visual Studio
Open `ObjectDetection.sln` and build the `ARM64` configuration. The project restores its dependencies automatically:
- **NuGet** (ONNX Runtime QNN) restores during build. If not, right-click the solution → `Restore NuGet Packages`.
- **vcpkg** (OpenCV) is configured in [manifest mode](https://learn.microsoft.com/en-us/vcpkg/concepts/manifest-mode). If OpenCV headers are missing, run `vcpkg integrate install` in a Visual Studio terminal.

### Option B: Using Docker
Build our Docker image with all required dependencies, including the supported MS Build Tools, ONNX Runtime QNN, and OpenCV.
```powershell
docker build --build-arg BUILD_TYPE=build -t aiha-detection-win .
```
Build the EXE:
```powershell
docker run --name detection-container aiha-detection-win powershell -c '. ./install_build.ps1; & $env:MSBUILD_EXE ObjectDetection.sln /p:Configuration=Release /p:Platform=ARM64'

mkdir ./ARM64

docker cp detection-container:C:\app\ARM64 .
```

### Install & Run
```powershell
.\ARM64\Release\ObjectDetection.exe --model ".\assets\models\detection.onnx" --labels ".\assets\models\labels.txt" --image ".\assets\images\kitchen.jpg" --output_image detection_output.jpg
```

Run `--help` to learn more about all available options, including `--qnn_options` ([QNN EP options](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html#configuration-options)):

```powershell
.\ARM64\Release\ObjectDetection.exe --help
```

### Sample Input

![sample_input](assets/images/kitchen.jpg)

### Sample Output

![sample_output](assets/images/sample_outputs/kitchen_output.jpg)

## AI Model Requirements

### Model Runtime Formats
- ONNX (.onnx)

### I/O Specification

| INPUT | Description | Shape | Data Type |
| -- | -- | -- | -- |
| Image | An RGB image | [1, 3, Height, Width] | float32 |

| OUTPUT | Description | Shape | Data Type |
| -- | -- | -- | -- |
| Boxes | Bounding boxes (post-processed) | [N, 4] | float32 |
| Scores | Per-box confidence scores | [N] | float32 |
| Labels | Per-box class indices | [N] | int |

By default the model input resolution is 640x640 (input images are resized). Use `--model_input_ht` / `--model_input_wt` if your model expects different dimensions. The app expects a model with post-processing embedded (e.g. YOLO); other models require updating the output handling.

## FAQ

1. If you get a DLL error message upon launch (for instance that `opencv_core4d.dll` was not found), try Build -> Clean Solution and re-build. If this still happens, please go over the NuGet and vcpkg instructions again carefully.
2. How do I use a model with a different input shape than 640x640?
   - Use `--model_input_ht` / `--model_input_wt` to set the model input dimensions.
3. I have a model that does not have post-processing embedded into the model. Can I still use the app?
   - You will have to modify the app and add the necessary post-processing to accommodate that model.

## License

This app is released under the [BSD-3 License](../../LICENSE) found at the root of this repository.

All models from [AI Hub Models](https://github.com/qualcomm/ai-hub-models) are released under separate license(s). Refer to the [AI Hub Models repository](https://github.com/qualcomm/ai-hub-models) for details on each model.

The QNN SDK dependency is also released under a separate license. Please refer to the LICENSE file downloaded with the SDK for details.
