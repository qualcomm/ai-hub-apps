[![Qualcomm® AI Hub Apps](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](https://aihub.qualcomm.com)

# Super Resolution CLI application

This sample app upscales an image and writes the enhanced result.

The app aims to showcase best practices for using **ONNX Runtime** with the [QNN execution provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) for model inference on Windows on Snapdragon® devices, accelerated on the Snapdragon® NPU.

## Requirements

- Windows on Snapdragon X Elite
- A Windows host machine (x86-64 or ARM) with Docker installed

## Setup

### Option A: Using the CLI (Recommended)

Install the CLI and fetch the app with the model:

```powershell
pip install qai-hub-apps
qai-hub-apps fetch super_resolution_windows_cpp --model xlsr --chipset qualcomm-snapdragon-x-elite --output-dir ~
cd ~\super_resolution_windows_cpp
```

This downloads the app source and places the model asset in the correct location automatically.

> [!NOTE]
> To use a model you exported yourself with [AI Hub Models](https://github.com/qualcomm/ai-hub-models),
> pass the exported model path to `--model` in place of a model ID. The CLI places the exported
> assets into the app automatically:
>
> ```powershell
> qai-hub-apps fetch super_resolution_windows_cpp --model <path\to\exported_model>
> ```

Browse the full set of compatible models on [AI Hub](https://aihub.qualcomm.com/models?domain=Computer+Vision&useCase=Super+Resolution).

### Option B: Cloning the Repo

If you cloned the release branch, the app directory is already self-contained — but **model weights are not included**. Download a compatible **float** ONNX model from [AI Hub](https://aihub.qualcomm.com/models?domain=Computer+Vision&useCase=Super+Resolution), unzip the bundle and copy the ONNX model to `assets\models\super_resolution.onnx` before building.

## Build

From the app directory (after either option above):

### Option A: Using Visual Studio
Open `SuperResolution.sln` and build the `ARM64` configuration. The project restores its dependencies automatically:
- **NuGet** (ONNX Runtime QNN) restores during build. If not, right-click the solution → `Restore NuGet Packages`.
- **vcpkg** (OpenCV) is configured in [manifest mode](https://learn.microsoft.com/en-us/vcpkg/concepts/manifest-mode). If OpenCV headers are missing, run `vcpkg integrate install` in a Visual Studio terminal.

### Option B: Using Docker
Build our Docker image with all required dependencies, including the supported MS Build Tools, ONNX Runtime QNN, and OpenCV.
```powershell
docker build --build-arg BUILD_TYPE=build -t aiha-superresolution-win .
```
Build the EXE:
```powershell
docker run --name superresolution-container aiha-superresolution-win powershell -c '. ./install_build.ps1; & $env:MSBUILD_EXE SuperResolution.sln /p:Configuration=Release /p:Platform=ARM64'

mkdir ./ARM64

docker cp superresolution-container:C:\app\ARM64 .
```

### Install & Run
```powershell
.\ARM64\Release\SuperResolution.exe --model ".\assets\models\super_resolution.onnx" --image ".\assets\images\Doll.jpg" --output_image ".\upscaled.png"
```

Run `--help` to learn more about all available options, including `--qnn_options` ([QNN EP options](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html#configuration-options)):

```powershell
.\ARM64\Release\SuperResolution.exe --help
```

### Sample Input

![sample_input](assets/images/Doll.jpg)

### Sample Output

![sample_output](assets/images/UpscaledImage.png)

## AI Model Requirements

### Model Runtime Formats
- ONNX (.onnx)

### I/O Specification

| INPUT | Description | Shape | Data Type |
| -- | -- | -- | -- |
| Image | An RGB image | [1, 3, Height, Width] | float32 |

| OUTPUT | Description | Shape | Data Type |
| -- | -- | -- | -- |
| Image | The upscaled RGB image | [1, 3, Height×scale, Width×scale] | float32 |

By default the model input resolution is 128x128 (input images are resized). Use `--model_scale` to set the upscaling factor based on your model.

## FAQ

1. If you get a DLL error message upon launch (for instance that `opencv_core4d.dll` was not found), try Build -> Clean Solution and re-build. If this still happens, please go over the NuGet and vcpkg instructions again carefully.
2. How do I use a model with a different scaling factor?
   - Use `--model_scale` to change the scaling based on your model.
3. I have a model that has different post-processing. Can I still use the app?
   - You will have to modify the app and add the necessary post-processing to accommodate that model.

## License

This app is released under the [BSD-3 License](../../LICENSE) found at the root of this repository.

All models from [AI Hub Models](https://github.com/qualcomm/ai-hub-models) are released under separate license(s). Refer to the [AI Hub Models repository](https://github.com/qualcomm/ai-hub-models) for details on each model.

The QNN SDK dependency is also released under a separate license. Please refer to the LICENSE file downloaded with the SDK for details.
