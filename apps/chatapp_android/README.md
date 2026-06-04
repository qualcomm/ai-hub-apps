[![Qualcomm® AI Hub Apps](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](https://aihub.qualcomm.com)

# Sample Chat App

Chat application for Android on Snapdragon® with LLMs from [Qualcomm® AI Hub Models](https://github.com/qualcomm/ai-hub-models/tree/main/src/qai_hub_models/models) using Genie SDK.

The app demonstrates how to use the Genie C++ APIs from [QAIRT SDK](https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK) to run and accelerate LLMs using the Snapdragon® Neural Processing Unit (NPU).

## Current limitations on running ChatApp

> [!IMPORTANT]
> This app requires a **mobile** device with Android 15+. It may work on select Android 14 devices. It **does not** work on Auto devices.

The Genie SDK requires newer meta-build to run LLMs on-device. Depending on which meta-build is picked by your phone vendor, this feature may or may not work (especially if you are on Android 14 or earlier, which in many cases lacks the necessary meta-build).

We recommend using a device from [QDC](https://qdc.qualcomm.com/) for rest of this demo to run models on-device.
Android devices on QDC have newer meta-build and can run this demo on Android 14+.

## Demo

https://github.com/user-attachments/assets/7b23c632-cc4e-48ed-b1df-ea98ec0f51b7

## Requirements

- Android device with [USB debugging enabled](https://developer.android.com/studio/debug/dev-options) (Android 15+, Snapdragon® Gen 3 or Newer)
- A Windows/MacOS/Linux host machine with Docker installed

## Setup

### Option A: Using the CLI (Recommended)

Install the CLI and fetch the app with the model:

```bash
pip install qai-hub-apps
qai-hub-apps fetch chatapp_android --model qwen3_4b_instruct_2507 --chipset qualcomm-snapdragon-8-elite --output-dir ~
cd ~/chatapp_android
```

This downloads the app source and places the model binaries and tokenizer in the correct location automatically.

### Option B: Cloning the Repo

If you cloned the release branch, the app directory is already self-contained — but **model weights are not included**. Obtain the LLM binaries from [AI Hub Models](https://aihub.qualcomm.com/models?domain=Generative+AI&useCase=Text+Generation) and unzip the bundle to `src/main/assets/models/llm/` before building. See [Exporting an LLM](#exporting-an-llm) for more details.

## Build

From the app directory (after either option above):

### Option A: Using Android Studio
In order to build using Android Studio, you need to first install a supported QAIRT SDK and set the environment variable `QAIRT_PATH` to its root directory.
You can use our provided helper utility to set up a supported QAIRT SDK.

Linux/MacOS users:
```bash
source ./scripts/qairt_utils.sh
install_qairt
echo "QAIRT_PATH=$QAIRT_PATH"
```

Windows users:
```powershell
. ".\scripts\qairt_utils.ps1"

Install-Qairt
echo "QAIRT_PATH=$env:QAIRT_PATH"
```
Replace `"<SET QAIRT SDK PATH>"` in `build.gradle` with `QAIRT_PATH` obtained from above.  

Build APK
- Open this folder in Android Studio
- Run gradle sync
- Build and run the `app` target
    - Click on `Build` -> `Generate App Bundles or APKs` -> `Generate APKs`
    - Click on `Run` -> `Run 'app'`

The APKs will be at:
- `build/outputs/apk/debug/app-debug.apk`
- `build/outputs/apk/androidTest/debug/app-debug-androidTest.apk`

### Option B: Using Docker
Build our Docker image with all required dependencies, including the supported Android and QAIRT SDKs.
```bash
docker build --build-arg BUILD_TYPE=build -t aiha-chatapp .
```
Build the APK:
```bash
docker run --name chatapp-container aiha-chatapp bash -c "source /app/scripts/android_utils.sh && cd /app && gradle assembleDebug assembleAndroidTest"

mkdir ./build

docker cp chatapp-container:/app/build/outputs ./build/outputs
```

#### Install & Run

Connect your Android device via USB, then:

```bash
adb install build/outputs/apk/debug/app-debug.apk
```

Launch the app from your device's app drawer.

## Exporting an LLM

1. Get QNN context binaries for the LLM of your choice from Qualcomm AI Hub. There are two ways to get these assets:

    - Run export script to get context binaries for Llama variants. We will export these models with context length 4096 by default. You can add the argument --context-length with your desired context length value while exporting the model for modifying (recommended to use lower or equal to 4096). Make sure the size option in the genie config matches your model's context length.

    - Download directly from our website. Make sure to select the correct device when downloading the context binaries.

    - Read more about [exporting LLMs via AI Hub here](https://github.com/qualcomm/ai-hub-apps/tree/main/tutorials/llm_on_genie#1-generate-genie-compatible-qnn-binaries-from-ai-hub)
        - You'll have to replace model name from the above tutorial with `llama_v3_2_3b_instruct` or the model id of your choice and reduce context length for this demo when exporting.

    - The following command exports Llama 3.2 3B model with context length 4096:

    ```bash
    python -m qai_hub_models.models.llama_v3_2_3b_instruct.export --device "Snapdragon 8 Elite QRD" --output-dir genie_bundle --skip-profiling --skip-inferencing
    ```

    - Exporting Llama 3.2 models will take a while depending on your internet connectivity.
    - This takes around 1-2 hours with good internet connectivity.

2. Navigate to `src/main/assets/models/llm` and use this directory to store the assets.

    - Download and save `tokenizer.json` from the [LLM On-Device Deployment](https://github.com/qualcomm/ai-hub-apps/tree/main/tutorials/llm_on_genie#genie-config) tutorial to `src/main/assets/models/llm/`.

    -  If you would like, you may also go to the [HuggingFace](https://huggingface.co/) repository of your desired model and save `tokenizer.json` from there.

3. Copy model binaries (`genie_bundle/*.bin`) from step 1 to `src/main/assets/models/llm/`

    ```bash
    cp genie_bundle/*.bin src/main/assets/models/llm/
    ```

## License

This app is released under the [BSD-3 License](../../LICENSE) found at the root of this repository.

All models from [AI Hub Models](https://github.com/qualcomm/ai-hub-models) are released under separate license(s). Refer to the [AI Hub Models repository](https://github.com/qualcomm/ai-hub-models) for details on each model.

The QNN SDK dependency is also released under a separate license. Please refer to the LICENSE file downloaded with the SDK for details.
