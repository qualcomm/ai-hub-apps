# CLI Chat application on Snapdragon X Elite

A Windows C++ CLI app that runs an on-device LLM (e.g. [Llama 3.2 3B](https://aihub.qualcomm.com/compute/models/llama_v3_2_3b_instruct)) using the Genie APIs from the [QAIRT SDK](https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK), accelerated on the Snapdragon® NPU.

## Requirements

- Windows on Snapdragon X Elite
- A Windows host machine (x86-64 or ARM) with Docker installed

## Setup

### Option A: Using the CLI (Recommended)

Install the CLI and fetch the app with the model:

```powershell
pip install qai-hub-apps
qai-hub-apps fetch chatapp_windows_cpp --model qwen3_4b_instruct_2507 --chipset qualcomm-snapdragon-x-elite --output-dir ~
cd ~\chatapp_windows_cpp
```

This downloads the app source and places the Genie bundle into `genie_bundle/`.

### Option B: Cloning the Repo

If you cloned the release branch, the app directory is already self-contained — but **model weights are not included**. Obtain the LLM binaries from [AI Hub Models](https://aihub.qualcomm.com/models?domain=Generative+AI&useCase=Text+Generation) and unzip the bundle to `genie_bundle` before building. See [Exporting an LLM](#exporting-an-llm) for more details. You should see `genie_bundle\*.bin` context binary files.

## Build

### Option A: Using Visual Studio
In order to build using Visual Studio, you need to first install a supported QAIRT SDK and set the environment variable `QNN_SDK_ROOT` to its root directory.
You can use our provided helper utility to set up a supported QAIRT SDK.

```powershell
. ".\scripts\qairt_utils.ps1"

Install-Qairt
echo "QAIRT_PATH=$env:QAIRT_PATH"
```
Set the global environment variable `QNN_SDK_ROOT` to your QAIRT_PATH obtained from above.
Open `ChatApp.sln`, and build the `ARM64` configuration. The project downloads `json.hpp` automatically as a pre-build step.

### Option B: Using Docker
Build our Docker image with all required dependencies, including the supported MS Build Tools and QAIRT SDKs.

```powershell
docker build --build-arg BUILD_TYPE=build -t aiha-chatapp-win .
```

Build the EXE:
```powershell
docker run --name chatapp-container aiha-chatapp-win powershell -c '. ./install_build.ps1; & $env:MSBUILD_EXE ChatApp.sln /p:Configuration=Debug /p:Platform=ARM64'

mkdir ./ARM64

docker cp chatapp-container:C:\app\ARM64 .
```

### Install & Run
```powershell
.\ARM64\Debug\ChatApp.exe --genie-config ".\genie_bundle\genie_config.json" --base-dir ".\genie_bundle"
```

Run `--help` to learn more:

```powershell
.\ARM64\Debug\ChatApp.exe --help
```

Type `exit` during the chat to terminate.


### Unicode characters

To use languages that require Unicode, follow the [UTF-8 support instructions](https://github.com/qualcomm/ai-hub-apps/blob/main/tutorials/llm_on_genie/powershell/README.md#utf-8-support).

### Sample Output

![sample_output](assets/images/sample_output.png)

## Exporting an LLM

1. Get QNN context binaries for the LLM of your choice from Qualcomm AI Hub. There are two ways to get these assets:

    - Run the export script to get context binaries for Llama variants. We export these models with context length 4096 by default. You can add the argument `--context-length` with your desired context length value while exporting (recommended to use lower or equal to 4096). Make sure the `size` option in the genie config matches your model's context length.

    - Download directly from our website. Make sure to select the correct device when downloading the context binaries.

    - Read more about [exporting LLMs via AI Hub here](https://github.com/qualcomm/ai-hub-apps/tree/main/tutorials/llm_on_genie#1-generate-genie-compatible-qnn-binaries-from-ai-hub)
        - You'll have to replace the model name from the above tutorial with `llama_v3_2_3b_instruct` or the model id of your choice and reduce context length for this demo when exporting.

    - The following command exports the Llama 3.2 3B model with context length 4096:

    ```powershell
    python -m qai_hub_models.models.llama_v3_2_3b_instruct.export --device "Snapdragon X Elite CRD" --output-dir genie_bundle --skip-profiling --skip-inferencing
    ```

    - Exporting Llama 3.2 models will take a while depending on your internet connectivity.
    - This takes around 1-2 hours with good internet connectivity.

2. Download and save `tokenizer.json` from the [LLM On-Device Deployment](https://github.com/qualcomm/ai-hub-apps/tree/main/tutorials/llm_on_genie#genie-config) tutorial to `genie_bundle\`.

    - If you would like, you may also go to the [HuggingFace](https://huggingface.co/) repository of your desired model and save `tokenizer.json` from there.

3. The exported context binaries (`genie_bundle\*.bin`) from step 1 are already in `genie_bundle\`. Confirm the directory contains the `*.bin` files, `genie_config.json`, and `tokenizer.json`.
