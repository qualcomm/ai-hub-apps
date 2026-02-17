# OpenAI-Compatible HTTP Chat Application

Chat application for Windows on Snapdragon® demonstrating a large language model (LLM, e.g., [Llama 3.2 3B](https://aihub.qualcomm.com/compute/models/llama_v3_2_3b_instruct)) using Genie SDK, now serving an OpenAI-compatible HTTP endpoint.

The app demonstrates how to use the Genie APIs from [QAIRT SDK](https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK) to run and accelerate LLMs using the Snapdragon® Neural Processing Unit (NPU).

## Requirements

### Platform

- Snapdragon® Platform (e.g. X Elite)
- Windows 11+

### Tools and SDK

- Visual Studio 22
  - Download any variant of [Visual Studio here](https://visualstudio.microsoft.com/vs/)
  - Make sure Desktop development with C++ tools are selected during installation or installed separately later
- QAIRT SDK: [Qualcomm AI Runtime SDK](https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK) (see [QNN SDK](https://qpm.qualcomm.com/#/main/tools/details/qualcomm_ai_engine_direct) for older versions)
  - Refer to [Setup QAIRT SDK](#setup-qairt-sdk) to install compatible QAIRT SDK for models downloaded from AI Hub.

- [cpp-httplib](https://github.com/yhirose/cpp-httplib) and [nlohmann/json](https://github.com/nlohmann/json) single-header libraries (see below)

## Build App

### Compile to Context Binary via AI Hub and Generate Genie Bundle

1. Clone this repository so that you have a local copy of `ChatServer`.
2. Please follow [this tutorial](https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie) to generate `genie_bundle` required by ChatApp. If you use any of the Llama 3 models, the app will work without modifications. If you use another model, you will need to update the prompt format in `PromptHandler.cpp` first.
3. Copy bundle assets from step 2 to `ChatApp\genie_bundle`. You should see `ChatApp\genie_bundle\*.bin` context binary files.

### Setup QAIRT SDK

Please ensure that the QAIRT (or QNN) SDK version installed on the system is the same as the one used by AI Hub for generating context binaries.
You can find the AI Hub QAIRT version in the compile job page as shown in the following screenshot:

![QAIRT version on AI Hub](assets/images/ai-hub-qnn-version.png)

Having different QAIRT versions could result in runtime or load-time failures.

Please follow the following steps to configure QAIRT SDKs for ChatApp:

1. Download and install [Qualcomm AI Runtime SDK](https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK) (see [QNN SDK](https://qpm.qualcomm.com/#/main/tools/details/qualcomm_ai_engine_direct) for older versions)
2. Set global environment variable `QNN_SDK_ROOT` to root path of QAIRT SDK e.g. `C:\Qualcomm\AIStack\QAIRT\2.32.0.250228`

    - Make sure you can run the following command with no error and that it prints the various libraries available with your QAIRT package:

    ```powershell
    ls ${env:QNN_SDK_ROOT}/lib
    ```

    ![QNN SDK Verion check](assets/images/sample-qnn-sdk-check.png)
3. If command from step 2 succeeds, QAIRT SDK is correctly configured to work with ChatApp.

### Add HTTP and JSON dependencies

Download the following single-header libraries and place them in the project's root directory:

- [`httplib.h`](https://github.com/yhirose/cpp-httplib/blob/master/httplib.h)
- [`nlohmann/json.hpp`](https://github.com/nlohmann/json/blob/develop/single_include/nlohmann/json.hpp)

Example (PowerShell):

```powershell
Invoke-WebRequest -Uri https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h -OutFile httplib.h
Invoke-WebRequest -Uri https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp -OutFile nlohmann/json.hpp
```

## Build project in Visual Studio 22

Make sure `QNN_SDK_ROOT` is set globally pointing to QAIRT SDK before you build the project.

**Important:**
- In Visual Studio, right-click your project (`ChatServer`) in Solution Explorer and select **Properties**.
- Go to **Configuration Properties > C/C++ > General > Additional Include Directories**.
- Add the following entry to the list:
  - `$(ProjectDir)`
- Click OK to save. This ensures the compiler can find `httplib.h` and `nlohmann/json.hpp` if they are in your project root.

1. Open `ChatServer.sln`
2. Build project in Visual Studio

## Running App

### Running via Visual Studio or CLI

Click on the green play button or run from CLI with:

```powershell
cd {Project directory}
.\ARM64\Debug\ChatApp.exe --genie-config .\\genie_bundle\\genie_config.json --base-dir .\\genie_bundle\\
```

Run `--help` to learn more:

```powershell
.\ARM64\Debug\ChatApp.exe --help
```

### HTTP API Usage

After starting, the app will listen on port 8080 by default and expose an OpenAI-compatible endpoint:

#### POST /v1/chat/completions

- **Request Body:**

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
}
```

- **Response:**

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1719148800,
  "model": "genie",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Hi! How can I help you today?"},
      "finish_reason": "stop"
    }
  ]
}
```

- **Example curl:**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]}'
```

### File Path Notes

Make sure to provide paths to local config file and models using `\\` or `/` as a path separator and not `\`

#### Correct file path examples

```powershell
1. C:\\Path\\To\\Model\\Config\\llama2_7b.json
2. C:/Path/To/Model/Config/llama2_7b.json
```

#### Incorrect file path example

```powershell
1. C:\Path\To\Model\Config\llama2_7b.json
```

#### Unicode characters

To use languages that require Unicode, please follow these instructions:

* [UTF-8 support](https://github.com/quic/ai-hub-apps/blob/main/tutorials/llm_on_genie/powershell/README.md#utf-8-support)

### Sample Output

![sample_output](assets/images/sample_output.png)
