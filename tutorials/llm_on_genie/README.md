# LLM On-Device Deployment

In this tutorial, we will run large language models (LLMs) on-device on
Snapdragon® platforms such as:

- Android: Snapdragon® 8 Elite, Snapdragon® 8 Gen 3 (e.g. Samsung Galaxy Series)
- Windows: Snapdragon® X Elite (e.g. Snapdragon® based Microsoft Surface Pro)
- Linux: Dragonwing® Platforms (e.g. Dragonwing® QCM8550, Dragonwing® IQ-9075)

We use the [Llama 3.2 3B
Instruct](https://aihub.qualcomm.com/automotive/models/llama_v3_2_3b_instruct)
model as the running example. In case of any questions, please feel free to post
them on the [Qualcomm AI Hub Slack
channel](https://aihub.qualcomm.com/community/slack).

## Overview

There are three steps to run
[Llama 3.2 3B Instruct](https://aihub.qualcomm.com/models/llama_v3_2_3b_instruct):

- **Step 1:** Install the [QAIRT SDK](https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community) on the target device.
- **Step 2:** Prepare models compatible with the QAIRT SDK on the host machine.
  - Get access to [Llama 3 weights from Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
  - Use [Qualcomm AI Hub Models](https://github.com/quic/ai-hub-models/tree/main/qai_hub_models/models) to export Llama 3 using [Qualcomm AI Hub](https://aihub.qualcomm.com/).
- **Step 3:** Run the LLM on the target device with an example prompt.

> [!IMPORTANT]
> Some models are available directly from [Qualcomm AI Hub
> Models](https://aihub.qualcomm.com/models). For those models, Step 2 is optional.

## Requirements

> [!IMPORTANT]
Target device requirements:

- Hexagon architecture v73 or above (please see [Devices](https://app.aihub.qualcomm.com/devices/) list).
- 16GB memory or more for 7B+ or 4096 context length models.
- 12GB memory or more for 3B+ models (and you may need to adjust down context length).

Software requirements:

- Android 15+, Windows 11
- [QAIRT SDK](https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK) v2.29.0+ (see [QNN SDK](https://qpm.qualcomm.com/#/main/tools/details/qualcomm_ai_engine_direct) for versions prior to 2.32; Automotive devices require Auto QAIRT SDK instead of the standard QAIRT SDK, see below for more details)
- [qai-hub-models](https://pypi.org/project/qai-hub-models/)

> [!IMPORTANT]
> Each [model card](https://aihub.qualcomm.com/models/llama_v3_2_3b_instruct)
> specifically lists compatible devices and minimum QAIRT SDK versions.
> Ensure device and software requirements are met before proceeding.

## Step 1: Install QAIRT (on the target device)

We recommend the use of the same version of QAIRT SDK on-target that Qualcomm AI
used to compile the assets.  The QAIRT version is displayed on model cards (for
pre-compiled models) in the [Qualcomm AI Hub model
cards](https://aihub.qualcomm.com/models) or by clicking on the job links
produced by the export scripts of the [Qualcomm AI Hub models python
package](https://github.com/quic/ai-hub-models).

Download the specific version of the QAIRT SDK from the [Qualcomm Software
Center](https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community)
and copy it to the target device. If the target has internet connectivity, you
can also download it directly using `wget` with the URL shown in the Software
Center. Alternately, download [QAIRT SDK](https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK) and install it via [QPM.](https://docs.qualcomm.com/bundle/publicresource/topics/80-88500-5/install_qualcomm_package_manager_qpm.html)

> [!IMPORTANT]
> Automotive devices require access to the Auto specific QAIRT SDK which can be obtained by contacting your Qualcomm Account Manager, for those looking to purchase Auto SoCs and gain access to Auto SDK, please reach out on our [Qualcomm AI Hub Slack Community](http://aihub.qualcomm.com/community/slack) for next steps. Additional steps required to set up auto devices can be found in [Android (Automotive)](https://github.com/qcom-ai-hub/ai-hub-apps-internal/tree/main/tutorials/llm_on_genie#android-automotive)


Once downloaded, please set the following environment variables:

### Android (bash)

Please make sure the architecture matches that of the device. The [mapping of
architecture to device](https://app.aihub.qualcomm.com/devices) can help.

```bash
export QAIRT_HOME= ## Location of downloaded QAIRT SDK on target device
export PATH=${QAIRT_HOME}/bin/aarch64-android/:${PATH}
export LD_LIBRARY_PATH=${QAIRT_HOME}/lib/aarch64-android:${LD_LIBRARY_PATH}

# This device uses v73
export ADSP_LIBRARY_PATH=${QAIRT_HOME}/lib/hexagon-v73/unsigned
```

### Windows (Powershell)

Note that in Windows Powershell, the binaries and libraries are loaded from the
`$env:Path` variable.

```powershell
$env:QAIRT_HOME = ## Location of downloaded QAIRT SDK
$env:Path = "$env:QAIRT_HOME\bin\aarch64-windows-msvc;" + $env:Path
$env:Path = "$env:QAIRT_HOME\lib\aarch64-windows-msvc;" + $env:Path

# Please make sure the architecture matches that of the device (v73, v75)
$env:ADSP_LIBRARY_PATH = "$env:QAIRT_HOME\lib\hexagon-v73\unsigned"
```

### Linux - Ubuntu (bash)

```bash
# Copy Genie to correct folders
cp ${QAIRT_HOME}/lib/aarch64-oe-linux-gcc11.2/libGenie.so ${QAIRT_HOME}/lib/aarch64-ubuntu-gcc9.4/
cp ${QAIRT_HOME}/bin/aarch64-oe-linux-gcc11.2/genie* ${QAIRT_HOME}/bin/aarch64-ubuntu-gcc9.4/

# Export environment variables
export QAIRT_HOME= ## Location of downloaded QAIRT SDK on target device
export PATH=${QAIRT_HOME}/bin/aarch64-ubuntu-gcc9.4/:${PATH}
export LD_LIBRARY_PATH=${QAIRT_HOME}/lib/aarch64-ubuntu-gcc9.4:${LD_LIBRARY_PATH}

# Please make sure the architecture matches that of the device (v73, v75)
export ADSP_LIBRARY_PATH=${QAIRT_HOME}/lib/hexagon-v73/unsigned
```

These changes can be made permanent by adding the above lines to the `~/.bashrc` file on
Android/Linux and `$PROFILE` on Windows PowerShell.

> [!IMPORTANT]
> Please make sure the `ADSP_LIBRARY_PATH` variable points to the libraries
> for the appropriate architecture. The [mapping of device to architecture](https://app.aihub.qualcomm.com/devices)
> can provide additional details.

### Android (Automotive)

```bash
export AUTO_QAIRT_HOME= ## Location of downloaded Auto QAIRT SDK on target device
export PATH=${AUTO_QAIRT_HOME}/bin/aarch64-android/:${PATH}

# Please make sure the architecture matches that of the device (v73, v75); replace vXX with appropriate architecture
export VENDOR_LIB=${AUTO_QAIRT_HOME}/lib/hexagon-vXX/unsigned

export ADSP_LIBRARY_PATH="/vendor/lib/rfsa/adsp;$VENDOR_LIB;"
export LD_LIBRARY_PATH=${AUTO_QAIRT_HOME}/lib/aarch64-android:/vendor/lib64/
```

For auto devices, once the SDK is installed, users need to copy additional library files (`libc++.so.1` and `libc++abi.so.1`) from QNX to Linux/Android Guest Virtual Machine (LA GVM) in order to run `genie-t2t`.

* A standard automotive device may have more than one VM running - a primary VM (QNX) and guest VMs through a hypervisor (Android or Linux or both). You will be running the application in LA GVM while the required files are present in QNX.
* In order to transfer the required library files, you need to FTP to QNX and copy the library files to a HOST system and then push it to the LA GVM (preferrably through ADB). The library files will be available in `/dsplib/image/dsp/cdsp0/` or `/mnt/etc/images/cdsp0/` under your QNX filesystem.
* Copy the `libc++.so.1` and `libc++abi.so.1` files from this directory to your HOST system and connect to LA GVM. Push these files to `$VENDOR_LIB` in LA GVM from your HOST system.

> [!NOTE]
> On QDC automotive devices, `libc++.so.1` and `libc++abi.so.1` are typically already present in the LA GVM at `/data/local/tmp/qxa.qa_adsplib/`. You can use the following command to copy the files:
> ```
> cp /data/local/tmp/qxa.qa_adsplib/libc++.so.1 $VENDOR_LIB
> cp /data/local/tmp/qxa.qa_adsplib/libc++abi.so.1 $VENDOR_LIB

## Step 2: Export QAIRT-compatible LLM models (on the host machine)

Export QAIRT-compatible models using the export scripts in
[qai-hub-models](https://github.com/quic/ai-hub-models/tree/main/qai_hub_models/)
on the host machine (Linux, Windows, or macOS).

> [!NOTE]
> Some models are available directly from
> [Qualcomm AI Hub Models](https://aihub.qualcomm.com/models). For those models,
> download and copy them to the device.

### Install Qualcomm AI Hub Models (on the host machine)

Use [qai-hub-models](https://github.com/quic/ai-hub-models/tree/main/qai_hub_models/)
to adapt Hugging Face Llama models for on-device inference. Most models have
open access and are downloaded automatically by the package.

If you have not set up a Python environment, follow
[Setting up a Python environment with Qualcomm AI Hub Models](#setting-up-a-python-environment-with-qualcomm-ai-hub-models).

### Set up Hugging Face tokens (models with restricted access)

Set up a Hugging Face token on the host by following the
[Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
```

> [!IMPORTANT]
> A Hugging Face token is required only for the Llama model family. Request
> [access to Llama 3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).

### Export models using Qualcomm AI Hub

Generate assets for Llama 3.2 3B using the export script below. It downloads
model weights from Hugging Face, compiles for your target device, and prepares a
bundle for deployment. First, install AI Hub Models with the right dependencies
for Llama 3.2 3B:

```
pip install "qai-hub-models[llama-v3-2-3b-instruct]"
```

For other models, please confirm the exact command in the model's README file
(linked from the model cards at [Qualcomm AI Hub
Models](https://aihub.qualcomm.com/models)).

> [!IMPORTANT]
> The export command may take 2–3 hours and requires significant memory (RAM +
> swap) on the host. If you are prompted that your memory is insufficient,
> please see [Increase Swap space](increase_swap.md).

```bash
# Snapdragon 8 Elite
python -m qai_hub_models.models.llama_v3_2_3b_instruct.export --chipset qualcomm-snapdragon-8-elite --skip-profiling --output-dir genie_bundle

# Snapdragon 8 Gen 3
python -m qai_hub_models.models.llama_v3_2_3b_instruct.export --chipset qualcomm-snapdragon-8gen3 --skip-profiling --output-dir genie_bundle

# Snapdragon X Elite
python -m qai_hub_models.models.llama_v3_2_3b_instruct.export --chipset qualcomm-snapdragon-x-elite --skip-profiling --output-dir genie_bundle
```

> [!NOTE]
> On memory-constrained target devices, reduce the context length with
> `--context-length <context-length>`.

The export script places context binaries, tokenizer, and Genie configuration
files into the `genie_bundle` folder. If you plan to run directly via
`genie-t2t-run`, follow the instructions printed at the end of the export.

For some older models, the tokenizer and Genie configuration is not
automatically created by the export script. In such case, see [Prepare Genie
bundle manually](manual_bundle.md).

## Step 3: Run the LLM on-device

You have three options to run the LLM on device:

- Option 1: Use the `genie-t2t-run` CLI command
- Option 2: Use the [CLI Windows ChatApp](https://github.com/quic/ai-hub-apps/tree/main/apps/windows/cpp/ChatApp)
- Option 3: Use the [Android ChatApp](https://github.com/quic/ai-hub-apps/tree/main/apps/android/ChatApp)

### *Option 1*: Run Genie via `genie-t2t-run`

The QAIRT SDK (Android, Windows, and Linux) provides an executable called
`genie-t2t-run` to run bundle-formatted LLM models exported via Qualcomm AI Hub.

Many system prompts contain the line feed character (`\n`, ascii 0x0a) as part
of the correct prompt format. We have to pay extra attention to how we pass this
into the LLM so that it is not passed in as `\` and `n` as two separate
characters. This is platform-specific, so more on this in the sections below.

See the section on [prompt formats for various models](#prompt-formats).

> [!IMPORTANT]
> On all platforms, we recommend copy-pasting the prompts into a separate file
> (e.g., `prompt.txt`) and passing that prompt into `genie-t2t-run` with
> `--prompt_file prompt.txt`. If you take this approach, please make sure you
> replace the `\n` characters with real newlines.

#### Windows on Snapdragon® X

In PowerShell, this can be run using the following command

```bash
genie-t2t-run.exe -c genie_config.json -p "<|begin_of_text|><|start_header_id|>user<|end_header_id|>`n`nWhat is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
```

> [!IMPORTANT]
> This prompt format is specific to Llama 3. Use `` `n `` (instead of `\n`) to
> pass a real line feed in Windows PowerShell.

For non‑Latin languages (e.g., Chinese, Arabic), first [configure Windows to use UTF‑8](windows/utf8.md).

#### Android

Copy `genie_bundle` from the host to the device using ADB, then open an
interactive shell:

```bash
adb push genie_bundle /data/local/tmp
adb shell
```

Once copied to the device, run the following:

```bash
genie-t2t-run -c genie_config.json -p "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"$'\n\n'$"What is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
```

> [!IMPORTANT]
> To pass real line feeds into Genie in Bash, use `$'\n\n'$` instead of
> `"\n\n"`. We generally recommend using a prompt file with real newlines.

#### Linux (Ubuntu)

This can be run using the following command:

```bash
genie-t2t-run -c genie_config.json -p "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"$'\n\n'$"What is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
```

#### Sample output

```text
Using libGenie.so version 1.1.0

[WARN]  "Unable to initialize logging in backend extensions."
[INFO]  "Using create From Binary List Async"
[INFO]  "Allocated total size = 323453440 across 10 buffers"
[PROMPT]: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

[BEGIN]: \n\nFrance's capital is Paris.[END]
```

Performance KPIs (token rate, time-to-first-token, etc.) can be obtained by
passing `--profile path_to_txt_file.txt` to `genie-t2t-run`.

### Option 2: Sample C++ Chat App Powered by Genie SDK

We provide a sample C++ app to show how to build an application using the Genie
SDK. See the [CLI Windows
ChatApp](https://github.com/quic/ai-hub-apps/tree/main/apps/windows/cpp/ChatApp)
for more details.

### Option 3: Sample Android Chat App Powered by Genie SDK

We provide a sample Android app (Java and C++) to show how to build an
application using the Genie SDK for mobile. See [Android
ChatApp](https://github.com/quic/ai-hub-apps/tree/main/apps/android/ChatApp) for
more details.

## Additional Assistance

In this section, we cover a few topics in more detail for those new to
some of these concepts.

### Setting up a Python environment with Qualcomm AI Hub Models

Following standard best practices, we recommend creating a virtual environment
specifically for exporting AI Hub models. The following steps can be performed
on Windows, Linux, or macOS. On Windows, you can either install x86-64 Python
(since package support is limited on native ARM64 Python) or use Windows
Subsystem for Linux (WSL).

#### Create Virtual Environment

Create a [virtualenv](https://virtualenv.pypa.io/en/latest/) for `qai-hub-models` with Python 3.10.
You can also use [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

For clarity, we recommend creating a virtual environment:

```bash
python3.10 -m venv llm_on_genie_venv
```

#### Install `qai-hub-models`

In a shell session, install `qai-hub-models` in the virtual environment:

```bash
source llm_on_genie_venv/bin/activate
pip install -U "qai-hub-models[llama-v3.2-3b-instruct]"
```

Replace `llama-v3.2-3b-instruct` with the desired Llama model from [AI Hub
Models](https://github.com/quic/ai-hub-models/tree/main/qai_hub_models/models).
Note to replace `_` with `-` (e.g., `llama_v3.2_3b_instruct` -> `llama-v3.2-3b-instruct`).


### Prompt Formats

Different LLMs have different prompt formats. To get sensible output, it is
important to use the correct prompt format for each model. These can also be
found on the Hugging Face repository for each model. A few examples are below.

| Model name                                                                                               | Sample Prompt                                                                                                                                                                                                                                                                                                                                                                            |
| -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Llama-v2-7B-Chat                                                                                         | &lt;s&gt;[INST] &lt;&lt;SYS&gt;&gt;You are a helpful AI Assistant.&lt;&lt;/SYS&gt;&gt;[/INST]&lt;/s>&lt;s&gt;[INST]What is France's capital?[/INST]                                                                                                                                                                                                                                      |
| Llama-v3-8B-Instruct <br> Llama-v3.1-8B-Instruct <br> Llama-v3.2-3B-Instruct <br> Llama-v3.2-1B-Instruct | <&#124;begin_of_text&#124;><&#124;start_header_id&#124;>user<&#124;end_header_id&#124;>\n\nWhat is France's capital?<&#124;eot_id&#124;><&#124;start_header_id&#124;>assistant<&#124;end_header_id&#124;>                                                                                                                                                                                |
| Llama3-TAIDE-LX-8B-Chat-Alpha1                                                                           | <&#124;begin_of_text&#124;><&#124;start_header_id&#124;>system<&#124;end_header_id&#124;>\n\n 你是一個來自台灣的 AI 助理，你的名字是 TAIDE，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題<&#124;eot_id&#124;>\n<&#124;start_header_id&#124;>user<&#124;end_header_id&#124;>\n\n 介紹台灣特色<&#124;eot_id&#124;>\n<&#124;start_header_id&#124;>assistant<&#124;end_header_id&#124;> |
| Llama-SEA-LION-v3.5-8B-R (non-thinking mode)                                                             | <&#124;begin_of_text&#124;><&#124;start_header_id&#124;>system<&#124;end_header_id&#124;>\n\ndetailed thinking off<&#124;eot_id&#124;><&#124;start_header_id&#124;>user<&#124;end_header_id&#124;>\n\nThủ đô của Việt Nam là thành phố nào?<&#124;eot_id&#124;><&#124;start_header_id&#124;>assistant<&#124;end_header_id&#124;>\n\n&lt;think&gt;\n\n&lt;/think&gt;>\n\n                 |
| Llama-SEA-LION-v3.5-8B-R (thinking mode)                                                                 | <&#124;begin_of_text&#124;><&#124;start_header_id&#124;>system<&#124;end_header_id&#124;>\n\ndetailed thinking on<&#124;eot_id&#124;><&#124;start_header_id&#124;>user<&#124;end_header_id&#124;>\n\nThủ đô của Việt Nam là thành phố nào?<&#124;eot_id&#124;><&#124;start_header_id&#124;>assistant<&#124;end_header_id&#124;>\n\n&lt;think&gt;\nHere is my thinking:\n                 |
| Qwen2-7B-Instruct <br> Qwen2.5-7B-Instruct                                                               | <&#124;im_start&#124;>system\nYou are a helpful AI Assistant<&#124;im_end&#124;><&#124;im_start&#124;>What is France's capital?\n<&#124;im_end&#124;>\n<&#124;im_start&#124;>assistant\n                                                                                                                                                                                                 |
| Phi-3.5-Mini-Instruct                                                                                    | <&#124;system&#124;>\nYou are a helpful assistant. Be helpful but brief.<&#124;end&#124;>\n<&#124;user&#124;>What is France's capital?\n<&#124;end&#124;>\n<&#124;assistant&#124;>\n                                                                                                                                                                                                     |
| Mistral-7B-Instruct-v0.3                                                                                 | &lt;s&gt;[INST] You are a helpful assistant\n\nTranslate 'Good morning, how are you?' into French.[/INST]                                                                                                                                                                                                                                                                                |
| IBM-Granite-v3.1-8B-Instruct                                                                             | <&#124;start_of_role&#124;>system<&#124;end_of_role&#124;>You are a helpful AI assistant.<&#124;end_of_text&#124;>\n <&#124;start_of_role&#124;>user<&#124;end_of_role&#124;>What is France's capital?<&#124;end_of_text&#124;>\n <&#124;start_of_role&#124;>assistant<&#124;end_of_role&#124;>\n                                                                                        |
| Falcon3-7B-Instruct                                                                                      | <&#124;system&#124;>\nYou are a helpful friendly assistant Falcon3 from TII, try to follow instructions as much as possible.\n<&#124;user&#124;>\nWhat is France's capital?\n<&#124;assistant&#124;>\n                                                                                        |
| Llama-3-ELYZA-JP-8B                                                                                      | <&#124;begin_of_text&#124;><&#124;start_header_id&#124;>system<&#124;end_header_id&#124;>\n\nあなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。<&#124;eot_id&#124;>\n<&#124;start_header_id&#124;>user<&#124;end_header_id&#124;>\n\n仕事の熱意を取り戻すためのアイデアを5つ挙げてください。<&#124;eot_id&#124;>\n<&#124;start_header_id&#124;>assistant<&#124;end_header_id&#124;>\n                                                                                        |
