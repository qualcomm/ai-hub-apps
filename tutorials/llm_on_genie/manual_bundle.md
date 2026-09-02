# Prepare Genie bundle manually

If you download the assets directly from http://aihub.qualcomm.com/, then you
must construct the Genie bundle manually.

## Tokenizer

Please download the tokenizer file (`tokenizer.json`) from the LLM's Hugging
Face source repository and place in your Genie bundle folder:

| Model name | Tokenizer | Notes |
| --- | --- | --- |
| Llama-v2-7B-Chat | [tokenizer.json](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/tokenizer.json) | |
| Llama-v3-8B-Instruct | [tokenizer.json](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/tokenizer.json) | |
| Llama-v3.1-8B-Instruct | [tokenizer.json](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/tokenizer.json) | |
| Llama-SEA-LION-v3.5-8B-R | [tokenizer.json](https://huggingface.co/aisingapore/Llama-SEA-LION-v3.5-8B-R/blob/main/tokenizer.json) | |
| Llama-v3.2-3B-Instruct | [tokenizer.json](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/tokenizer.json) | |
| Llama-v3.2-1B-Instruct | [tokenizer.json](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/tokenizer.json) | |
| Llama3-TAIDE-LX-8B-Chat-Alpha1 | [tokenizer.json](https://huggingface.co/taide/Llama3-TAIDE-LX-8B-Chat-Alpha1/blob/main/tokenizer.json) | |
| Baichuan2-7B | [tokenizer.json](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/baichuan2_7b_quantized/v2/tokenizer.json) | |
| Qwen2-7B-Instruct | [tokenizer.json](https://huggingface.co/Qwen/Qwen2-7B-Instruct/blob/main/tokenizer.json) | |
| Qwen2.5-7B-Instruct | [tokenizer.json](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/tokenizer.json) | |
| Phi-3.5-Mini-Instruct | [tokenizer.json](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/tokenizer.json) | To see appropriate spaces in the output, remove lines 193-196 (Strip rule) in the tokenizer file. |
| Mistral-7B-Instruct-v0.3 | [tokenizer.json](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/main/tokenizer.json) | |
| IBM-Granite-v3.1-8B-Instruct | [tokenizer.json](https://huggingface.co/ibm-granite/granite-3.1-8b-base/blob/main/tokenizer.json) | |
| Falcon3-7B-Instruct | [tokenizer.json](https://huggingface.co/tiiuae/Falcon3-7B-Instruct/blob/main/tokenizer.json) | |

## [Optional] Use the Windows PowerShell LLM Runner

**Do not use this script to create your Genie bundle if you are building Windows ChatApp. Continue with the rest of the tutorial instead.**

The easiest path to running an LLM on a Windows on Snapdragon® device is to use the [PowerShell implementation](powershell/)
of the rest of this tutorial. It will automatically generate the appropriate configuration files and execute `genie-t2t-run.exe`
on a prompt of your choosing.

## Genie Config

Check out the [AI Hub Apps repository](https://github.com/qualcomm/ai-hub-apps)
using Git:

```bash
git clone https://github.com/qualcomm/ai-hub-apps.git
```

Now run (replacing `llama_v3_8b_instruct` with the desired model id):

```bash
cp ai-hub-apps/tutorials/llm_on_genie/configs/genie/llama_v3_8b_instruct.json genie_bundle/genie_config.json
```

For Windows laptops, please set `use-mmap` to `false`.

If you customized context length by adding `--context-length` to the export
command, please open `genie_config.json` and modify the `"size"` option (under
`"dialog"` -> `"context"`) to be consistent.

In `genie_bundle/genie_config.json`, also ensure that the list of bin files in
`ctx-bins` matches with the bin files under `genie_bundle`. Genie will look for
QNN binaries specified here.

## HTP Backend Config

Copy the HTP config template:

```bash
cp ai-hub-apps/tutorials/llm_on_genie/configs/htp/htp_backend_ext_config.json.template genie_bundle/htp_backend_ext_config.json
```

Edit `soc_model` and `dsp_arch` in `genie_bundle/htp_backend_ext_config.json`
depending on your target device (should be consistent with the `--chipset` you
specified in the export command):

| Generation               | `soc_model` | `dsp_arch` |
|--------------------------|--------|----------|
| Snapdragon® Gen 2        | 43     | v73      |
| Snapdragon® Gen 3        | 57     | v75      |
| Snapdragon® 8 Elite      | 69     | v79      |
| Snapdragon® X Elite      | 60     | v73      |
| Snapdragon® X Plus       | 60     | v73      |

## Collect & Finalize Genie Bundle

When finished with the above steps, your bundle should look like this:
```
genie_bundle/
   genie_config.json
   htp_backend_ext_config.json
   tokenizer.json
   <model_id>_part_1_of_N.bin
   ...
   <model_id>_part_N_of_N.bin
```

where `<model_id>` is the name of the model. This is the name of the json you
copied from `configs/genie/<model_name>.json`. If you want to run
`genie-t2t-run` (as opposed to the apps), then you also need to copy files from
the QAIRT SDK into the Genie bundle.

## Genie on Windows with Snapdragon® X

Copy Genie's library files and executable to our bundle. Please make sure
the environment variable `QNN_SDK_ROOT` points to your QAIRT SDK root before
executing these steps in PowerShell:

```powershell
Copy-Item "$env:QNN_SDK_ROOT/lib/hexagon-v73/unsigned/*" genie_bundle
Copy-Item "$env:QNN_SDK_ROOT/lib/aarch64-windows-msvc/*" genie_bundle
Copy-Item "$env:QNN_SDK_ROOT/bin/aarch64-windows-msvc/genie-t2t-run.exe" genie_bundle
```

#### Genie on Android

Copy Genie's shared libraries and executable to our bundle.

```bash
# For 8 Gen 2
cp "$QNN_SDK_ROOT"/lib/hexagon-v73/unsigned/* genie_bundle
# For 8 Gen 3
cp "$QNN_SDK_ROOT"/lib/hexagon-v75/unsigned/* genie_bundle
# For 8 Elite
cp "$QNN_SDK_ROOT"/lib/hexagon-v79/unsigned/* genie_bundle
# For all devices
cp "$QNN_SDK_ROOT"/lib/aarch64-android/* genie_bundle
cp "$QNN_SDK_ROOT"/bin/aarch64-android/genie-t2t-run genie_bundle
```

Now, continue the [main tutorial](README.md).
