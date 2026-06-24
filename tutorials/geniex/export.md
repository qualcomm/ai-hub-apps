# Export QAIRT-compatible LLM models (on the host machine)

This page covers exporting a Qualcomm® AI Engine Direct (QAIRT) compatible model
yourself, for use with GenieX. Use this path when ready-made assets are not
available for the model you want to run (for example, the Llama family, which
requires gated Hugging Face access).

Export is performed with the export scripts in
[qai-hub-models](https://github.com/qualcomm/ai-hub-models/tree/main/src/qai_hub_models/)
on the host machine (Linux, Windows, or macOS). It produces a model bundle that
GenieX can then run via `geniex pull --local-path` and `geniex infer`.

## Set up a Python environment with Qualcomm AI Hub Models

Following standard best practices, we recommend creating a virtual environment
specifically for exporting AI Hub models. The following steps can be performed
on Windows, Linux, or macOS. On Windows, you can either install x86-64 Python
(since package support is limited on native ARM64 Python) or use Windows
Subsystem for Linux (WSL).

Create a [virtualenv](https://virtualenv.pypa.io/en/latest/) for `qai-hub-models`
with Python 3.10. You can also use
[conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

```bash
python3.10 -m venv geniex_venv
source geniex_venv/bin/activate
```

Use [qai-hub-models](https://github.com/qualcomm/ai-hub-models/tree/main/src/qai_hub_models/)
to adapt Hugging Face models for on-device inference. Most models have open
access and are downloaded automatically by the package.

## Set up Hugging Face tokens (models with restricted access)

Set up a Hugging Face token on the host by following the
[Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
```

> [!IMPORTANT]
> A Hugging Face token is required only for the Llama model family. Request
> [access to Llama 3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).

## Export the model using Qualcomm AI Hub

Generate assets for Llama 3.2 3B using the export script below. It downloads
model weights from Hugging Face, compiles for your target device, and prepares a
bundle for deployment. First, install AI Hub Models with the right dependencies
for Llama 3.2 3B:

```bash
pip install "qai-hub-models[llama-v3-2-3b-instruct]"
```

For other models, please confirm the exact command in the model's README file
(linked from the model cards at
[Qualcomm AI Hub Models](https://aihub.qualcomm.com/models)). Note to replace
`_` with `-` (e.g., `llama_v3_2_3b_instruct` -> `llama-v3-2-3b-instruct`).

> [!IMPORTANT]
> The export command may take 2–3 hours and requires significant memory (RAM +
> swap) on the host. If you are prompted that your memory is insufficient,
> please see [Increase Swap space](increase_swap.md).

```bash
# Snapdragon 8 Elite Gen 5
python -m qai_hub_models.models.llama_v3_2_3b_instruct.export --chipset qualcomm-snapdragon-8-elite-gen5 --target-runtime geniex_qairt --skip-profiling --output-dir geniex_bundle

# Snapdragon 8 Elite
python -m qai_hub_models.models.llama_v3_2_3b_instruct.export --chipset qualcomm-snapdragon-8-elite --target-runtime geniex_qairt --skip-profiling --output-dir geniex_bundle

# Snapdragon X2 Elite
python -m qai_hub_models.models.llama_v3_2_3b_instruct.export --chipset qualcomm-snapdragon-x2-elite --target-runtime geniex_qairt --skip-profiling --output-dir geniex_bundle

# Snapdragon X Elite
python -m qai_hub_models.models.llama_v3_2_3b_instruct.export --chipset qualcomm-snapdragon-x-elite --target-runtime geniex_qairt --skip-profiling --output-dir geniex_bundle
```

> [!NOTE]
> On memory-constrained target devices, reduce the context length with
> `--context-lengths <length>`, where `<length>` is a single value (e.g.
> `4096`) or a comma-separated list (e.g. `1024,4096`).

The export script places context binaries, tokenizer, and configuration files
into the `geniex_bundle` folder.

## Run the exported bundle with GenieX

Register the exported folder with GenieX and run it directly:

```powershell
# Windows ARM64
geniex pull local/llama_v3_2_3b_instruct --local-path (Resolve-Path .\geniex_bundle).Path
geniex infer local/llama_v3_2_3b_instruct
```

```bash
# Linux ARM64
geniex pull local/llama_v3_2_3b_instruct --local-path "$(realpath ./geniex_bundle)"
geniex infer local/llama_v3_2_3b_instruct
```

> [!NOTE]
> `geniex pull` copies the bundle into the GenieX cache. After a successful
> pull you can safely delete the export folder. Use `geniex list` to confirm the
> model is cached.

For installation and all other run options (Python SDK, Android SDK, local
server), see the
[GenieX documentation](https://refactored-happiness-4qyl9vn.pages.github.io/).
