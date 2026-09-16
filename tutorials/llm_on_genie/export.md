# Export QAIRT-compatible LLM models (on the host machine)

This page covers Option B of Step 2 in the [main tutorial](README.md): exporting
a QAIRT-compatible model yourself. Use this path when ready-made assets are not
available for the model you want to run (for example, the Llama family, which
requires gated Hugging Face access).

Export QAIRT-compatible models using the export scripts in
[qai-hub-models](https://github.com/qualcomm/ai-hub-models/tree/main/src/qai_hub_models/)
on the host machine (Linux, Windows, or macOS).

## Install Qualcomm AI Hub Models (on the host machine)

Use [qai-hub-models](https://github.com/qualcomm/ai-hub-models/tree/main/src/qai_hub_models/)
to adapt Hugging Face models for on-device inference. Most models have open
access and are downloaded automatically by the package.

If you have not set up a Python environment, follow
[Setting up a Python environment with Qualcomm AI Hub Models](README.md#setting-up-a-python-environment-with-qualcomm-ai-hub-models).

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

## Export models using Qualcomm AI Hub

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
# Snapdragon 8 Elite Gen 5
python -m qai_hub_models.models.llama_v3_2_3b_instruct.export --chipset qualcomm-snapdragon-8-elite-gen5 --skip-profiling --output-dir genie_bundle

# Snapdragon 8 Elite
python -m qai_hub_models.models.llama_v3_2_3b_instruct.export --chipset qualcomm-snapdragon-8-elite --skip-profiling --output-dir genie_bundle

# Snapdragon X2 Elite
python -m qai_hub_models.models.llama_v3_2_3b_instruct.export --chipset qualcomm-snapdragon-x2-elite --skip-profiling --output-dir genie_bundle

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

Once the bundle is ready, continue with [Step 3 in the main tutorial](README.md#step-3-run-the-llm-on-device).
