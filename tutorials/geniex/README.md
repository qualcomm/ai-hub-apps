# LLM On-Device Deployment with GenieX

This tutorial runs large language models (LLMs) on-device on Snapdragon®
platforms using **[GenieX](https://github.com/qualcomm/geniex)** — Qualcomm's
packaged inference runtime. GenieX wraps model download, preparation, and
execution behind a single CLI/SDK: `geniex pull` fetches and prepares the
model, `geniex infer` runs it.

Supported platforms:

- **Windows ARM64** — Snapdragon® X Elite, Snapdragon® X2 Elite
- **Android** — Snapdragon® 8 Elite, Snapdragon® 8 Elite Gen 5
- **Linux ARM64** — Dragonwing® IoT (e.g. Dragonwing® IQ-9075)

> [!IMPORTANT]
> **The complete, authoritative
> [GenieX documentation](https://refactored-happiness-4qyl9vn.pages.github.io/)**
> covers installation, running
> models, supported platforms and runtimes, the Python and Android SDKs, the
> local server, and troubleshooting. This tutorial intentionally keeps only a
> short quickstart plus the model-export workflow (which is an upstream
> `qai-hub-models` step, not part of the GenieX docs) so there is a single
> source of truth to maintain.

## Quickstart

First install the GenieX CLI by following the
[installation guide](https://refactored-happiness-4qyl9vn.pages.github.io/),
then run your first model:

```bash
geniex infer ai-hub-models/Qwen3-4B
```

GenieX downloads the pre-compiled bundle, prepares it for your device, and
drops you into an interactive chat session — no manual runtime install, bundle
assembly, or prompt formatting required.

To bring your own GGUF model from Hugging Face:

```bash
geniex infer <org>/<repo>-GGUF
```

For installation, the Python and Android SDKs, the OpenAI-compatible local
server, the list of tested models, runtime/compute-unit options, and
troubleshooting, see the **[GenieX documentation](https://refactored-happiness-4qyl9vn.pages.github.io/)**.

## Export your own model bundle

Some models — notably the Llama family, which requires gated Hugging Face
access — are not distributed as pre-compiled assets and must be exported from
source with [qai-hub-models](https://github.com/qualcomm/ai-hub-models). This
export step is an upstream workflow not covered by the GenieX
documentation.

See **[export.md](export.md)** for the full export walkthrough. Once you have a
bundle, register it with GenieX using `geniex pull --local-path` and run it with
`geniex infer`, for example:

```bash
geniex pull local/llama_v3_2_3b_instruct --local-path ./geniex_bundle
geniex infer local/llama_v3_2_3b_instruct
```

See [export.md](export.md#run-the-exported-bundle-with-geniex) for the
platform-specific path-resolution details.

## Sample chat applications

GenieX powers higher-level applications as well as the CLI:

- [GenieX Android ChatApp](https://github.com/qualcomm/ai-hub-apps/tree/main/apps/geniex_chat_android)

## Questions

Please post questions on the
[Qualcomm AI Hub Slack channel](https://aihub.qualcomm.com/community/slack).
