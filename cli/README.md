# qai-hub-apps

CLI for browsing and downloading [Qualcomm® AI Hub Apps](https://aihub.qualcomm.com/apps) —
sample apps for deploying ML models on Qualcomm® devices.

## Installation

```bash
pip install qai-hub-apps
```

## Quick Start

```bash
qai-hub-apps list                          # browse available apps
qai-hub-apps info whisper_windows_py       # inspect an app
qai-hub-apps fetch whisper_windows_py      # download source to current directory
```

## Commands

### list

List all available apps.

```bash
qai-hub-apps list
```

```
+----------------------------------------------------------------------+
|                        Qualcomm® AI Hub Apps                         |
+---------------------+------------------+---------------+-------------+
| ID                  | Name             | Domain        | Languages   |
+---------------------+------------------+---------------+-------------+
| whisper_windows_py  | Whisper Windows  | Audio         | Python      |
| stable_diffusion_py | Stable Diffusion | Generative AI | Python, C++ |
+---------------------+------------------+---------------+-------------+
Total: N apps
```

### info

Show details for an app.

```bash
qai-hub-apps info <app_id>
```

```
+=============================================================================+
|                               Whisper Windows                               |
|  https://github.com/qualcomm/ai-hub-apps/tree/main/apps/whisper_windows_py  |
+=============================================================================+

+---------------------------+
|          Headline         |
+---------------------------+
| Speech to text on Windows |
+---------------------------+

+-----------------------------------+
|            Description            |
+-----------------------------------+
| Run Whisper on-device using ONNX. |
+-----------------------------------+

+------------------------------------+
|              Metadata              |
+---------------+--------------------+
| ID            | whisper_windows_py |
| Type          | windows            |
| Languages     | Python             |
| Runtime       | onnx               |
| Domain        | Audio              |
| Use Case      | Speech Recognition |
| Precision     | float              |
| Models        | whisper_base       |
| AI Hub Models | 0.30.0             |
+---------------+--------------------+
```

### fetch

Download and extract an app's source to a local directory.

```bash
# Download app source only
qai-hub-apps fetch <app_id>

# Download app source + a model to download for a specific chipset
qai-hub-apps fetch <app_id> --model <model_id> --chipset <chipset>

# Download app source + bundle a locally-exported model
qai-hub-apps fetch <app_id> --model <path/to/model>
```

| Flag | Description |
|------|-------------|
| `--output-dir PATH` | Output directory (default: current directory) |
| `--model MODEL_ID_OR_PATH` | Model to bundle: a model ID to download (must be supported by the app), or a path to a locally-exported model (directory or `.zip`). Use `--model-id`/`--model-path` to be explicit |
| `--model-id MODEL_ID` | Model ID to download (must be supported by the app) |
| `--model-path PATH` | Path to a locally-exported model (directory or `.zip`) |
| `--chipset CHIPSET` | Target chipset for the model download. Only applies to a downloaded model (`--model <id>` or `--model-id`); not valid with a local model path |

`--model`, `--model-id`, and `--model-path` are mutually exclusive.

On success, the path to the fetched app directory is printed.

**Example — fetch app with model:**

```bash
qai-hub-apps fetch stable_diffusion_windows_py --model stable_diffusion_v2_1 --chipset qualcomm-snapdragon-x-elite
```

## Logging

Control verbosity with a global flag or the `QAI_HUB_APPS_LOG_LEVEL` environment
variable. The command-line flag takes precedence over the environment variable.
The default level is `info`.

| Flag | Description |
|------|-------------|
| `--log-level {debug,info,error}` | Set log verbosity |
| `-v`, `--verbose` | Shorthand for `--log-level debug` |
| `-q`, `--quiet` | Shorthand for `--log-level error` |

```bash
qai-hub-apps -v list
QAI_HUB_APPS_LOG_LEVEL=debug qai-hub-apps list
```
