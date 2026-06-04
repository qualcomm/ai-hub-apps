# Contributing to QAI Hub Apps

This guide covers dev environment setup, repo architecture, app conventions, testing infrastructure, and how to add new apps.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Repo Architecture](#repo-architecture)
- [App Structure by Platform](#app-structure-by-platform)
- [info.yaml Schema](#infoyaml-schema)
- [Shared Scripts](#shared-scripts-apps_sharedscripts)
- [Testing Infrastructure](#testing-infrastructure)
- [Code Style and Linting](#code-style-and-linting)
- [Adding a New App](#adding-a-new-app)

---

## Development Setup

### Prerequisites

- Python 3.10+
- [git-lfs](https://git-lfs.com/)
- Docker

### Environment

```bash
# Create and activate the dev virtual environment
bash tools/setup_env.sh
source qaiha-dev/bin/activate
```

Available flags for `setup_env.sh`:

| Flag | Description |
|------|-------------|
| `--with-cli` | Also install the `qai-hub-apps` CLI package (`cli/`) |
| `--with-qdc-sdk` | Download and install the Qualcomm Device Cloud SDK wheel |
| `--venv <path>` | Custom venv path (default: `qaiha-dev`) |
| `--python <exe>` | Python executable to use (default: `python3`) |
| `--extras <name>` | Extras to install: `dev` (default) or `precommit` |

### Pre-commit Hooks

```bash
pip install pre-commit && pre-commit install

# Run all hooks manually
pre-commit run --all-files

# Separate line-ending check
pre-commit run --config .pre-commit-line-ending-check.yaml --all-files
```

---

## Repo Architecture

```
ai-hub-apps/
├── apps/               # Sample applications (one subdirectory per app)
│   └── _shared/        # Shared scripts, Gradle config, Python utilities
├── cli/                # qai-hub-apps CLI package (fetch, list, info)
├── tools/
│   ├── python/         # qai_hub_apps_test: bundlers, builders, QDC, CI scripts
│   └── ci/             # CI utility scripts (download-qdc-wheel, build reports)
├── .github/workflows/  # GitHub Actions CI/CD
└── pyproject.toml      # Root Python config (pydoclint, mypy)
```

### The CLI ↔ Registry Flow

```
generate_registry.py
  └─ bundles apps → cli/qai_hub_apps/registry.yaml (published to S3 on release)

qai-hub-apps fetch <app_id> --model <model_id>
  └─ downloads app source (from S3 or bundled dev source) + model asset
     └─ dev installs: bundles on-the-fly via bundlers/ (no S3 needed)
```

### The Bundlers

The bundler packages an app into a self-contained directory or zip for distribution. See [`tools/python/qai_hub_apps_test/bundlers/README.md`](tools/python/qai_hub_apps_test/bundlers/README.md) for full details.

- **Android bundler** — deep copies the app directory resolving all symlinks, copies and rewrites shell script `source` lines to bundle-local paths (same as Python bundler), inlines version variables from `versions.env` into `build.gradle`, and empties `common.gradle`
- **Python bundler** — scans app source with AST to find `qai_hub_apps_utils` imports, copies only needed SDK modules, merges `requirements.txt`, and rewrites shell script `source` lines to bundle-local paths
- **Shell bundler** — transitively copies referenced shared scripts into `scripts/` and copies `versions.env`

### Key Internal Packages

| Package | Location | Purpose |
|---------|----------|---------|
| `qai_hub_apps_test` | `tools/python/` | Bundlers, builders, QDC, config parsers — see [`tools/python/README.md`](tools/python/README.md) |
| `qai_hub_apps` (CLI) | `cli/` | End-user CLI — see [`cli/README.md`](cli/README.md) |

---

## App Structure by Platform

### Naming Convention

App directories follow the pattern: `{app_name}_{platform}[_{language}]`

| Token | Values |
|-------|--------|
| `{app_name}` | `image_classification`, `object_detection`, `semantic_segmentation`, `super_resolution`, `chatapp`, `whisper`, `stable_diffusion`, `mediapipe_hand_gesture`, … |
| `{platform}` | `android`, `windows`, `ubuntu` |
| `{language}` | `cpp`, `py` (omitted for Java/Kotlin Android apps) |

**Examples:** `image_classification_android`, `whisper_windows_py`, `super_resolution_windows_cpp`, `mediapipe_hand_gesture_ubuntu_py`

---

### Android Apps

```
<app>_android/
├── info.yaml
├── README.md
├── build.gradle            # uses ext vars from common.gradle (ANDROID_NDK_VERSION, TF_LITE_VERSION, etc.)
├── settings.gradle         # must include: rootProject.name = 'app'
├── gradle.properties
├── install_build.sh        # sources android_utils.sh, calls install_android_sdk
├── _shared/
│   └── android/
│       └── common.gradle   # symlink → apps/_shared/android/common.gradle
└── src/
    ├── main/
    │   ├── java/com/quicinc/<app>/
    │   ├── assets/         # model files (.tflite)
    │   └── AndroidManifest.xml
    └── androidTest/
        └── java/com/quicinc/<app>/
            └── <App>Test.java   # UI Automator instrumented tests
```

**Key conventions:**
- `build.gradle` must use ext vars from `common.gradle` — no hardcoded SDK/NDK versions
- `settings.gradle` must set `rootProject.name = 'app'` so APKs are named `app-debug.apk`
- `_shared/android/common.gradle` is a **symlink** pointing to `apps/_shared/android/common.gradle`
- Tests live in `src/androidTest/` and use UI Automator

---

### Ubuntu Python Apps

```
<app>_ubuntu_py/
├── info.yaml
├── README.md
├── install_runtime.sh      # sources shared scripts (python_utils, pip_utils, etc.)
├── test.sh                 # entry point run on-device by run_linux.sh (e.g. runs main.py)
├── requirements.txt
└── <source files>
```

---

### Windows Apps (C++ or Python)

```
<app>_windows_cpp/          <app>_windows_py/
├── info.yaml               ├── info.yaml
├── README.md               ├── README.md
├── <App>.sln               ├── install_runtime.ps1
├── <App>.vcxproj           ├── requirements.txt
├── vcpkg.json              └── <source files>
└── src/
```

---

## info.yaml Schema

Every app must have an `info.yaml`. Copy from a similar app and adjust.

### Mandatory fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier — must match the directory name |
| `name` | string | Human-readable display name |
| `headline` | string | One-line UI description |
| `description` | string | Full UI description |
| `app_type` | `android` \| `windows` \| `ubuntu` | Platform type |
| `runtime` | `tflite` \| `onnx` \| `genie` \| … | ML runtime |
| `status` | `published` \| `unpublished` | Set `unpublished` until ready |
| `languages` | list | e.g. `['Java']`, `['Python']`, `['Java', 'C++']` |
| `related_models` | list | All compatible model IDs (used for CLI fetch + testing) |
| `precisions` | list | e.g. `[float]`, `[w4a16]` |
| `supported_devices` | list | Device names — **must match a key in `HUB_DEVICE_TO_QDC_DEVICE_MAP`** in `tools/python/qai_hub_apps_test/qdc/qdc_jobs.py` |
| `license_type` | `bsd-3-clause` | License (almost always BSD-3) |
| `license_url` | string | Link to LICENSE file |
| `app_repo_relative_path` | string | Path within the public repo, usually `id` (e.g. `image_classification_android`) |

### Optional fields

| Field | Default | Description |
|-------|---------|-------------|
| `model_file_paths` | — | **Required** (if not using `model_file_dir`) — relative destination paths for each downloaded model file; all paths must share the same parent directory |
| `model_file_dir` | — | **Required** (if not using `model_file_paths`) — single directory to extract all model files into; mutually exclusive with `model_file_paths` |
| `include_in_cli` | `true` | Set `false` to exclude from CLI registry (e.g. external apps) |
| `skip_test` | — | String reason to skip CI testing |
| `app_repo_url` | — | Explicit GitHub URL (overrides `app_repo_relative_path` — use for external repos) |

---

## Shared Scripts (`apps/_shared/scripts/`)

Sourced by app install scripts — never executed directly.

### `versions.env`

The **single source of truth** for all version pins:

```
TF_LITE_VERSION, TF_LITE_SUPPORT_VERSION, QNN_VERSION
JAVA_SDK_VERSION, GRADLE_VERSION
ANDROID_NDK_VERSION, ANDROID_COMPILE_API, ANDROID_TARGET_API, ANDROID_MIN_API
QAIRT_SDK_VERSION, QAIRT_SDK_FULL_VERSION
PYTHON_VERSION, ONNX_RUNTIME_VERSION
```

Android `build.gradle` reads these at build time via `common.gradle`. Shell scripts source them via `load_versions.sh`.

### Bash utilities (`.sh`)

| Script | Functions | Description |
|--------|-----------|-------------|
| `android_utils.sh` | `install_android_sdk [--force]` | Installs SDKMAN → Java → Gradle → Android SDK + NDK. Also exports `ANDROID_HOME`, `JAVA_HOME`, `GRADLE_HOME`. |
| `qairt_utils.sh` | `install_qairt [--force]` | Downloads and extracts QAIRT SDK. Exports `QAIRT_ROOT`, `QAIRT_PATH`. |
| `apt_utils.sh` | `install_apt_pkg <pkg>`, `install_apt_pkgs <pkg>...` | Idempotent apt installation |
| `python_utils.sh` | `install_python` | Installs Python + venv + uv |
| `pip_utils.sh` | `install_pip_deps [--venv <dir>] <req>...` | Creates `.venv` and installs via uv |

### PowerShell utilities (`.ps1`)

| Script | Functions |
|--------|-----------|
| `qairt_utils.ps1` | `Install-Qairt [-Force]` — exports `$env:QAIRT_ROOT`, `$env:QAIRT_PATH` |
| `winget_utils.ps1` | `Install-WingetPackage` |
| `python_utils.ps1` | `Install-Python` |
| `pip_utils.ps1` | `Install-PipDeps` |

### `NON_INTERACTIVE` environment variable

Set `NON_INTERACTIVE=true` (done automatically in Docker/CI) to auto-accept SDK licenses without prompting. Leave unset for interactive developer use.

---

## Testing Infrastructure

### How testing works by app type

Testing follows three stages, implemented in `tools/python/qai_hub_apps_test/test/device_apps_test.py`:

1. **Fetch** — `qai-hub-apps fetch <app_id> --model <model_id>` downloads app source + model asset
2. **Build** — platform-specific build step (see below)
3. **On-device** — submits the built app to Qualcomm Device Cloud (QDC) for execution on real hardware

#### Android apps

- **Build:** Docker container with `BUILD_TYPE=build` runs `install_build.sh` (installs Android SDK) then `gradle assembleDebug assembleAndroidTest`. APKs are copied back via `docker cp`.
- **Tests:** UI Automator instrumented tests in `src/androidTest/java/`. The test runner (`run_android.py`) installs the APKs via `adb` and runs the instrumentation suite.
- **Test content:** Tests should wake the device, dismiss the keyguard, exercise the main inference flow, and assert on results. See `image_classification_android` as the reference implementation.

#### Ubuntu Python apps

- **Build:** No build step — Python apps run directly from the fetched source.
- **Tests:** `test.sh` is executed on the QDC device via `run_linux.sh`.

#### Windows apps

- Work in progress.

### Running tests locally

```bash
# Full setup (CLI + QDC SDK required)
bash tools/setup_env.sh --with-cli --with-qdc-sdk
source qaiha-dev/bin/activate
cd tools/python

# Stage 1 only — validate fetch works (no QDC needed)
pytest -m device_test --model-selection first --test-stage fetch

# Stages 1 + 2 — fetch + build (Docker required for Android)
pytest -m device_test --model-selection first --test-stage build

# All stages — full on-device test (QDC token required)
pytest -m device_test --model-selection first --test-stage all \
  --qdc-token $QDC_API_TOKEN

# Test a specific app
pytest -m device_test --model-selection first --test-stage fetch \
  -k image_classification_android
```

---

## Code Style and Linting

Run:

```bash
pre-commit run --all-files
pre-commit run --config .pre-commit-line-ending-check.yaml --all-files
```

---

## Adding a New App

### 1. Create the directory

```bash
mkdir apps/<app_name>_<platform>[_<language>]
```

### 2. Write `info.yaml`

Copy from a similar app (e.g. `image_classification_android/info.yaml`). Set:
- `status: unpublished` until the app is ready
- `include_in_cli: false` until bundling and testing are verified
- `supported_devices` — check `HUB_DEVICE_TO_QDC_DEVICE_MAP` in `tools/python/qai_hub_apps_test/qdc/qdc_jobs.py` for valid device names

### 3. Add platform-specific files

Follow the structure in [App Structure by Platform](#app-structure-by-platform).

**Android-specific:**

```bash
# Create the _shared/android symlink
mkdir -p apps/<app>/_shared/android
ln -sf "../../../_shared/android/common.gradle" apps/<app>/_shared/android/common.gradle

# Update build.gradle to use ext vars from common.gradle
# Move 'apply from' to the top (before android {} and dependencies {})
# Replace hardcoded versions: ndkVersion ANDROID_NDK_VERSION, etc.

# Add rootProject.name to settings.gradle
echo "rootProject.name = 'app'" >> apps/<app>/settings.gradle
```

### 4. Add `install_build.sh` (Android)

```bash
#!/usr/bin/env bash
set -euo pipefail
source ../_shared/scripts/android_utils.sh
install_android_sdk
```

### 5. Add instrumented tests (Android)

Create `src/androidTest/java/com/quicinc/<app>/<App>Test.java`. See `image_classification_android` as the reference. Tests should:
- Wake the device and dismiss the keyguard
- Launch the app via `Intent`
- Interact with the UI (select image, tap Run)
- Wait for results and assert expected output

### 6. Run bundler unit tests and CLI tests

```bash
# Bundler unit tests — validates info.yaml loads, bundling works for all app types
cd tools/python
pytest -m bundler_unit -v

# CLI tests — validates registry generation, fetch, and CLI commands
cd cli
pytest -v
```

Both test suites must pass before opening a PR.

### 7. Register in CLI

Check the app appears in the registry dry-run (requires `--with-cli` from the dev setup):

```bash
python -m qai_hub_apps_test.scripts.generate_registry \
  --output_dir /tmp/reg_test
```

When satisfied, set `include_in_cli: true` and `status: published` in `info.yaml`.

### 8. Run on-device tests locally

```bash
cd tools/python

# Full on-device test (requires QDC API token)
pytest -m device_test --model-selection first --test-stage all \
  --qdc-token $QDC_API_TOKEN \
  -k <your_app_id>
```


## Questions?

Open a GitHub issue or pull request at [qualcomm/ai-hub-apps](https://github.com/qualcomm/ai-hub-apps) repo.
