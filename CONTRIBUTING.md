# Contributing to QAI Hub Apps

This guide covers dev environment setup, repo architecture, app conventions, testing infrastructure, and how to add new apps.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Repo Architecture](#repo-architecture)
- [App Structure by Platform](#app-structure-by-platform)
- [info.yaml Schema](#infoyaml-schema)
- [Shared Scripts](#shared-scripts-apps_sharedscripts)
- [Shared Python Utilities](#shared-python-utilities-qai_hub_apps_utils)
- [Building Apps](#building-apps)
- [Experimental CLI Features](#experimental-cli-features)
- [Testing Infrastructure](#testing-infrastructure)
- [Code Style and Linting](#code-style-and-linting)
- [Adding a New App](#adding-a-new-app)

---

## Development Setup

### Prerequisites

- Python 3.10+
- [git-lfs](https://git-lfs.com/)
- Docker

> [!IMPORTANT]
> The expected contributor environment is a **Linux or macOS host**. Some dev dependencies (e.g. `onnxruntime` via `qai_hub_models`) lack `win-arm64` wheels, so the full tooling — including `generate_registry` — may not install cleanly on native Windows.
>
> **On Windows, enable Developer Mode and set `git config --global core.symlinks true` *before cloning*.** Otherwise symlinks are checked out as plain text files and builds will fail.

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

The Java/Kotlin hooks (`google-java-format`, `ktlint`) need a JVM toolchain.
Bootstrapping with the `precommit` extra provisions it automatically into
`.lint-tools/` (git-ignored):

```bash
bash tools/setup_env.sh --extras precommit     # Linux
. tools/setup_env.ps1 -Extras precommit        # Windows
```

Tool versions are pinned in `tools/versions.env`.

---

## Repo Architecture

```
ai-hub-apps/
├── apps/               # Sample applications (one subdirectory per app)
│   └── _shared/        # Shared scripts, Gradle config, Python utilities
├── cli/                # qai-hub-apps CLI package (fetch, list, info)
├── tools/
│   ├── python/         # qai_hub_apps_test: bundlers, script generators, QDC, CI scripts
│   └── ci/             # CI utility scripts (download-qdc-wheel, build reports)
├── .github/workflows/  # GitHub Actions CI/CD
└── pyproject.toml      # Root Python config (pydoclint, mypy)
```

### The CLI ↔ Registry Flow

The CLI is organized around three stages — **fetch → build → run** — mirroring the on-device test flow:

```
generate_registry.py
  └─ bundles apps → cli/qai_hub_apps/registry.yaml (published to S3 on release)

qai-hub-apps fetch <app_id> --model <model_id>
  └─ downloads app source (from S3 or bundled dev source) + model asset
     └─ dev installs: bundles on-the-fly via bundlers/ (no S3 needed)
     └─ writes a provenance manifest (qai_hub_apps.json) into the fetched dir

qai-hub-apps build <app_id_or_path>          # experimental (see below)
  └─ fetches the app if needed, then execs the bundled build.sh / build.ps1
```

### Fetch provenance manifest

Every `fetch` writes a `qai_hub_apps.json` manifest into the fetched app directory
(`App._write_manifest`, `cli/qai_hub_apps/registry/base.py`) recording the versions
that produced the bundle:

| Field | Source |
|-------|--------|
| `cli_version` | `qai_hub_apps.__version__` |
| `qai_hub_models_version` | the app's `qaihm_version`, else the installed `qai_hub_models` version |
| `registry_version` | `Registry.load().version` |
| `fetched_at` | UTC ISO-8601 timestamp of the fetch |

This is provenance only — which app/model *content* was fetched already lives in the
model's `metadata.json`, so the manifest deliberately does not duplicate it. It lets
support/CI reconstruct exactly which toolchain produced a given fetched app.

### The Bundlers

The bundler packages an app into a self-contained directory or zip for distribution. See [`tools/python/qai_hub_apps_test/bundlers/README.md`](tools/python/qai_hub_apps_test/bundlers/README.md) for full details.

- **Android bundler** — deep copies the app directory resolving all symlinks, copies and rewrites shell script `source` lines to bundle-local paths (same as Python bundler), inlines version variables from `versions.env` into `build.gradle`, and empties `common.gradle`
- **Python bundler** — scans app source with AST to find `qai_hub_apps_utils` imports, copies only the needed shared modules (see [Shared Python Utilities](#shared-python-utilities-qai_hub_apps_utils)), merges `requirements.txt`, and rewrites shell script `source` lines to bundle-local paths
- **Shell bundler** — transitively copies referenced shared scripts into `scripts/` and copies `versions.env`

### Key Internal Packages

| Package | Location | Purpose |
|---------|----------|---------|
| `qai_hub_apps_test` | `tools/python/` | Bundlers, script generators (registry + build), QDC, config parsers — see [`tools/python/README.md`](tools/python/README.md) |
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

> [!IMPORTANT]
> CI validates *every* app's `info.yaml` regardless of `status`, so a half-filled file fails CI even for an unpublished app.

### Mandatory fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier — must match the directory name |
| `name` | string | Human-readable display name |
| `headline` | string | One-line UI description |
| `description` | string | Full UI description |
| `domain` | string | Model domain (e.g. `Computer Vision`) — copy from a similar app |
| `use_case` | string | Use-case label (e.g. `Image Classification`) — copy from a similar app |
| `app_type` | `android` \| `windows` \| `ubuntu` | Platform type |
| `os` | mapping | Target operating system — `name` (`Android` \| `Windows` \| `Ubuntu`) and `version` (e.g. `11+`, `24.04+`) |
| `runtime` | closed enum | `tflite` \| `onnx` \| `genie` \| `precompiled_qnn_onnx` \| … — must be an existing `TargetRuntime` value (a new one needs an upstream `qai_hub_models` change) |
| `status` | closed enum | Lifecycle stage — see [App Status](#app-status). Set `unpublished` until ready |
| `languages` | list | e.g. `['Java']`, `['Python']`, `['Java', 'C++']` |
| `related_models` | list | All compatible model IDs (used for CLI fetch + testing). The first entry is the seed model tested on-device. CI fails if any entry has no AI Hub Models asset matching the app's `runtime`/`precisions`/`supported_devices` |
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
| `disable_cli_model_fetch` | `false` | Set `true` for apps that **download their model at runtime** instead of bundling a model asset. Mutually exclusive with `model_file_paths` / `model_file_dir`. Without it, `fetch` fails trying to download a non-existent asset. |
| `deprecation_notice` | — | Message shown by the CLI for `deprecated` apps. If unset, a default deprecation message is used |
| `skip_test` | — | String reason to skip CI testing |
| `skip_related_models_verify` | — | String reason to skip `related_models` verification for this app |
| `app_repo_url` | — | Explicit GitHub URL (overrides `app_repo_relative_path` — use for external repos) |
| `qaihm_version` | — | Explicit Qualcomm AI Hub Models CLI version to use when dowloading model assets. By default, the installed version of the models-cli is used, if provided, this version will be used instead |

### App Status

The `status` field is an app's lifecycle stage. It controls where the app appears: on the public **website**, in the bundled **CLI** registry, and in the **CI test** set.

| `status` | Website | CLI | Tested in CI | Meaning |
|----------|:-------:|:---:|:------------:|---------|
| `unpublished` | ✗ | ✗ | ✓ | Work in progress — not released yet, but exercised by CI |
| `published` | ✓ | ✓ | ✓ | Live everywhere |
| `published_website_only` | ✓ | ✗ | ✗ | On the website but not CLI-fetchable (e.g. external apps) |
| `deprecated` | ✓ | ✓ | ✓ | Like `published`, but the CLI shows a deprecation notice (`deprecation_notice`, or a default) |

Deprecated apps are removed completely from the repo once their deprecation period ends.

Registry generation selects which apps to include via `--scope` (see [Regenerate and commit the registry](#7-regenerate-and-commit-the-registry)):

| `--scope` | Includes | Used for |
|-----------|----------|----------|
| `production` (default) | `published` + `deprecated` | The released CLI registry (only scope allowed with `--build_and_upload`) |
| `test` | `unpublished` + `published` + `deprecated` | CI test runs (`skip_test` still gates per-app) |
| `all` | Every app, including `published_website_only` | Inspection / completeness |

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

> [!IMPORTANT]
> For using any shared scripts, **`QAIHA_APP_ROOT`** (the app directory) must be set
> before using the utilities.

#### Per-app version overrides

To override versions for an app, add them to `apps/<id>/versions.override.env`
(creating the file if needed) — same `KEY="VALUE"` format; its keys win over the
global `versions.env`. Example: `apps/whisper_windows_py/versions.override.env` pins `PYTHON_VERSION="3.12"` for win-arm64.

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

## Shared Python Utilities (`qai_hub_apps_utils`)

Python apps share common helpers (drawing, bounding-box math, image pre/post-processing, FPS counting, a minimal web UI, …) through the `qai_hub_apps_utils` package (distribution name `qai-hub-apps-utils`), which lives in `apps/_shared/python/`. Instead of copy-pasting this code into every app, import it from the package — the bundler ([The Bundlers](#the-bundlers)) copies only the modules an app actually imports (plus their transitive imports) into the standalone bundle, so end users never need the internal repo.

Browse `apps/_shared/python/qai_hub_apps_utils/` for the available modules and their (numpy-style) docstrings. `apps/posenet_ubuntu_py/` is a good end-to-end reference — it imports the `draw`, `webui`, `fps`, `image_processing`, and `quantization` modules.

### Using it in an app

1. **Import the module(s)** you need in your app source, e.g.:

   ```python
   from qai_hub_apps_utils.fps import FpsCounter
   from qai_hub_apps_utils.image_processing import resize_pad
   ```

2. **Don't list `qai-hub-apps-utils` in the app's `requirements.txt`.** The bundler copies the imported modules directly into the bundle, so only their third-party dependencies need to be installed at runtime (see next point). For local development, install the package (and any per-module extras) in editable mode:

   ```bash
   # base package
   pip install -e apps/_shared/python/

   # with a module's optional deps (extras are auto-discovered per module)
   pip install -e 'apps/_shared/python/[draw]'
   pip install -e 'apps/_shared/python/[full]'   # union of all module extras
   ```

3. **The bundler wires dependencies automatically.** Each module declares its third-party deps in `apps/_shared/python/qai_hub_apps_utils/requirements/requirements-<module>.txt`. When the bundler copies a module, it merges that file into the bundle's `requirements.txt` alongside the app's own.

### Adding a new shared utility

1. Add a module under `apps/_shared/python/qai_hub_apps_utils/` (e.g. `my_helper.py`) with numpy-style docstrings.
2. If it has third-party dependencies, create `qai_hub_apps_utils/requirements/requirements-my_helper.txt` listing them. This file is what the bundler merges and what becomes the `[my_helper]` pip extra — no `setup.py` change is needed (extras are discovered dynamically).
3. Import it from your app as `from qai_hub_apps_utils.my_helper import ...`. The AST-based collector picks it up automatically at bundle time; transitive imports of other `qai_hub_apps_utils` modules are followed too.

---

## Building Apps

Building an app is driven by a per-app **build script** — `build.sh` (bash) for
Ubuntu/Android apps, `build.ps1` (PowerShell) for Windows apps — that ships inside
the app bundle. The CLI's `build` command is a thin wrapper: it resolves/fetches the
app, then execs the bundled script. Keeping the build logic *in the script* (not in
the CLI) means a fetched app is buildable on its own — a user who runs `fetch` can
build it directly with `bash build.sh` without the CLI installed.

### Scripts are generated, not hand-written

Build scripts are produced from Jinja2 templates by
`tools/python/qai_hub_apps_test/scripts/generate_build_scripts.py` — the same pattern
as `generate_registry.py`. **Do not hand-edit a committed `build.sh` / `build.ps1`**;
edit the template and regenerate.

```bash
# Generate for every in-scope app (default scope: production)
python -m qai_hub_apps_test.scripts.generate_build_scripts

# CI regenerates with --scope test and fails if the result differs from what's committed
python -m qai_hub_apps_test.scripts.generate_build_scripts --scope test

# Single app
python -m qai_hub_apps_test.scripts.generate_build_scripts --app_id posenet_ubuntu_py
```

The generated script (and the shared scripts it sources, e.g. `interactive.sh`) is
carried into the bundle by the [shell bundler](tools/python/qai_hub_apps_test/bundlers/README.md).

### Templates by app type

The generator picks a template from the app's `app_type` + `languages`
(`_plan()` in `generate_build_scripts.py`). Templates live under
`tools/python/qai_hub_apps_test/scripts/templates/`:

| App type | Template | Output | What it does |
|----------|----------|--------|--------------|
| Android | `android/build_sh.j2` | `build.sh` | Docker build → `gradle assembleDebug assembleAndroidTest` → `docker cp` the APKs out |
| Windows C++ | `windows/build_ps1.j2` | `build.ps1` | MSBuild the `.sln` for ARM64 (native via `-NoDocker`, or in a Windows container) |
| Ubuntu Python | `noop_build_sh.j2` | `build.sh` | No-op — prints "Nothing to build; run it directly" |
| Windows Python | `noop_build_ps1.j2` | `build.ps1` | No-op (PowerShell variant) |

An app whose `app_type` / `languages` combination matches none of the above (e.g. an
Android app with no build recipe) makes the generator **fail loudly** (`SystemExit`)
rather than silently skip it.

### Common script conventions

- **Args:** `--no-docker` / `--docker` (docker is the default) and `--clean`. The
  PowerShell equivalents are `-NoDocker` and `-Clean`.
- **`--clean`** tears down prior build state — host-side outputs, the Docker image,
  and any leftover container — then rebuilds the image from scratch (`--no-cache`).
  Without it the image is left in place so the next build reuses its cache.
- **Container lifetime:** the transient build container is removed on exit (bash
  `trap`, PowerShell `finally`); the *image* is kept for cache reuse. Image/container
  names are derived from a hash of the app directory so two copies of the same app in
  different directories never collide.
- **Leftover container:** if a container from a prior run still holds the name (e.g.
  after a hard kill), the script asks the user before removing it, via
  `require_consent` (bash) / `Invoke-WithConsent` (PowerShell) from
  `apps/_shared/scripts/interactive.{sh,ps1}`.
- **Error handling:** bash uses `set -euo pipefail`; PowerShell sets
  `$ErrorActionPreference = "Stop"` **and** an `Assert-Success` helper checked after
  each native command (because `Stop` does not abort on a non-zero native exit code,
  only on cmdlet errors).

### CI-only Docker build args (`QC_INTERNAL_HOST`)

The Android Docker build passes `REGISTRY_PREFIX` (an internal registry mirror) and
`INSTALL_QUALCOMM_CA` (Qualcomm CA certs) **only** when `QC_INTERNAL_HOST=1`. Those
resources are reachable only from the Qualcomm internal network — CI runners or a
corp-network machine. The workflow (`test-app.yaml`) sets `QC_INTERNAL_HOST: '1'`;
off the internal network the Dockerfile defaults apply (public base image, no CA
injection), so external contributors can build unchanged.

### CI keeps generated scripts in sync

`build-and-test.yaml`'s **"Check generated files are up to date"** job regenerates
both the registry and the build scripts (`--scope test`) and fails if the working
tree differs — so a template change with stale committed output is caught in CI, just
like `registry.yaml`. Regenerate and commit whenever you touch a template or add an
app.

---

## Experimental CLI Features

Some CLI features are **experimental** — opt-in, unstable, and hidden from end users
until they graduate. The `build` command is currently experimental.

The gate lives in `cli/qai_hub_apps/experimental/__init__.py` and is driven entirely
by the `QAI_HUB_APPS_EXPERIMENTAL` environment variable (`1`/`true`/`yes`/`on`):

- `is_enabled()` — returns whether the env var is set to a truthy value.
- `add_experimental_parser(subparsers, name, **kwargs)` — registers an experimental
  subcommand **only** when experimental features are enabled; otherwise it returns a
  throwaway `ArgumentParser` so `main.py` can keep configuring the parser
  unconditionally. When enabled, the command's `help` text is tagged `[experimental]`.

Because the subparser is only *registered* when the env var is set, the command is
invisible in `--help` (and unrunnable) for normal users, with no `argparse` internals
poked. To use or test it:

```bash
QAI_HUB_APPS_EXPERIMENTAL=1 qai-hub-apps build <app_id_or_path>
```

The on-device build stage (`device_apps_test.py`) sets this env var automatically for
its `build` subprocess, so CI exercises the experimental command without any global
opt-in.

---

## Testing Infrastructure

### How testing works by app type

Testing follows three stages, implemented in `tools/python/qai_hub_apps_test/test/device_apps_test.py`:

1. **Fetch** — `qai-hub-apps fetch <app_id> --model <model_id> --output-dir <fetched_dir>` downloads app source + model asset
2. **Build** — `qai-hub-apps build --app-path <fetched_dir>` runs the app's bundled build script (see [Building Apps](#building-apps))
3. **On-device** — submits the built app to Qualcomm Device Cloud (QDC) for execution on real hardware

The build stage shells out to the real CLI (`build` is experimental, so the test sets
`QAI_HUB_APPS_EXPERIMENTAL=1`), which execs the shipped `build.sh` / `build.ps1`. It
builds *in place* from the already-fetched directory, so no model is re-fetched. This
means the test exercises the exact build path an end user gets. A missing/failed build
script fails the stage loudly (no silent skip).

#### Android apps

- **Build:** the generated `build.sh` runs a Docker build (`BUILD_TYPE=build`, which runs `install_build.sh` to install the Android SDK) then `gradle assembleDebug assembleAndroidTest` inside the container; APKs are copied back via `docker cp`. Note the instrumented test only compiles under `assembleAndroidTest` — a green `assembleDebug` does **not** mean the test compiles, so always build both locally.
- **Tests:** UI Automator instrumented tests in `src/androidTest/java/`. The test runner (`run_android.py`) installs the APKs via `adb` and runs the instrumentation suite.
- **Test content:** Tests should wake the device, dismiss the keyguard, exercise the main inference flow, and assert on results. See `apps/chatapp_android/src/androidTest/java/com/quicinc/chatapp/ChatAppTest.java` for a concrete reference (it also asserts on TTFT / tokens-per-sec performance metrics).

#### Ubuntu Python apps

- **Build:** the generated `build.sh` is a no-op (prints "Nothing to build") — Python apps run directly from the fetched source.
- **Tests:** `test.sh` is executed on the QDC device via `run_linux.sh`.

#### Windows apps

- Work in progress.

### Running tests locally

```bash
# Full setup (CLI + QDC SDK required)
bash tools/setup_env.sh --with-cli --with-qdc-sdk
source qaiha-dev/bin/activate
cd tools/python/qai_hub_apps_test

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
- `status: unpublished` until the app is ready (see [App Status](#app-status))
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
> [!IMPORTANT]
> **On Windows, this symlink is only checked out correctly if `core.symlinks=true` was set before cloning** — see the note in [Development Setup](#development-setup).

### 4. Add `install_build.sh` (Android)

```bash
#!/usr/bin/env bash
set -euo pipefail
source ../_shared/scripts/android_utils.sh
install_android_sdk
```

### 5. Add instrumented tests (Android)

Create `src/androidTest/java/com/quicinc/<app>/<App>Test.java`. See `apps/chatapp_android/src/androidTest/java/com/quicinc/chatapp/ChatAppTest.java` as the reference. Note the test only compiles under `gradle assembleAndroidTest` (not `assembleDebug`). Tests should:
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

### 7. Regenerate and commit the registry

Adding (or changing) an app means the bundled registry must be regenerated **and committed** — CI fails if `cli/qai_hub_apps/registry.yaml` is out of date. Run (requires `--with-cli` from the dev setup):

```bash
python -m qai_hub_apps_test.scripts.generate_registry --output_dir cli/qai_hub_apps/
```

Then commit the updated `cli/qai_hub_apps/registry.yaml`. When the app is ready to publish, set `status: published` in `info.yaml` and regenerate again.

To inspect the output without touching the bundled file, point `--output_dir` at a scratch path (e.g. `/tmp/reg_test`). To preview the full test set (including unpublished apps), add `--scope test` (see [App Status](#app-status)).

### 8. Generate and commit the build script

Every in-scope app needs a committed `build.sh` / `build.ps1` — CI's "Check generated files" job regenerates them with `--scope test` and fails if any is missing or stale (see [Building Apps](#building-apps)). Generate for your new app and commit the result:

```bash
python -m qai_hub_apps_test.scripts.generate_build_scripts --app_id <your_app_id> --scope test
```

If the generator errors that there's no template for your app's `app_type` / `languages`, that combination has no build recipe yet — add a template and wire it into `_plan()` in `generate_build_scripts.py`.

### 9. Run on-device tests locally

```bash
cd tools/python/qai_hub_apps_test

# Full on-device test (requires QDC API token)
pytest -m device_test --model-selection first --test-stage all \
  --qdc-token $QDC_API_TOKEN -vv -s \
  -k <your_app_id>
```


## Questions?

Open a GitHub issue or pull request at [qualcomm/ai-hub-apps](https://github.com/qualcomm/ai-hub-apps) repo.
