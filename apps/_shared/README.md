<!-- THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY. -->
<!-- Regenerate with: python -m qai_hub_apps._internal.scripts.run_codegen -->
# Shared App Utilities (`apps/_shared/`)

Code shared across apps. Each description is the file's `Description:` header
line, or the first line of the module docstring for Python.

## Scripts (`scripts/`)

Sourced by app install/test scripts and the generated build/launch scripts, never executed directly. Bundling copies the ones an app uses into its `scripts/` directory.

| File | Description |
|------|-------------|
| [`adb_utils.ps1`](scripts/adb_utils.ps1) | PowerShell version of adb_utils.sh. |
| [`adb_utils.sh`](scripts/adb_utils.sh) | adb helpers for the generated Android launch.sh. |
| [`android_utils.sh`](scripts/android_utils.sh) | Android build toolchain installation utilities. |
| [`apt_utils.sh`](scripts/apt_utils.sh) | Ubuntu/Linux apt package installation utilities. |
| [`docker_utils.ps1`](scripts/docker_utils.ps1) | PowerShell version of docker_utils.sh. |
| [`docker_utils.sh`](scripts/docker_utils.sh) | Docker helpers for the generated build and launch scripts. |
| [`exit_codes.ps1`](scripts/exit_codes.ps1) | PowerShell version of exit_codes.sh. |
| [`exit_codes.sh`](scripts/exit_codes.sh) | Exit codes an app's launch/run/test script uses to signal the CLI. |
| [`imsdk_utils.sh`](scripts/imsdk_utils.sh) | Fetches the per-device CDI and env auxiliary files an IMSDK app's container |
| [`interactive.ps1`](scripts/interactive.ps1) | PowerShell version of interactive.sh. |
| [`interactive.sh`](scripts/interactive.sh) | User-consent gating for install/system-mutating commands. |
| [`load_versions.ps1`](scripts/load_versions.ps1) | PowerShell version of load_versions.sh. |
| [`load_versions.sh`](scripts/load_versions.sh) | Loads versions.env (KEY=VALUE) into the current shell environment. |
| [`msvc_utils.ps1`](scripts/msvc_utils.ps1) | Windows MSVC (Visual Studio Build Tools) installation utilities. |
| [`pip_utils.ps1`](scripts/pip_utils.ps1) | PowerShell version of pip_utils.sh. |
| [`pip_utils.sh`](scripts/pip_utils.sh) | Ubuntu/Linux pip/venv installation utilities. |
| [`python_utils.ps1`](scripts/python_utils.ps1) | PowerShell version of python_utils.sh. |
| [`python_utils.sh`](scripts/python_utils.sh) | Ubuntu/Linux Python installation utilities. |
| [`qairt_utils.ps1`](scripts/qairt_utils.ps1) | PowerShell version of qairt_utils.sh. |
| [`qairt_utils.sh`](scripts/qairt_utils.sh) | Ubuntu/Linux QAIRT SDK installation utilities. |
| [`qcom_host_utils.sh`](scripts/qcom_host_utils.sh) | Qualcomm host (driver) package installation for Ubuntu devices. |
| [`retry.ps1`](scripts/retry.ps1) | PowerShell version of retry.sh. |
| [`retry.sh`](scripts/retry.sh) | Retry wrapper for flaky commands. |
| [`runtime_utils.ps1`](scripts/runtime_utils.ps1) | PowerShell version of runtime_utils.sh. |
| [`runtime_utils.sh`](scripts/runtime_utils.sh) | Runtime-install marker helpers for the generated launch scripts. |
| [`sudo.sh`](scripts/sudo.sh) | Sets $SUDO for privileged commands on a host or in a container. |
| [`vcpkg_utils.ps1`](scripts/vcpkg_utils.ps1) | Windows vcpkg + NuGet provisioning utilities for ONNX C++ apps. |
| [`versions.env`](scripts/versions.env) | Canonical SDK/toolchain versions for AI Hub Apps. |
| [`winget_utils.ps1`](scripts/winget_utils.ps1) | Windows winget package installation utilities. |

## Python (`python/`)

`qai_hub_apps_utils/` is installed as `qai-hub-apps-utils`; each module's third-party dependencies are the pip extra listed below.

| File | Extra | Description |
|------|-------|-------------|
| [`compare_images.py`](python/compare_images.py) |  | Compare two images using mean absolute difference. |
| [`qai_hub_apps_utils/bbox_processing.py`](python/qai_hub_apps_utils/bbox_processing.py) | `[bbox_processing]` | Bounding-box math: IoU, NMS, and box format/rotation conversions. |
| [`qai_hub_apps_utils/config.py`](python/qai_hub_apps_utils/config.py) | `[config]` | An app's own configuration, read from its ``info.yaml``. |
| [`qai_hub_apps_utils/draw.py`](python/qai_hub_apps_utils/draw.py) | `[draw]` | OpenCV drawing helpers for keypoints, connections, and boxes. |
| [`qai_hub_apps_utils/fps.py`](python/qai_hub_apps_utils/fps.py) |  | Frames-per-second counter for processing loops. |
| [`qai_hub_apps_utils/image_processing.py`](python/qai_hub_apps_utils/image_processing.py) | `[image_processing]` | Image and coordinate transforms: resize/pad, affines, and denormalization. |
| [`qai_hub_apps_utils/imsdk.py`](python/qai_hub_apps_utils/imsdk.py) |  | Building blocks for IMSDK (``qimsdk``) GStreamer apps. |
| [`qai_hub_apps_utils/input_devices.py`](python/qai_hub_apps_utils/input_devices.py) | `[input_devices]` | Helpers for selecting live capture devices on the host. |
| [`qai_hub_apps_utils/onnxruntime_qnn.py`](python/qai_hub_apps_utils/onnxruntime_qnn.py) | `[onnxruntime_qnn]` | Shared helpers for running ONNX models via the ONNX Runtime QNN Execution Provider. |
| [`qai_hub_apps_utils/platform.py`](python/qai_hub_apps_utils/platform.py) | `[platform]` | Access the target device that ``qai-hub-apps run`` configured for this run. |
| [`qai_hub_apps_utils/quantization.py`](python/qai_hub_apps_utils/quantization.py) | `[quantization]` | Quantize and dequantize tensors with zero points and scales. |
| [`qai_hub_apps_utils/webui.py`](python/qai_hub_apps_utils/webui.py) | `[webui]` | Shared Flask-based live-video web UI for Ubuntu-py apps. |
| [`setup.py`](python/setup.py) |  | Dynamic optional-dependency discovery for qai-hub-apps-utils. |

## Other shared files

| File | Description |
|------|-------------|
| [`android/common.gradle`](android/common.gradle) | Shared Gradle config: loads versions.env into ext for every Android app. |
| [`android/image_preprocessing_helpers/ImageProcessing.java`](android/image_preprocessing_helpers/ImageProcessing.java) | Bitmap preprocessing helpers (aspect-preserving resize and pad). |
| [`android/tflite_helpers/AIHubDefaults.java`](android/tflite_helpers/AIHubDefaults.java) | TFLite delegate and CPU thread defaults matching AI Hub's profiling setup. |
| [`android/tflite_helpers/TFLiteHelpers.java`](android/tflite_helpers/TFLiteHelpers.java) | TFLite model loading and interpreter creation with QNN NPU / GPU delegates. |
| [`cpp/PromptHandler.cpp`](cpp/PromptHandler.cpp) | Implementation of PromptHandler.hpp. |
| [`cpp/PromptHandler.hpp`](cpp/PromptHandler.hpp) | Builds chat prompts with the model's system/user/assistant tags. |
| [`msvc/versions.props`](msvc/versions.props) | Shared MSVC toolchain versions for AI Hub Apps Windows C++ apps. |
