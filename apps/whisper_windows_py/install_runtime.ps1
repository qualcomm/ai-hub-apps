# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Install runtime dependencies for the Whisper demo.
# All dependencies ship native ARM64 wheels, so this installs whatever Python
# winget provides by default on the host (ARM64 on Snapdragon).
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

. ..\_shared\scripts\load_versions.ps1
. ..\_shared\scripts\winget_utils.ps1
. ..\_shared\scripts\pip_utils.ps1
. ..\_shared\scripts\python_utils.ps1

# Override the shared default (3.11) for win-arm64: pyyaml (pulled in transitively
# via transformers) only publishes win_arm64 wheels for cp312+. On 3.11 uv falls
# back to a from-source build that needs MSVC, which the Python app flow does not
# provision -- so the install breaks whenever the device lacks a stray toolchain.
# Set after the dot-source block: pip_utils/python_utils each re-source
# load_versions.ps1, which would otherwise reset this back to 3.11.
$PYTHON_VERSION = "3.12"

Install-Python
# ffmpeg is required for reading audio files.
Install-WingetPackage -Id "Gyan.FFmpeg" -ExtraArgs @("--source", "winget")

Install-PipDeps -Packages @("-r", "$ScriptDir\requirements.txt")
