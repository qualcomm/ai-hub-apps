# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Install runtime dependencies for the Whisper demo.
# All dependencies ship native ARM64 wheels, so this installs whatever Python
# winget provides by default on the host (ARM64 on Snapdragon).
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

. "$PSScriptRoot\scripts\load_versions.ps1"
. "$PSScriptRoot\scripts\winget_utils.ps1"
. "$PSScriptRoot\scripts\pip_utils.ps1"
. "$PSScriptRoot\scripts\python_utils.ps1"

Install-Python
# ffmpeg is required for reading audio files.
Install-WingetPackage -Id "Gyan.FFmpeg" -ExtraArgs @("--source", "winget")

Install-PipDeps -Packages @("-r", "$ScriptDir\requirements.txt")
