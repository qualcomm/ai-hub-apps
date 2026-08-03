# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Install runtime dependencies for the Stable Diffusion demo.
# All dependencies ship native ARM64 wheels, so this installs whatever Python
# winget provides by default on the host (ARM64 on Snapdragon).
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

. "$PSScriptRoot\scripts\load_versions.ps1"
. "$PSScriptRoot\scripts\winget_utils.ps1"
. "$PSScriptRoot\scripts\pip_utils.ps1"
. "$PSScriptRoot\scripts\python_utils.ps1"

Install-Python
Install-PipDeps -Packages @("-r", "$ScriptDir\requirements.txt")


# Pre-fetch the CLIP tokenizer here (with visible progress) rather than letting
# demo.py hit the Hugging Face Hub silently on first run -- on a slow/unstable
# connection that hidden fetch can hang for a long time with no log output.
Write-Host "::step::Pre-caching CLIP tokenizer"
$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
& $VenvPython -c "from transformers import CLIPTokenizer; CLIPTokenizer.from_pretrained('sd2-community/stable-diffusion-2-1', subfolder='tokenizer')"
Write-Host "::done::tokenizer cache"
