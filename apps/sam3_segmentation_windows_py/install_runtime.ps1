# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Install runtime dependencies for the SAM3 segmentation demo.
# All dependencies ship native ARM64 wheels, so this installs whatever Python
# winget provides by default on the host (ARM64 on Snapdragon).
#
# The public CLIP tokenizer (openai/clip-vit-base-patch32) is fetched by main.py
# on first run and needs no HuggingFace token. To use it offline, pass a local
# tokenizer.json via --tokenizer.
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

. ..\_shared\scripts\load_versions.ps1
. ..\_shared\scripts\winget_utils.ps1
. ..\_shared\scripts\pip_utils.ps1
. ..\_shared\scripts\python_utils.ps1

Install-Python
Install-PipDeps -Packages @("-r", "$ScriptDir\requirements.txt")

# Editable-install the shared SDK when running from a git clone (the CLI vendors it in otherwise).
$SharedPythonDir = Join-Path $ScriptDir "..\_shared\python"
if ((Test-Path $SharedPythonDir) -and -not (Test-Path (Join-Path $ScriptDir "qai_hub_apps_utils"))) {
    Install-PipDeps -Packages @("-e", $SharedPythonDir)
}
