# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$VenvPython = "$ScriptDir\.venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment not found. Run install_runtime.ps1 first."
    exit 1
}

$OutputDir = "$env:TEMP\stable_diffusion"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

& $VenvPython "$ScriptDir\demo.py" --prompt "A cat sitting on a bench" --output-dir $OutputDir @args
exit $LASTEXITCODE
