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

# With no --prompt, demo.py asks for one; with no --output-dir it opens the
# generated image in the default viewer.
& $VenvPython "$ScriptDir\demo.py" @args
exit $LASTEXITCODE
