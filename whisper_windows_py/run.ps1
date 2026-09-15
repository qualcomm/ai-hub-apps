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

# With no --audio-file, demo.py streams live from the default input device.
Write-Host "Transcribing live from the microphone. Speak after the model loads."
& $VenvPython "$ScriptDir\demo.py" @args
exit $LASTEXITCODE
