# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$VenvPython = "$ScriptDir\.venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment not found. Run install_runtime.ps1 first."
    exit 1
}

# Test image is hosted on S3, not shipped with the app, so the app source stays
# lean and the CLI-fetched app has no stray assets.
$ImageUrl = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-apps/apps/sam3_segmentation_windows_py/test/kitchen.jpg"
$ImagePath = "$env:TEMP\kitchen.jpg"

Write-Host "Downloading test image..."
Invoke-WebRequest -Uri $ImageUrl -OutFile $ImagePath

$Output = "$env:TEMP\sam3_output.png"

Write-Host "Running SAM3 segmentation (prompt: 'cup')..."
& $VenvPython "$ScriptDir\main.py" `
    --image $ImagePath `
    --text-prompts "cup" `
    --output $Output
if ($LASTEXITCODE -ne 0) {
    Write-Error "FAIL: segmentation exited with code $LASTEXITCODE."
    exit $LASTEXITCODE
}

if (Test-Path $Output) {
    $size = (Get-Item $Output).Length
    if ($size -lt 1024) {
        Write-Error "FAIL: output image is suspiciously small ($size bytes)."
        exit 1
    }
    Write-Host "PASS: wrote overlay to $Output."
} else {
    Write-Error "FAIL: no output image produced."
    exit 1
}
