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

# Segmentation of a live camera frame is not worth it at this model's frame rate,
# so this runs on the same sample image the test uses. It is hosted on S3 rather
# than shipped with the app, so the app source stays lean.
$ImageUrl = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-apps/apps/sam3_segmentation_windows_py/test/kitchen.jpg"
$ImagePath = "$env:TEMP\kitchen.jpg"
if (-not (Test-Path $ImagePath)) {
    Write-Host "Downloading sample image..."
    Invoke-WebRequest -Uri $ImageUrl -OutFile $ImagePath
}

# Segment a few things that are in the sample image, unless the caller asked for
# their own prompts.
$PromptArgs = @()
if (-not ($args | Where-Object { $_ -like "--text-prompts*" })) {
    $PromptArgs = @("--text-prompts", "cup,broccoli,bottle")
}

# With no --output, main.py opens the overlay in the default image viewer.
& $VenvPython "$ScriptDir\main.py" --image $ImagePath @PromptArgs @args
exit $LASTEXITCODE
