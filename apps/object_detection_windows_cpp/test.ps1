# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$Exe = "$ScriptDir\ARM64\Release\ObjectDetection.exe"
if (-not (Test-Path $Exe)) {
    Write-Error "ObjectDetection.exe not found at $Exe. Run install_build.ps1 first."
    exit 1
}

$Model = "$ScriptDir\assets\models\detection.onnx"
$Image = "$ScriptDir\assets\images\kitchen.jpg"
if (-not (Test-Path $Model)) {
    Write-Error "Model not found at $Model. Fetch the model first."
    exit 1
}

$OutImage = "$env:TEMP\detection_output.jpg"
Write-Host "Running ObjectDetection on $Image"
& $Exe --model $Model --image $Image --output_image $OutImage
if ($LASTEXITCODE -ne 0) {
    Write-Error "ObjectDetection exited with code $LASTEXITCODE."
    exit 1
}

Write-Host "ObjectDetection test passed."
