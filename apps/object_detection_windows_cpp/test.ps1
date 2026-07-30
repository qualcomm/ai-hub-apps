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
$Labels = "$ScriptDir\assets\models\labels.txt"
$Image = "$ScriptDir\assets\images\kitchen.jpg"
if (-not (Test-Path $Model)) {
    Write-Error "Model not found at $Model. Fetch the model first."
    exit 1
}
if (-not (Test-Path $Labels)) {
    Write-Error "Labels not found at $Labels. Fetch the model first."
    exit 1
}

$OutImage = "$env:TEMP\detection_output.jpg"
Write-Host "Running ObjectDetection on $Image"
$output = & $Exe --model $Model --labels $Labels --image $Image --output_image $OutImage | Out-String
Write-Host $output
if ($LASTEXITCODE -ne 0) {
    Write-Error "ObjectDetection exited with code $LASTEXITCODE."
    exit 1
}

# The app prints "Number of objects: <n>" after NMS. Require at least one detection.
if ($output -notmatch "Number of objects:\s*(\d+)") {
    Write-Error "Could not find object count in output."
    exit 1
}
if ([int]$Matches[1] -lt 1) {
    Write-Error "No objects detected in $Image."
    exit 1
}

Write-Host "ObjectDetection test passed."
