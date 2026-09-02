# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$Exe = "$ScriptDir\ARM64\Release\Classification.exe"
if (-not (Test-Path $Exe)) {
    Write-Error "Classification.exe not found at $Exe. Run install_build.ps1 first."
    exit 1
}

$Model = "$ScriptDir\assets\models\classification.onnx"
$Image = "$ScriptDir\assets\images\keyboard.jpg"
# Expected class for the sample image (keyboard.jpg).
$Expected = "computer keyboard"
if (-not (Test-Path $Model)) {
    Write-Error "Model not found at $Model. Fetch the model first."
    exit 1
}

$OutImage = "$env:TEMP\classification_output.png"
Write-Host "Running Classification on $Image"
$output = & $Exe --model $Model --image $Image --output_image $OutImage | Out-String
Write-Host $output
if ($LASTEXITCODE -ne 0) {
    Write-Error "Classification exited with code $LASTEXITCODE."
    exit 1
}

if ($output -notmatch "classification = $([regex]::Escape($Expected))") {
    Write-Error "Expected classification '$Expected' not found in output."
    exit 1
}

Write-Host "Classification test passed."
