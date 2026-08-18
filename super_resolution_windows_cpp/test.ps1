# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$Exe = "$ScriptDir\ARM64\Release\SuperResolution.exe"
if (-not (Test-Path $Exe)) {
    Write-Error "SuperResolution.exe not found at $Exe. Run install_build.ps1 first."
    exit 1
}

$Model = "$ScriptDir\assets\models\super_resolution.onnx"
$Image = "$ScriptDir\assets\images\Doll.jpg"
if (-not (Test-Path $Model)) {
    Write-Error "Model not found at $Model. Fetch the model first."
    exit 1
}

$OutImage = "$env:TEMP\super_resolution_output.png"
Write-Host "Running SuperResolution on $Image"
& $Exe --model $Model --image $Image --output_image $OutImage
if ($LASTEXITCODE -ne 0) {
    Write-Error "SuperResolution exited with code $LASTEXITCODE."
    exit 1
}

Write-Host "SuperResolution test passed."
