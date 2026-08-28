# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$Exe = "$ScriptDir\ARM64\Release\ObjectDetection.exe"
if (-not (Test-Path $Exe)) {
    Write-Error "ObjectDetection.exe not found at $Exe. Build the app first."
    exit 1
}

$Model = "$ScriptDir\assets\models\detection.onnx"
$Labels = "$ScriptDir\assets\models\labels.txt"
$Image = "$ScriptDir\assets\images\kitchen.jpg"
$OutImage = "$env:TEMP\detection_output.jpg"

& $Exe --model $Model --labels $Labels --image $Image --output_image $OutImage @args
exit $LASTEXITCODE
