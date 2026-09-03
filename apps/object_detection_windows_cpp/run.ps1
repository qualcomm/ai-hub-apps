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

# Detect on a live feed from the first camera, unless the caller picked their own
# input source (another camera with --camera <index>, or a still image with
# --image <path>). The app takes exactly one of the two.
$InputArgs = @()
if (-not (($args -contains "--camera") -or ($args -contains "--image"))) {
    $InputArgs = @("--camera", "0")
}

# Annotated frames are shown in a window; press any key in it to stop.
& $Exe --model $Model --labels $Labels @InputArgs @args
exit $LASTEXITCODE
