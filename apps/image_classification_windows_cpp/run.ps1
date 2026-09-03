# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$Exe = "$ScriptDir\ARM64\Release\Classification.exe"
if (-not (Test-Path $Exe)) {
    Write-Error "Classification.exe not found at $Exe. Build the app first."
    exit 1
}

$Model = "$ScriptDir\assets\models\classification.onnx"
# Classifying a live camera frame is not worth it at this model's frame rate, so
# this runs on the sample image that ships with the app.
$Image = "$ScriptDir\assets\images\keyboard.jpg"

# With no --output_image, the app opens the labelled image in a window instead of
# writing it to disk. Press any key in that window to exit.
& $Exe --model $Model --image $Image @args
exit $LASTEXITCODE
