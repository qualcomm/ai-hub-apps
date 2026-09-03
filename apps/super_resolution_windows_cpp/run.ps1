# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$Exe = "$ScriptDir\ARM64\Release\SuperResolution.exe"
if (-not (Test-Path $Exe)) {
    Write-Error "SuperResolution.exe not found at $Exe. Build the app first."
    exit 1
}

$Model = "$ScriptDir\assets\models\super_resolution.onnx"
# Upscaling live camera frames is not worth it at this model's frame rate, so this
# runs on the sample image that ships with the app.
$Image = "$ScriptDir\assets\images\Doll.jpg"

# With no --output_image, the app opens the input and upscaled images side by side
# in a window instead of writing the result to disk. Press any key in that window
# to exit.
& $Exe --model $Model --image $Image @args
exit $LASTEXITCODE
