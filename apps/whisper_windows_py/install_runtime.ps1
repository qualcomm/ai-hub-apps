# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# qai-hub-models requires x64 Python (ARM64 Python is not supported).
$ErrorActionPreference = "Stop"

. ..\_shared\scripts\load_versions.ps1
. ..\_shared\scripts\winget_utils.ps1
. ..\_shared\scripts\pip_utils.ps1

$majorMinor = ($PYTHON_VERSION -split "\.")[ 0..1] -join "."
Install-WingetPackage -Id "Python.Python.$majorMinor" -ExtraArgs @("--architecture", "x64", "--source", "winget")
Install-WingetPackage -Id "Microsoft.Git" -ExtraArgs @("--source", "winget")
# ffmpeg is required for reading audio files.
Install-WingetPackage -Id "Gyan.FFmpeg" -ExtraArgs @("--source", "winget")

# Install app dependencies using x64 Python explicitly.
# onnxruntime and onnxruntime-qnn conflict — uninstall both then reinstall qnn.
Install-PipDeps -Python "py -$majorMinor-64" -Packages @("qai-hub-models[whisper_base]~=0.48.0")
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $VenvPython -m pip uninstall --yes onnxruntime onnxruntime-qnn
Install-PipDeps -Python "py -$majorMinor-64" -Packages @("onnxruntime-qnn==1.24.4")
