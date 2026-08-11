# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Build-time setup for ChatApp: installs Visual Studio Build Tools (C++ ARM64
# workload) and the QAIRT SDK, and exports $env:MSBUILD_EXE for the caller to
# invoke MSBuild.
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:QAIHA_APP_ROOT = $ScriptDir

. ..\_shared\scripts\load_versions.ps1
. ..\_shared\scripts\winget_utils.ps1
. ..\_shared\scripts\msvc_utils.ps1
. ..\_shared\scripts\qairt_utils.ps1

# Install Visual Studio Build Tools (C++ ARM64); sets $env:MSBUILD_EXE.
Install-MSVC
Install-WingetPackage -Id "Microsoft.Git" -ExtraArgs @("--source", "winget")

# Install QAIRT and point the build at it via QNN_SDK_ROOT.
Install-Qairt
$env:QNN_SDK_ROOT = $env:QAIRT_PATH
