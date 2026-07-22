# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Windows MSVC (Visual Studio Build Tools) installation utilities.
#
# Functions:
#   Install-MSVC
#       Install Visual Studio 2022 Build Tools with the C++ ARM64 toolchain
#       via winget (if not already installed), then locate MSBuild and export
#       its path as $env:MSBUILD_EXE for callers to invoke.
#
# Usage: . msvc_utils.ps1
# ---------------------------------------------------------------------
$_MsvcUtilsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$_MsvcUtilsDir\winget_utils.ps1"
. "$_MsvcUtilsDir\interactive.ps1"

function _Install-MSVC {
    Install-WingetPackage -Id "Microsoft.VisualStudio.2022.BuildTools" -ExtraArgs @(
        "--source", "winget",
        "--override",
        "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.ARM64 --includeRecommended"
    )

    # Locate MSBuild via vswhere and export it for callers. Finds the first (latest) entry.
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $msbuild = & $vswhere -latest -products Microsoft.VisualStudio.Product.BuildTools `
        -requires Microsoft.Component.MSBuild `
        -find "MSBuild\**\Bin\MSBuild.exe" | Select-Object -First 1
    if (-not $msbuild) {
        Write-Error "MSBuild not found via vswhere."
        exit 1
    }
    $env:MSBUILD_EXE = $msbuild
    Write-Host "::done::MSBuild at $env:MSBUILD_EXE"
}

function Install-MSVC {
    Invoke-WithConsent -Description "Install Visual Studio 2022 Build Tools (C++ ARM64 toolchain) via winget" -Action {
        _Install-MSVC
    }
}
