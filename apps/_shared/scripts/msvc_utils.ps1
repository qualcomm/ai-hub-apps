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

# Locate MSBuild via vswhere. Finds the first (latest) entry; $null if not installed.
function Find-MSBuild {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return $null }
    return & $vswhere -latest -products Microsoft.VisualStudio.Product.BuildTools `
        -requires Microsoft.Component.MSBuild `
        -find "MSBuild\**\Bin\MSBuild.exe" | Select-Object -First 1
}

function _Install-MSVC {
    Install-WingetPackage -Id "Microsoft.VisualStudio.2022.BuildTools" -ExtraArgs @(
        "--no-upgrade", "--source", "winget",
        "--override",
        "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.ARM64 --includeRecommended"
    )
}

function Install-MSVC {
    $msbuild = Find-MSBuild
    if ($msbuild) {
        Write-Host "::skip::Visual Studio 2022 Build Tools already installed"
    } else {
        Invoke-WithConsent -Description "Install Visual Studio 2022 Build Tools (C++ ARM64 toolchain) via winget" -Action {
            _Install-MSVC
        }
        $msbuild = Find-MSBuild
        if (-not $msbuild) {
            Write-Error "MSBuild not found via vswhere."
            exit 1
        }
    }
    # Export for callers whether or not we installed anything.
    $env:MSBUILD_EXE = $msbuild
    Write-Host "::done::MSBuild at $env:MSBUILD_EXE"
}
