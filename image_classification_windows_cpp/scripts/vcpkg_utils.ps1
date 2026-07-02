# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Windows vcpkg + NuGet provisioning utilities for ONNX C++ apps.
#
# Functions:
#   Install-Vcpkg
#       Clone + bootstrap vcpkg (if not present), run 'vcpkg integrate install',
#       and export $env:VCPKG_ROOT. Manifest-mode dependencies (vcpkg.json) are
#       then restored automatically during the MSBuild step.
#   Install-NuGet
#       Install the NuGet CLI via winget (if not already installed) and export
#       its path as $env:NUGET_EXE for callers to invoke.
#
# Usage: . vcpkg_utils.ps1
# ---------------------------------------------------------------------
$_VcpkgUtilsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$_VcpkgUtilsDir\winget_utils.ps1"

# Default install location used only when vcpkg is not already on PATH.
$DEFAULT_VCPKG_ROOT = "C:\vcpkg"

# winget adds the install dir to the machine PATH, but the running process won't see it; so
# temporarily reload PATH to resolve the exe, then restore the original PATH.
function Resolve-InstalledExe {
    param([string]$Name)
    $originalPath = $env:PATH
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH", "User")
    $exe = (Get-Command $Name -ErrorAction SilentlyContinue).Source
    $env:PATH = $originalPath
    return $exe
}

function Install-Vcpkg {
    # Prefer a vcpkg already on PATH; only clone to the default location if none is found.
    $vcpkgExe = Resolve-InstalledExe -Name "vcpkg.exe"
    if ($vcpkgExe) {
        $vcpkgRoot = Split-Path -Parent $vcpkgExe
        Write-Host "::skip::vcpkg already installed at $vcpkgRoot"
    } else {
        $vcpkgRoot = $DEFAULT_VCPKG_ROOT
        $vcpkgExe = "$vcpkgRoot\vcpkg.exe"
        Write-Host "::step::Installing vcpkg"
        if (-not (Test-Path $vcpkgRoot)) {
            # vcpkg is fetched via git clone, so ensure Git is available first.
            $git = (Get-Command git -ErrorAction SilentlyContinue).Source
            if (-not $git) {
                Install-WingetPackage -Id "Microsoft.Git" -ExtraArgs @("--source", "winget")
                $git = Resolve-InstalledExe -Name "git.exe"
                if (-not $git) {
                    Write-Error "git not found after install."
                    exit 1
                }
            }
            & $git clone https://github.com/microsoft/vcpkg $vcpkgRoot
        }
        & "$vcpkgRoot\bootstrap-vcpkg.bat" -disableMetrics
    }
    $env:VCPKG_ROOT = $vcpkgRoot
    & $vcpkgExe integrate install
    Write-Host "::done::vcpkg at $vcpkgRoot"
}

function Install-NuGet {
    Install-WingetPackage -Id "Microsoft.NuGet" -ExtraArgs @("--source", "winget")
    $nuget = Resolve-InstalledExe -Name "nuget.exe"
    if (-not $nuget) {
        Write-Error "nuget.exe not found after install."
        exit 1
    }
    $env:NUGET_EXE = $nuget
    Write-Host "::done::NuGet at $env:NUGET_EXE"
}
