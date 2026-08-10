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
. "$_VcpkgUtilsDir\interactive.ps1"
. "$_VcpkgUtilsDir\retry.ps1"

# Default install location used only when vcpkg is not already on PATH.
$DEFAULT_VCPKG_ROOT = "C:\vcpkg"


function Install-Vcpkg {
    Invoke-WithConsent -Description "Install vcpkg (clone + bootstrap; installs Git via winget if missing)" -Action {
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
                Invoke-WithRetry -Description "git clone vcpkg" -Action {
                    # Clear any partial clone; git refuses to clone into a non-empty dir (exit 128).
                    Remove-Item -Recurse -Force $vcpkgRoot -ErrorAction SilentlyContinue
                    & $git clone https://github.com/microsoft/vcpkg $vcpkgRoot
                }
            }
            Invoke-WithRetry -Description "bootstrap vcpkg" -Action {
                & "$vcpkgRoot\bootstrap-vcpkg.bat" -disableMetrics
            }
        }
        $env:VCPKG_ROOT = $vcpkgRoot
        & $vcpkgExe integrate install
        Write-Host "::done::vcpkg at $vcpkgRoot"
    }
}

function Install-NuGet {
    Invoke-WithConsent -Description "Install the NuGet CLI via winget" -Action {
        Install-WingetPackage -Id "Microsoft.NuGet" -ExtraArgs @("--source", "winget")
        $nuget = Resolve-InstalledExe -Name "nuget.exe"
        if (-not $nuget) {
            Write-Error "nuget.exe not found after install."
            exit 1
        }
        $env:NUGET_EXE = $nuget
        Write-Host "::done::NuGet at $env:NUGET_EXE"
    }
}
