# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Windows winget package installation utilities.
#
# Functions:
#   Install-WingetPackage -Id <package_id> [-ExtraArgs <string[]>]
#       Install a winget package if it is not already installed.
#   Install-WingetPackages -Ids <string[]>
#       Install multiple winget packages, each idempotently.
#
# Usage: . winget_utils.ps1
# ---------------------------------------------------------------------
$_WingetUtilsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$_WingetUtilsDir\load_versions.ps1"
. "$_WingetUtilsDir\interactive.ps1"
. "$_WingetUtilsDir\retry.ps1"

# add winget to PATH if installed but not yet on PATH
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Error "winget not found. Please install it from https://learn.microsoft.com/en-us/windows/package-manager/winget/ and re-run."
    exit 1
}

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

function _Install-WingetPackage {
    param(
        [string]$Id,
        [string[]]$ExtraArgs = @()
    )
    $list = winget list --id $Id --exact --accept-source-agreements 2>&1
    if ($LASTEXITCODE -eq 0 -and ($list -match [regex]::Escape($Id))) {
        Write-Host "::skip::$Id"
    } else {
        Write-Host "::step::Installing $Id"
        Invoke-WithRetry -Description "winget install $Id" -Action {
            winget install --id $Id --exact --silent --accept-package-agreements --accept-source-agreements @ExtraArgs
        }
        Write-Host "::done::$Id"
    }
}

function Install-WingetPackage {
    param(
        [string]$Id,
        [string[]]$ExtraArgs = @()
    )
    Invoke-WithConsent -Description "Install winget package '$Id'" -Action {
        _Install-WingetPackage -Id $Id -ExtraArgs $ExtraArgs
    }
}

function Install-WingetPackages {
    param(
        [string[]]$Ids
    )
    Invoke-WithConsent -Description "Install winget packages: $($Ids -join ' ')" -Action {
        foreach ($id in $Ids) {
            _Install-WingetPackage -Id $id
        }
    }
}
